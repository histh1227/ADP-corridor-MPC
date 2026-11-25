"""LiDAR 스캔 기반 장애물 인식 및 안전 여유 검사 모듈.

기존 cmpc_berlin_tt.py의 _perceive, _boxes, _mem_clear, _view_of, _body_clr,
_plan_clear, _fwd_clear 로직을 분리하고, 비공개 센서 노이즈 대응용 스캔 필터링 기능을 추가했다.
"""
import math
import numpy as np

from common.numba_kernels import _boxes_k, _carve_k, _cluster_k, _hits_k, _merge_k
from common.vehicle_params import ANG_INC, ANG_MIN, BODY_PTS, FRONT_PTS, WB, wrap_angle


class ObstacleDetector:
    def __init__(self, track, obs_edt: float = 0.09, r_max: float = 9.0, link: float = 0.22,
                 carve: bool = True, obs_pad_s: float = 0.06, obs_pad_d: float = 0.02,
                 ds: float = 0.2, n_stations: int = 40,
                 creep_m_obs: float = 0.04, creep_m_wall: float = 0.03,
                 scan_median_win: int = 0):
        self.track = track
        self.obs_edt = obs_edt
        self.r_max = r_max
        self.link = link
        self.carve = carve
        self.obs_pad_s = obs_pad_s
        self.obs_pad_d = obs_pad_d
        self.ds = ds
        self.n_stations = n_stations
        self.creep_m_obs = creep_m_obs
        self.creep_m_wall = creep_m_wall
        self.scan_median_win = int(scan_median_win)

        ang = ANG_MIN + ANG_INC * np.arange(1080)
        self.ca, self.sa = np.cos(ang), np.sin(ang)

        # Numba 커널 워밍업
        t = self.track
        _hits_k(np.ones(1080), 0.0, 0.0, 1.0, 0.0, self.ca, self.sa, 9.0, t.ox, t.oy, t.res, t.W, t.H,
                t.edt, 0.09, t.mres, t.MW)
        c1 = np.ones(1, dtype=np.int64)
        _merge_k(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int16), *(np.zeros(0, dtype=np.float32),) * 4,
                 np.zeros(0, dtype=np.bool_), c1, t.cell_x, t.cell_y, t.cell_s, t.cell_d, t.cell_ok)
        _carve_k(np.ones(1, dtype=np.int16), np.ones(1, dtype=np.float32), c1 * 500, np.ones(1080))
        z32 = np.zeros(2, dtype=np.float32)
        for ok_ in (np.ones(2, dtype=np.bool_), np.zeros(2, dtype=np.bool_)):
            _boxes_k(np.array([0, 1]), ok_, z32, z32, z32, z32, t.MW, t.W, 0.0, float(t.L), 9.0, 0.22)
        _cluster_k(np.zeros(3), np.zeros(3), 0.1)

        self.reset()

    def reset(self):
        self.mem_clear()
        self.blind_until = -1.0

    def mem_clear(self):
        self.active = np.zeros(0, dtype=np.int64)
        self.act_mem = np.zeros(0, dtype=np.int16)
        self.act_x, self.act_y, self.act_s, self.act_d = (np.zeros(0, dtype=np.float32) for _ in range(4))
        self.act_ok = np.zeros(0, dtype=np.bool_)

    def filter_scan_noise(self, scan: np.ndarray) -> np.ndarray:
        """스캔의 -inf 정규화 및 선택적 스파이크 노이즈 억제 필터 (Perception 개선)."""
        if np.isneginf(scan).any():
            scan = np.where(np.isneginf(scan), 0.0, scan)
        if self.scan_median_win >= 3:
            # 3-point median으로 단독 이상치(salt-and-pepper 노이즈)만 제거
            s_pad = np.pad(scan, 1, mode='edge')
            stacked = np.stack([s_pad[:-2], s_pad[1:-1], s_pad[2:]], axis=0)
            valid = np.isfinite(stacked).all(axis=0)
            med = np.median(stacked, axis=0)
            scan = np.where(valid, med, scan)
        return scan

    def _view_of(self, lx: float, ly: float, yaw: float):
        dx = self.act_x - lx
        dy = self.act_y - ly
        rng = np.hypot(dx, dy)
        brg = wrap_angle(np.arctan2(dy, dx) - yaw)
        return rng, brg, np.rint((brg - ANG_MIN) / ANG_INC).astype(np.int64)

    def perceive(self, scan: np.ndarray, lx: float, ly: float, yaw: float, t: float):
        """지도 거리변환(EDT) 기반 장애물 셀 추출 및 자유 공간 Ray Carving."""
        tr = self.track
        cy, sy = math.cos(yaw), math.sin(yaw)
        if t >= self.blind_until:
            cells = _hits_k(scan, float(lx), float(ly), cy, sy, self.ca, self.sa, float(self.r_max),
                            tr.ox, tr.oy, tr.res, tr.W, tr.H, tr.edt, float(self.obs_edt), tr.mres, tr.MW)
            if cells.size:
                (self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                 self.act_ok) = _merge_k(self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                                         self.act_ok, cells, tr.cell_x, tr.cell_y, tr.cell_s, tr.cell_d, tr.cell_ok)
        if self.active.size == 0 or not self.carve:
            return
        view = self._view_of(lx, ly, yaw)
        keep = _carve_k(self.act_mem, view[0], view[2], scan)
        if not keep.all():
            for nm in ('active', 'act_mem', 'act_x', 'act_y', 'act_s', 'act_d', 'act_ok'):
                setattr(self, nm, getattr(self, nm)[keep])

    def extract_boxes(self, s_car: float):
        """기억된 장애물 셀을 군집화하여 크기 가정 없는 순수 측정 기반 Frenet 박스로 반환."""
        if self.active.size == 0:
            return []
        tr = self.track
        rmin, rmax, dmin, dmax = _boxes_k(
            self.active, self.act_ok, self.act_s, self.act_x, self.act_y, self.act_d,
            tr.MW, tr.W, float(s_car), float(tr.L), float(self.n_stations * self.ds + 1.0), float(self.link),
        )
        out = []
        half = tr.mres / 2
        s32 = np.float32(s_car)
        for c in range(rmin.shape[0]):
            s_lo, s_hi = float(rmin[c] + s32) - half, float(rmax[c] + s32) + half
            d_lo, d_hi = float(dmin[c]) - half, float(dmax[c]) + half
            s_hi = max(s_hi, s_lo + self.ds)
            out.append([s_lo - self.obs_pad_s, s_hi + self.obs_pad_s, d_lo - self.obs_pad_d, d_hi + self.obs_pad_d])
        out.sort(key=lambda b: b[0])
        return out

    def body_clearance(self, x: float, y: float, yaw: float, dl: float, dist: float):
        sb = dist
        yw = yaw - sb * math.tan(dl) / WB
        ym = 0.5 * (yaw + yw)
        cx, cy = x - sb * math.cos(ym), y - sb * math.sin(ym)
        c, s_ = math.cos(yw), math.sin(yw)
        P = np.array([(cx + lx_ * c - ly_ * s_, cy + lx_ * s_ + ly_ * c) for lx_, ly_ in BODY_PTS])
        cw = float(np.min(self.track.clearance(P[:, 0], P[:, 1])))
        co = math.inf
        if self.active.size:
            dx = self.act_x[:, None] - P[None, :, 0]
            dy = self.act_y[:, None] - P[None, :, 1]
            co = math.sqrt(float(np.min(dx * dx + dy * dy)))
        return cw, co

    def plan_clear(self, plan: dict, dist: float, shift: float) -> bool:
        st, Xs, cum = plan['st'], plan['Xs'], plan['cum']
        c0 = float(np.interp(shift, st - st[0], cum)) if shift > 0 else 0.0
        k = np.flatnonzero((cum >= c0) & (cum <= c0 + dist))
        if k.size == 0:
            return True
        xs, ys = self.track.xy(st[k], Xs[k, 0])
        hd = np.array([self.track.heading(float(q)) for q in st[k]]) + Xs[k, 1]
        c, s_ = np.cos(hd), np.sin(hd)
        pts = [np.stack([xs + lx_ * c - ly_ * s_, ys + lx_ * s_ + ly_ * c], 1) for lx_, ly_ in FRONT_PTS]
        P = np.concatenate(pts)
        if float(np.min(self.track.clearance(P[:, 0], P[:, 1]))) < self.creep_m_wall:
            return False
        if self.active.size:
            dx = self.act_x[:, None] - P[None, :, 0]
            dy = self.act_y[:, None] - P[None, :, 1]
            if float(np.min(dx * dx + dy * dy)) < self.creep_m_obs ** 2:
                return False
        return True

    def fwd_clear(self, x: float, y: float, yaw: float, dl: float, dist: float) -> bool:
        for i in (1, 2, 3):
            cw, co = self.body_clearance(x, y, yaw, dl, -dist * i / 3)
            if cw < self.creep_m_wall or co < self.creep_m_obs:
                return False
        return True
