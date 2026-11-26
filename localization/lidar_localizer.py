"""LiDAR + 지도 + 제어 명령 기반 위치 추정 모듈 (GT 미사용).

기존 drivers/lidar_loc.py를 모듈화하고 다음 사항을 개선했다:
- 하드코딩된 ~/forza_ws/race_stack/stack_master/maps 경로 제거 -> common.paths.resolve_maps_dir() 사용
- 기본 inner 드라이버를 통합 단일 주행기(drivers.unified_corridor_mpc:UnifiedCorridorMPC)로 연결
"""
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from common.paths import resolve_maps_dir, resolve_project_root
from common.vehicle_params import A_MAX, ANG_INC, ANG_MIN, KP_ACC, KP_BRK, LIDAR_X, NB
from localization.loc_kernels import _corr_scores, _mm_predict, _mm_update, _predict_ekf, _register

INSTANCES = []
EVAL_OBSTACLES = None


class MapModel:
    """점유 지도, 부호 있는 거리장(SDF) 및 상관 탐색용 우도장(Likelihood Field) 모델."""

    _cache = {}

    def __init__(self, map_name):
        d = os.path.join(str(resolve_maps_dir()), map_name)
        y = f'{d}/{map_name}_orig.yaml'
        if not os.path.exists(y):
            y = f'{d}/{map_name}.yaml'
        meta = yaml.safe_load(open(y))
        img = np.array(Image.open(f"{d}/{meta['image']}").convert('L').transpose(Image.FLIP_TOP_BOTTOM))
        occ = img <= 128
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.H, self.W = occ.shape
        d_free = ndimage.distance_transform_edt(~occ)
        d_occ = ndimage.distance_transform_edt(occ)
        self.sdf = np.ascontiguousarray(np.where(occ, -(d_occ - 0.5), d_free - 0.5) * self.res, dtype=np.float64)
        self.occ = occ
        self._lf = {}
        try:
            w = json.load(open(f'{d}/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts']
            self.start = (w[0]['x_m'], w[0]['y_m'], w[0]['psi_rad'])
            self.line = np.array([[p['x_m'], p['y_m'], p['psi_rad']] for p in w])
        except Exception:
            self.start, self.line = None, None

    @classmethod
    def get(cls, map_name):
        if map_name not in cls._cache:
            cls._cache[map_name] = MapModel(map_name)
        return cls._cache[map_name]

    def lfield(self, tau, off, grow=0):
        key = (round(tau, 4), round(off, 4), grow)
        if key not in self._lf:
            lf = np.exp(-0.5 * ((self.sdf - off) / tau) ** 2)
            if grow > 0:
                lf = ndimage.maximum_filter(lf, size=2 * grow + 1)
            self._lf[key] = np.ascontiguousarray(lf, dtype=np.float32)
        return self._lf[key]

    def free_points(self, step, clear):
        k = max(1, int(round(step / self.res)))
        rr, cc = np.mgrid[k // 2:self.H:k, k // 2:self.W:k]
        ok = self.sdf[rr, cc] >= clear
        return (self.ox + (cc[ok] + 0.5) * self.res, self.oy + (rr[ok] + 0.5) * self.res)


class LidarLocalizer:
    """상태 [x, y, yaw, v, r, beta]에 대한 Iterated EKF + 다중 모델 지연 추정기."""

    DEFAULTS = dict(
        stride=2, r_min=0.06, r_max=20.0, sig_b=0.014, sig_ang=0.0005, sdf_off=-0.005, sdf_off_b=-0.015, huber=2.0,
        gate=0.25, gate2_min=0.06, gate2_k=5.0, inl_thr=0.08, info_scale=3.0, n_iter=4,
        q_xy=0.002, q_yaw=0.001, q_v=0.03, q_r=0.15, q_beta=0.005,
        p0_xy=0.05, p0_yaw=0.03, p0_v=0.1, p0_r=0.1, p0_beta=0.02,
        q_bad=0.9, q_lost=0.3, lost_n=3, q_ok=0.95, jump_gate=0.5, q_min_inl=0.2, q_min_inl_reloc=0.25,
        obs_link=0.2, obs_ext=1.0, rms_ok_k=1.3, rms_ok_add=0.004,
        loc_win=0.8, loc_yaw=0.4, loc_step=0.05, loc_ystep=0.0175,
        glob_step=0.12, glob_ystep=0.0524, glob_clear=0.10, glob_beams=90, glob_tau=0.12, glob_top=40,
        retry_global=0.5, t_buf=0.02, sig_adapt=True, sig_alpha=0.05,
        lat_mode='mm', lat_min=0.0, lat_max=0.15, lat_step=0.01, lat_tau=2.0, lat_hyst=2.0, lat_vmin=1.0, lat0=0.03,
        clock='sim', n_var0=0.36, n_var_min=0.04, n_var_alpha=0.02, n_sig1=0.1, rate0=100.0, rate_margin=9.0,
        rate_vmin=1.5, rate_lam=0.99,
    )

    def __init__(self, map_name, **kw):
        p = dict(self.DEFAULTS)
        for k, v in kw.items():
            if k not in p:
                raise KeyError(f'unknown localizer parameter {k}')
            p[k] = v
        self.p = p
        for k, v in p.items():
            setattr(self, k, v)
        self.map = MapModel.get(map_name)
        ang = ANG_MIN + ANG_INC * np.arange(NB)
        self.cb, self.sb = np.cos(ang), np.sin(ang)
        self.ang = ang
        self.Qd = np.array([self.q_xy, self.q_xy, self.q_yaw, self.q_v, self.q_r, self.q_beta]) ** 2
        self.Pmax = np.array([1.0, 1.0, 0.3, 2.0, 2.0, 0.2]) ** 2
        self.P0 = np.diag(np.array([self.p0_xy, self.p0_xy, self.p0_yaw, self.p0_v, self.p0_r, self.p0_beta]) ** 2)
        self.lats = np.round(np.arange(self.lat_min, self.lat_max + 1e-9, self.lat_step), 4)
        self.K = self.lats.size if self.lat_mode == 'mm' else 1
        self.Jsum = np.zeros(self.K)
        self.j = int(np.argmin(np.abs(self.lats - self.lat0))) if self.K > 1 else 0
        self.t_lat_est = float(self.lats[self.j]) if self.K > 1 else self.lat0

        m = self.map
        z = np.zeros(1)
        _predict_ekf(z, z, z, 0, 0.03, 0.02, 0.0, 2, np.array([0.0, 0, 0, 1.0, 0, 0]), np.eye(6), 0.0, 1.0,
                     self.Qd, self.Pmax)
        _register(np.zeros(6), np.eye(6), np.ones(NB), self.cb, self.sb, 2, 0.06, 20.0, m.sdf, m.ox, m.oy, m.res,
                  m.W, m.H, 0.015, 0.0015, -0.02, 0.0, 2.0, 0.25, 0.08, 0.08, 10.0, 2, True, 0.2, 1.0)
        S0 = np.zeros((2, 6))
        S0[:, 3] = 1.0
        S1, P1, _ = _mm_predict(z, z, z, 0, np.array([0.02, 0.03]), 0.02, 0.0, 2, S0, np.stack([np.eye(6)] * 2),
                                np.zeros(2), 1.0, self.Qd, self.Pmax)
        _mm_update(S1, P1, np.zeros(3), np.eye(3), np.zeros(3))
        lf = m.lfield(self.glob_tau, self.sdf_off + 0.7 * self.sdf_off_b)
        _corr_scores(lf, m.ox, m.oy, m.res, m.W, m.H, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.zeros(1),
                     np.zeros((1, 1)))
        self.reset()

    def reset(self):
        self.s = None
        self.P = None
        self.delta = 0.0
        self.S = self.PP = self.D = None
        self.t = None
        self.Ht, self.Hs, self.Hv = [], [], []
        self.n_bad = 0
        self.lost = False
        self.t_glob = -1e9
        self.q = 1.0
        self.last_good_t = None
        self.good_streak = 0
        self.last_info = None
        if not hasattr(self, 'rms_ema'):
            self.rms_ema = self.sig_b
        if not hasattr(self, 'rate'):
            self.rate = float(self.rate0)
            self.rate_N, self.rate_T = 20.0, 20.0 / self.rate0
            self.n_var = float(self.n_var0)
        self.t_wall = None
        self.debug_scores = getattr(self, 'debug_scores', False)
        self.last_r = None
        self.n_last = 0
        self.n_dup = getattr(self, 'n_dup', 0)

    def add_command(self, t, steer, speed):
        self.Ht.append(float(t))
        self.Hs.append(float(steer))
        self.Hv.append(float(speed))

    def _hist(self, t_keep):
        k = 0
        while k + 1 < len(self.Ht) and self.Ht[k + 1] < t_keep:
            k += 1
        if k:
            del self.Ht[:k], self.Hs[:k], self.Hv[:k]
        n = len(self.Ht)
        return (np.array(self.Ht, dtype=np.float64), np.array(self.Hs, dtype=np.float64),
                np.array(self.Hv, dtype=np.float64), n)

    def init_pose(self, t, x, y, yaw, scan, refine=True, verified=True):
        s = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        q = 1.0
        if refine and scan is not None:
            s3, q = self.local_search(scan, s[:3], win=0.3, yaw_win=0.15)
            s[:3] = s3
        self._set_all(s, self.P0.copy(), 0.0)
        self.t = t
        self.t_wall = t
        self.last_r = self._ranges(scan) if scan is not None else None
        self.n_bad = 0
        self.lost = False
        self.q = q
        if verified:
            self.last_good_t = t
            self.good_streak = 1
        else:
            self.last_good_t = None
            self.good_streak = 0
        return q

    def pose_std(self) -> float:
        """현재 추정 위치(x, y)의 표준편차(불확실성, m)를 반환한다."""
        if self.P is None:
            return math.inf
        return math.sqrt(max(float(self.P[0, 0]) + float(self.P[1, 1]), 0.0))

    def initialise(self, t, scan, pose0):
        if pose0 is None:
            pz, q = self.global_search(scan)
        else:
            pz, q = self.local_search(scan, pose0, win=0.3, yaw_win=0.15)
            if q < self.q_ok:
                pz2, q2 = self.local_search(scan, pose0, win=1.0, yaw_win=0.5)
                if q2 > q:
                    pz, q = pz2, q2
        ok = pz is not None and q >= self.q_ok
        if ok:
            self.init_pose(t, pz[0], pz[1], pz[2], scan, refine=False)
            self.q = q
        return ok, q

    def _set_all(self, s, P, delta):
        self.S = np.tile(s, (self.K, 1))
        self.PP = np.tile(P, (self.K, 1, 1))
        self.D = np.full(self.K, float(delta))
        self._select()

    def _select(self):
        self.s, self.P, self.delta = self.S[self.j], self.PP[self.j], float(self.D[self.j])

    def _ranges(self, scan):
        r = np.asarray(scan, dtype=np.float64)
        if r.shape[0] != NB:
            raise ValueError(f'scan with {r.shape[0]} beams, expected {NB}')
        return np.where(np.isfinite(r), r, -1.0)

    def _gate2(self, sig):
        return max(self.gate2_min, self.gate2_k * sig)

    def _quality(self, nval, ninl, nsmall, min_inl=None):
        if ninl < (self.q_min_inl if min_inl is None else min_inl) * max(nval, 1):
            return 0.0
        return ninl / max(nval - nsmall, 1)

    def refine(self, scan, pose3, iters=8):
        m = self.map
        r = self._ranges(scan)
        s = np.array([pose3[0], pose3[1], pose3[2], 0.0, 0.0, 0.0])
        P = np.diag(np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0]) ** 2)
        nval, ninl, nsmall = 1, 0, 0
        sig = max(self.sig_b, self.rms_ema) if self.sig_adapt else self.sig_b
        for gate, nit in ((0.5, iters), (0.25, 30)):
            s, _, nval, ninl, nfree, nsmall, rms, A, _, _, _, _ = _register(
                s, P, r, self.cb, self.sb, 1, self.r_min, self.r_max, m.sdf, m.ox, m.oy, m.res, m.W, m.H, sig,
                self.sig_ang, self.sdf_off, self.sdf_off_b, self.huber, gate,
                self._gate2(sig) if gate < 0.3 else gate, self.inl_thr, self.info_scale, nit, False,
                self.obs_link, self.obs_ext,
            )
        ref = max(self.sig_b, self.rms_ema)
        qt = self._quality(nval, ninl, nsmall, self.q_min_inl_reloc)
        if rms > max(self.rms_ok_k * ref, ref + self.rms_ok_add):
            qt = 0.0
        return s[:3].copy(), (round(ninl / max(nval, 1), 2), -rms), qt

    def _car_points(self, scan, nbeam):
        r = self._ranges(scan)
        idx = np.arange(0, NB, max(1, NB // nbeam))
        idx = idx[(r[idx] > self.r_min) & (r[idx] < self.r_max)]
        return LIDAR_X + r[idx] * self.cb[idx], r[idx] * self.sb[idx]

    def _scores(self, scan, cx, cy, yaws, tau, nbeam, grow=0):
        m = self.map
        bx, by = self._car_points(scan, nbeam)
        cyw, syw = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        qx = (cyw * bx[None, :] - syw * by[None, :]) / m.res
        qy = (syw * bx[None, :] + cyw * by[None, :]) / m.res
        out = np.zeros((cx.size, yaws.size))
        _corr_scores(m.lfield(tau, self.sdf_off + 0.7 * self.sdf_off_b, grow), m.ox, m.oy, m.res, m.W, m.H, qx, qy,
                     np.ascontiguousarray(cx, dtype=np.float64), np.ascontiguousarray(cy, dtype=np.float64), out)
        return out / max(bx.size, 1)

    def local_search(self, scan, pose3, win=None, yaw_win=None):
        win = self.loc_win if win is None else win
        yaw_win = self.loc_yaw if yaw_win is None else yaw_win
        g = np.arange(-win, win + 1e-9, self.loc_step)
        gx, gy = np.meshgrid(g, g)
        cx, cy = pose3[0] + gx.ravel(), pose3[1] + gy.ravel()
        yaws = pose3[2] + np.arange(-yaw_win, yaw_win + 1e-9, self.loc_ystep)
        sc = self._scores(scan, cx, cy, yaws, 0.06, 120)
        best = []
        flat = np.argsort(sc.ravel())[::-1][:5]
        for f in flat:
            k, j = divmod(int(f), yaws.size)
            best.append(self.refine(scan, (cx[k], cy[k], yaws[j])))
        best.append(self.refine(scan, pose3))
        b = max(best, key=lambda b: b[1])
        b = self._polish(scan, b)
        return b[0], b[2]

    def _polish(self, scan, b):
        for _ in range(3):
            b2 = self.refine(scan, b[0])
            moved = math.hypot(b2[0][0] - b[0][0], b2[0][1] - b[0][1])
            b = b2
            if moved < 1e-3:
                break
        return b

    def global_search(self, scan):
        cx, cy = self.map.free_points(self.glob_step, self.glob_clear)
        yaws = np.arange(-math.pi, math.pi, self.glob_ystep)
        sc = self._scores(scan, cx, cy, yaws, self.glob_tau, self.glob_beams, grow=1)
        bj = np.argmax(sc, axis=1)
        bs = sc[np.arange(cx.size), bj]
        order = np.argsort(bs)[::-1]
        picked = []
        for k in order:
            if all(math.hypot(cx[k] - cx[q], cy[k] - cy[q]) > 0.3 for q in picked):
                picked.append(k)
            if len(picked) >= self.glob_top:
                break
        best = (None, (-1.0, 0.0), 0.0)
        good = 0
        for k in picked:
            pz, key, qt = self.refine(scan, (cx[k], cy[k], yaws[bj[k]]))
            if key > best[1]:
                best = (pz, key, qt)
            good += key[0] > 0.97
            if good >= 3:
                break
        if best[0] is not None:
            best = self._polish(scan, best)
        return best[0], best[2]

    def step(self, t, scan, t_lat, c_tyre=1.0):
        m = self.map
        r = self._ranges(scan)
        if self.last_r is not None and np.array_equal(r, self.last_r):
            self.n_last = 0
            self.n_dup += 1
            s = self.s
            return s[0], s[1], s[2], s[3], s[4]
        dtw = t - self.t_wall
        if self.clock == 'wall':
            n0 = int(round(dtw / 0.01))
            if n0 <= 0:
                s = self.s
                return s[0], s[1], s[2], s[3], s[4]
            n_exp = float(n0)
        else:
            n_exp = max(dtw, 0.0) * self.rate
        self.last_r = r
        self.t_wall = max(self.t_wall, t)
        Ht, Hs, Hv, nh = self._hist(self.t - 0.5)
        j = self.j
        lat_j = float(self.lats[j]) if self.K > 1 else float(t_lat)
        sig = max(self.sig_b, self.rms_ema) if self.sig_adapt else self.sig_b
        sig_n = math.sqrt(self.n_var + (self.n_sig1 * n_exp) ** 2)
        self.dbg_sc = []
        cache = {}

        def evaluate(n):
            s_pred, P_pred, d1 = _predict_ekf(
                Ht, Hs, Hv, nh, lat_j, float(self.t_buf), float(self.t), n, self.S[j],
                self.PP[j], float(self.D[j]), float(c_tyre), self.Qd, self.Pmax,
            )
            reg = _register(
                s_pred, P_pred, r, self.cb, self.sb, self.stride, self.r_min, self.r_max, m.sdf, m.ox,
                m.oy, m.res, m.W, m.H, sig, self.sig_ang, self.sdf_off, self.sdf_off_b, self.huber,
                self.gate, self._gate2(sig), self.inl_thr, self.info_scale, self.n_iter, True,
                self.obs_link, self.obs_ext,
            )
            score = reg[11] + ((n - n_exp) / sig_n) ** 2
            cache[n] = (score, n, s_pred, P_pred, d1, reg)

        if self.clock == 'wall':
            evaluate(min(int(round(n_exp)), 500))
        else:
            hi = min(max(2, int(round(n_exp)) + 1), 500)
            for nn in range(1, hi + 1):
                evaluate(nn)
            while min(cache.values(), key=lambda c: c[0])[1] == hi and hi < 500:
                hi += 1
                evaluate(hi)

        scores = sorted(c[0] for c in cache.values())
        best = min(cache.values(), key=lambda c: c[0])
        _, n, s_pred, P_pred, d1, reg = best
        s_new, P_new, nval, ninl, nfree, nsmall, rms, A, it, p0, bq, jc = reg
        self.n_last = n
        if len(scores) > 1:
            if scores[1] - scores[0] > self.rate_margin and s_new[3] > self.rate_vmin:
                self.n_var += self.n_var_alpha * ((n - n_exp) ** 2 - self.n_var)
                self.n_var = max(self.n_var, self.n_var_min)
                self.rate_N = self.rate_lam * self.rate_N + n
                self.rate_T = self.rate_lam * self.rate_T + max(dtw, 1e-3)
                self.rate = min(max(self.rate_N / self.rate_T, 50.0), 150.0)

        if self.K > 1:
            S1, P1, D1 = _mm_predict(Ht, Hs, Hv, nh, self.lats, float(self.t_buf), float(self.t), n, self.S, self.PP,
                                     self.D, float(c_tyre), self.Qd, self.Pmax)
        else:
            S1, P1, D1 = s_pred[None, :].copy(), P_pred[None, :, :].copy(), np.array([d1])

        dt = n * 0.01
        self.D = D1
        self.t = self.t + dt
        t = self.t
        q = self._quality(nval, ninl, nsmall)
        self.free_frac = nfree / max(nval, 1)
        self.free_unexpl = (nfree - nsmall) / max(nval, 1)
        jump = math.hypot(s_new[0] - s_pred[0], s_new[1] - s_pred[1])
        self.q = q
        self.last_info = (nval, ninl, rms, it, jump)

        if q >= self.q_bad and jump < self.jump_gate:
            if self.K > 1:
                try:
                    S2, P2, J = _mm_update(S1, P1, p0, A, bq)
                    bad = ~np.isfinite(S2).all(axis=1) | ~np.isfinite(P2).all(axis=(1, 2)) |                         (np.abs(S2[:, 3] - s_new[3]) > 1.0) | ~np.isfinite(J)
                    if bad.any():
                        S2[bad] = s_new
                        P2[bad] = P_new
                        J = np.where(bad, np.max(np.where(bad, -np.inf, J)) if (~bad).any() else 0.0, J)
                        self.n_reanchor = getattr(self, 'n_reanchor', 0) + int(bad.sum())
                except np.linalg.LinAlgError:
                    S2, P2, J = np.tile(s_new, (self.K, 1)), np.tile(P_new, (self.K, 1, 1)), np.zeros(self.K)
                    self.n_reanchor = getattr(self, 'n_reanchor', 0) + self.K
                S2[j], P2[j] = s_new, P_new
                if s_new[3] > self.lat_vmin and n <= 10:
                    lam = math.exp(-dt / self.lat_tau)
                    self.Jsum = lam * self.Jsum + (J - J.min())
                    k = int(np.argmin(self.Jsum))
                    if self.Jsum[k] < self.Jsum[j] - self.lat_hyst:
                        self.j = k
                        self.t_lat_est = float(self.lats[k])
                self.S, self.PP = S2, P2
            else:
                self.S, self.PP = s_new[None, :].copy(), P_new[None, :, :].copy()
            self.n_bad = 0
            self.rms_ema += self.sig_alpha * (rms - self.rms_ema)
            self.lost = False
            self.last_good_t = t
            self.good_streak += 1
        else:
            self.S, self.PP = S1, P1
            self.n_bad += 1
            self.good_streak = 0
            if self.n_bad >= self.lost_n or q < self.q_lost:
                self._select()
                self._relocalise(t, scan)
        self._select()
        s = self.s
        return s[0], s[1], s[2], s[3], s[4]

    def _relocalise(self, t, scan):
        pz, q = self.local_search(scan, self.s[:3])
        if q < self.q_ok and t - self.t_glob >= self.retry_global:
            self.t_glob = t
            pg, qg = self.global_search(scan)
            if pg is not None and qg > q:
                pz, q = pg, qg
        self.lost = q < self.q_ok
        self.n_reloc = getattr(self, 'n_reloc', 0) + 1
        if not self.lost:
            for k in range(self.K):
                self.S[k, :3] = pz
                P = self.PP[k]
                P[:3, :] = 0.0
                P[:, :3] = 0.0
                P[0, 0] = P[1, 1] = self.p0_xy ** 2
                P[2, 2] = self.p0_yaw ** 2
            self._select()
            self.n_bad = 0
            self.last_good_t = t
            self.good_streak = 0  # 재위치추정 직후: 연속 정상 관측으로 검증될 때까지 재가속 제한
        self.q = q


def _load_kwargs(profile):
    if profile is None:
        return {}
    if isinstance(profile, dict):
        return dict(profile)
    path = profile if os.path.isabs(profile) else os.path.join(str(resolve_project_root()), profile)
    return json.load(open(path))


class LidarLocDriver:
    """LiDAR 위치 추정기와 내부 주행기(UnifiedCorridorMPC)를 결합하는 GT-free 래퍼.

    위치추정 신뢰도 기반 안전 계층(Fail-Safe):
      - 마지막 정상 정합 이후 경과시간, 연속 실패(n_bad), 위치 불확실성(pose_std)을 종합해
        정상(OK) -> 감속(DEGRADED) -> 정지(STOP)로 단계적 전환한다.
      - lost 또는 강제 초기화(init_force) 이후에는 연속 정상 관측(loc_recover_streak)이
        확인되기 전까지 재가속을 제한한다.
      - 중복 스캔(n_last == 0) 시 last_obs_valid=False를 표시해 상위 Watchdog이 감지할 수 있게 한다.
    """

    def __init__(self, inner='drivers.unified_corridor_mpc:UnifiedCorridorMPC',
                 profile='profiles/berlin_unified.json',
                 inner_kwargs=None, init='start', forbid_gt=True, shadow=False, eval_log=False, record=False,
                 loc=None, map_name=None, use_inner_c=True, init_force=10,
                 allow_forced_init=True, loc_degrade_dt=0.15, loc_stop_dt=0.40,
                 loc_degrade_bad=2, loc_uncert_degrade=0.18, loc_uncert_stop=0.45,
                 loc_degrade_v=2.0, loc_forced_init_v=1.5, loc_recover_streak=4, **extra):
        kw = _load_kwargs(profile)
        kw.update(inner_kwargs or {})
        kw.update(extra)
        if map_name is not None:
            kw['map_name'] = map_name
        loc = dict(loc or {})
        self.lat_own = loc.get('lat_mode', LidarLocalizer.DEFAULTS['lat_mode']) == 'mm' and not shadow
        if self.lat_own:
            kw['lat_est'] = False
        self.map_name = kw.get('map_name', 'berlin')
        mod, cls = inner.split(':')
        self.inner = getattr(importlib.import_module(mod), cls)(**kw)
        self.loc = LidarLocalizer(self.map_name, **loc)
        if init not in ('gt', 'global', 'start'):
            raise ValueError(f'init must be gt, global or start, not {init}')
        if forbid_gt and init == 'gt':
            raise ValueError('forbid_gt=True needs init global / start (or set_initial_pose before every act)')
        self.init, self.forbid_gt, self.shadow = init, forbid_gt, shadow
        self.eval_log, self.record = eval_log, record
        self.use_inner_c = use_inner_c
        self.init_force = init_force
        self.allow_forced_init = bool(allow_forced_init)
        self.loc_degrade_dt = float(loc_degrade_dt)
        self.loc_stop_dt = float(loc_stop_dt)
        self.loc_degrade_bad = int(loc_degrade_bad)
        self.loc_uncert_degrade = float(loc_uncert_degrade)
        self.loc_uncert_stop = float(loc_uncert_stop)
        self.loc_degrade_v = float(loc_degrade_v)
        self.loc_forced_init_v = float(loc_forced_init_v)
        self.loc_recover_streak = int(loc_recover_streak)
        self.pending_pose = None
        self.n_gt_reads = 0
        self.n_resets = 0
        self.stats = dict(err=[], loc_ms=[], q=[], reloc=0, lost_calls=0, init_q=[], init_ms=[], tlat=[], c=[],
                          free=[], dobs=[], boxes=0, phantom=0, phantom_calls=0, calls=0, dup=0,
                          degraded_calls=0, stop_calls=0)
        self.rec = []
        INSTANCES.append(self)
        self.reset()

    def reset(self):
        self.inner.reset()
        self.loc.reset()
        self.need_init = True
        self.init_tries = 0
        self.t0 = None
        self.last_cmd = None
        self._issued = False
        self.loc_state = 'OK'
        self.loc_recovering = False
        self._forced_init_active = False
        self.last_obs_valid = False
        self.n_resets += 1

    def set_initial_pose(self, x, y, yaw):
        self.pending_pose = (float(x), float(y), float(yaw))
        self.need_init = True
        self.init_tries = 0

    @property
    def gt_free(self):
        return self.forbid_gt

    @property
    def v_est(self):
        return float(self.loc.s[3]) if self.loc.s is not None else math.nan

    @property
    def loc_degraded(self) -> bool:
        return self.loc_state != 'OK' or self.loc_recovering

    def _read_gt_once(self, x, y, yaw):
        if self.forbid_gt:
            raise RuntimeError('LidarLocDriver: GT read requested with forbid_gt=True')
        self.n_gt_reads += 1
        return float(x), float(y), float(yaw)

    def _initialise(self, t, scan, x, y, yaw):
        t_a = time.perf_counter()
        if self.pending_pose is not None:
            pose0, src = self.pending_pose, 'operator'
        elif self.init == 'gt':
            pose0, src = self._read_gt_once(x, y, yaw), 'gt'
        elif self.init == 'start':
            pose0, src = self.loc.map.start, 'start'
        else:
            pose0, src = None, 'global'
        self.pending_pose = None
        ok, q = self.loc.initialise(t, scan, pose0)
        if not ok and pose0 is not None:
            self.pending_pose = pose0
        self.stats['init_q'].append(round(float(q), 3))
        self.stats['init_ms'].append(round((time.perf_counter() - t_a) * 1e3, 1))
        self.init_src = src
        return ok

    def _eval_loc_safety(self) -> float:
        """위치추정 신뢰도를 평가해 (OK / DEGRADED / STOP) 상태와 허용 최고 속도를 결정한다."""
        loc = self.loc
        if loc.t is not None and loc.last_good_t is not None:
            dt_good = max(0.0, float(loc.t - loc.last_good_t))
        else:
            dt_good = math.inf
        pos_std = loc.pose_std()

        # 강제 초기화 직후에는 정합이 1회라도 성공하기 전까지 저속 출발만 허용 (즉시 정지 데드락 방지)
        if self._forced_init_active and loc.last_good_t is None and not loc.lost:
            self.loc_state = 'DEGRADED'
            self.loc_recovering = True
            self.stats['degraded_calls'] += 1
            return self.loc_forced_init_v

        if loc.lost or dt_good >= self.loc_stop_dt or pos_std >= self.loc_uncert_stop:
            self.loc_state = 'STOP'
            self.loc_recovering = True
            self.stats['stop_calls'] += 1
            return 0.0

        if (loc.n_bad >= self.loc_degrade_bad or dt_good >= self.loc_degrade_dt
                or pos_std >= self.loc_uncert_degrade):
            self.loc_state = 'DEGRADED'
            self.loc_recovering = True
            self.stats['degraded_calls'] += 1
            return self.loc_degrade_v

        self.loc_state = 'OK'
        if self.loc_recovering:
            if loc.good_streak >= self.loc_recover_streak:
                self.loc_recovering = False
                self._forced_init_active = False
                return math.inf
            self.stats['degraded_calls'] += 1
            return self.loc_forced_init_v if self._forced_init_active else self.loc_degrade_v
        return math.inf

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        t = float(t)
        if self.t0 is None:
            self.t0 = t
        tr = t - self.t0
        t_a = time.perf_counter()
        self.last_obs_valid = False

        if self.need_init:
            self.init_tries += 1
            if not self._initialise(tr, scan, x, y, yaw):
                pose0 = self.pending_pose
                if self.allow_forced_init and pose0 is not None and self.init_tries >= self.init_force:
                    # 강제 초기화 시에도 즉시 고속 출발하지 않고, 연속 정합 확인 전까지 저속 출발만 허가
                    self.loc.init_pose(tr, pose0[0], pose0[1], pose0[2], scan, refine=False, verified=False)
                    self.pending_pose = None
                    self._forced_init_active = True
                    self.loc_recovering = True
                    self.stats['init_forced'] = self.stats.get('init_forced', 0) + 1
                else:
                    return 0.0, 0.0
            else:
                self._forced_init_active = False
                self.loc_recovering = False
            self.need_init = False
            self.last_obs_valid = True
            est = (self.loc.s[0], self.loc.s[1], self.loc.s[2], 0.0, 0.0)
        else:
            t_lat = getattr(self.inner, 't_lat', self.loc.lat0)
            c = getattr(self.inner, 'c_tyre', 1.0) if self.use_inner_c else 1.0
            try:
                est = self.loc.step(tr, scan, t_lat, c)
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                s_keep = self.loc.s.copy() if self.loc.s is not None else None
                self.stats['loc_exc'] = self.stats.get('loc_exc', 0) + 1
                self.loc_recovering = True
                if s_keep is None or not np.isfinite(s_keep).all():
                    self.need_init = True
                    self.pending_pose = None
                    return (self.last_cmd[0] if self.last_cmd is not None else 0.0), 0.0
                self.loc._set_all(s_keep, self.loc.P0.copy(), self.loc.delta)
                self.loc.t_wall = tr
                self.loc.good_streak = 0
                est = (s_keep[0], s_keep[1], s_keep[2], s_keep[3], s_keep[4])

            if self.loc.lost:
                self.stats['lost_calls'] += 1

            if self.loc.n_last == 0 and self.last_cmd is not None and not self.shadow:
                # 동일한 스캔 반복: 새 관측이 없으므로 last_obs_valid=False 유지 (상위 Watchdog 감지용)
                self.stats['dup'] += 1
                self._issued = False
                speed_cap = self._eval_loc_safety()
                return self.last_cmd[0], min(max(0.0, self.last_cmd[1]), speed_cap)

            self.last_obs_valid = (not self.loc.lost) and (self.loc.n_last > 0)

        loc_ms = (time.perf_counter() - t_a) * 1e3
        self.stats['loc_ms'].append(loc_ms)
        self.stats['q'].append(self.loc.q)
        self.stats['free'].append((getattr(self.loc, 'free_frac', 0.0), getattr(self.loc, 'free_unexpl', 0.0)))

        speed_cap = self._eval_loc_safety()
        if hasattr(self.inner, 'loc_degraded'):
            self.inner.loc_degraded = self.loc_degraded

        if self.lat_own:
            self.inner.t_lat = self.loc.t_lat_est
        if self.shadow:
            steer, speed = self.inner.act(scan, x, y, yaw, v, yaw_rate, t)
        else:
            steer, speed = self.inner.act(scan, *est, self.loc.t)

        # 후진 불가(>= 0.0) 및 위치추정 신뢰도 기반 속도 상한 적용
        speed = min(max(0.0, float(speed)), speed_cap)
        self.loc.add_command(self.loc.t, steer, speed)
        self.last_cmd = (steer, speed)
        self._issued = True
        self.stats['reloc'] = getattr(self.loc, 'n_reloc', 0)
        return steer, speed

    def __getattr__(self, name):
        if name == 'inner' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.inner, name)
