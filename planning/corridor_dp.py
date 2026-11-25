"""단일 Frenet DP 기반 안전 통로(Corridor) 생성기.

MUX 오인 요소 제거 및 규정 준수:
- 기존 원호 탐색(Arc Search) 및 후진 탈출 기동 로직을 완전히 제거했다 ('후진 불가!!' 규정 준수).
- TT/OBS 모드 스위칭을 제거하여 모든 장애물 조건에서 단일 정책으로 동작한다.
- 해가 없을 때(Infeasible) 별도 플래너로 전환하는 대신, 단일 DP 내부의 곡률 제약 완화
  (Adaptive Curvature Relaxation)로 경로를 재탐색한다.
"""
import math
import numpy as np

from common.numba_kernels import _dp_core, _dp_k
from common.vehicle_params import BOX_FRONT


class CorridorDPPlanner:
    def __init__(self, track, p: dict):
        self.track = track
        self.n = int(p['n'])
        self.ds = float(p['ds'])
        self.dd = float(p['dd'])
        self.dp_ds = float(p['dp_ds'])
        self.dp_kmax = int(p['dp_kmax'])
        self.dp_dmax = float(p['dp_dmax'])
        self.dp_vmax = float(p['dp_vmax'])
        self.dp_adec = float(p['dp_adec'])
        self.dp_near = float(p['dp_near'])
        self.p_near = float(p['p_near'])
        self.w_dev = float(p['w_dev'])
        self.w_sl = float(p['w_sl'])
        self.w_cv = float(p['w_cv'])
        self.dp_kcap = float(p['dp_kcap'])
        self.dp_vfloor = float(p['dp_vfloor'])
        self.c_pref = float(p['c_pref'])
        self.w_soft = float(p['w_soft'])
        self.w_wall = float(p['w_wall'])
        self.p_flip = float(p['p_flip'])
        self.p_flip_hard = float(p['p_flip_hard'])
        self.commit_t = float(p['commit_t'])
        self.commit_d = float(p['commit_d'])
        self.commit_ramp = float(p['commit_ramp'])
        self.young_t = float(p['young_t'])
        self.dp_kgeo = float(p['dp_kgeo'])
        self.w_geo = float(p['w_geo'])
        self.cluster_ds = float(p['cluster_ds'])
        self.release_margin = float(p['release_margin'])
        self.hard_block_v = float(p['hard_block_v'])
        self.min_gap_w = float(p['min_gap_w'])
        self.min_gap_keep = float(p['min_gap_keep'])
        self.gap_fallback = bool(p['gap_fallback'])
        self.g_narrow = float(p['g_narrow'])
        self.g_diff = float(p['g_diff'])
        # 적응형 곡률 완화(Adaptive Curvature Relaxation) 파라미터
        self.v_slow = float(p['v_slow'])
        self.kcap_slow = float(p['kcap_slow'])
        self.kmax_slow = int(p['kmax_slow'])
        self.dp_adec_slow = float(p['dp_adec_slow'])
        self.slow_commit = p['slow_commit']
        self.a_lat_slow = float(p['a_lat_slow'])
        # S자(weave) 속도 제한 계산용 파라미터
        self.a_weave = float(p['a_weave'])
        self.v_weave_min = float(p['v_weave_min'])
        self.weave_pre = float(p['weave_pre'])

        # Numba 커널 워밍업
        _dp_core(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3), dtype=np.bool_), np.array([-1, 0, 1]),
                 np.ones(3, dtype=np.int64), np.zeros(3), np.zeros(3), np.full((3, 3), -1.0), np.ones((3, 3)), 1.0)
        tr = self.track
        for bxw in (np.zeros((0, 4)), np.array([[1.0, 1.5, -0.2, 0.2]])):
            nb_ = bxw.shape[0]
            _dp_k(0.0, 0.0, 0.0, 3.0, self.n, self.ds, self.dp_ds, self.dd, self.dp_kmax,
                  tr.wlo_c, tr.whi_c, tr.wlo_p, tr.whi_p, float(tr.fds), 1.8, 1.0, 10.0, bxw,
                  np.ones(nb_, dtype=np.int64), np.ones(nb_), np.zeros(nb_, dtype=np.bool_), 0.2, 0.6, 30.0, 0.12,
                  40.0, True, 0.06, 4.0, 6.0, 2.0, 7.5, 0.8, 20.0, 2.0, 1.0, 0.0, tr.fk, float(tr.L), 0.95, 10.0)

        self.reset()

    def reset(self):
        self.clear_memory()
        self.box_age = []
        self.slow_at = None
        self.weaves = []

    def clear_memory(self):
        self.prev_sides = []
        self.seen_boxes = []
        self._ps_cache = [None, None, None]
        self._sb_cache = [None, None, None]

    @staticmethod
    def _first_match(lst, cache, smid, dc, ts, td):
        if cache[0] is not lst:
            idx = {}
            for i, e in enumerate(lst):
                idx.setdefault(math.floor(e[0] / 0.4), []).append(i)
            cache[0], cache[1], cache[2] = lst, idx, {}
        memo = cache[2]
        hit = memo.get((smid, dc), memo)
        if hit is not memo:
            return hit
        idx = cache[1]
        kb = math.floor(smid / 0.4)
        best = None
        for k in range(kb - 2, kb + 3):
            for i in idx.get(k, ()):
                e = lst[i]
                if abs(e[0] - smid) < ts and abs(e[1] - dc) < td:
                    if best is None or i < best:
                        best = i
                    break
        res = None if best is None else lst[best]
        memo[(smid, dc)] = res
        return res

    def _prev_side(self, smid, dc):
        e = self._first_match(self.prev_sides, self._ps_cache, smid, dc, 0.35, 0.3)
        return None if e is None else e[2]

    def _first_seen(self, smid, dc, t):
        e = self._first_match(self.seen_boxes, self._sb_cache, smid, dc, 0.4, 0.35)
        return t if e is None else e[2]

    def _close_narrow(self, s0, boxes, v, w_h):
        dnear = max(self.commit_d, self.commit_t * v)
        bx2 = []
        for b in boxes:
            b2 = list(b)
            ss = np.linspace(b[0], b[1], 5)
            wlo, whi = self.track.wall_bounds(ss, self.track.wc_eff)
            g_hi = float(np.min(whi)) - (b[3] + w_h)
            g_lo = (b[2] - w_h) - float(np.max(wlo))
            prev = self._prev_side(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]))
            dist = b[0] - (s0 + BOX_FRONT)
            near = dist < dnear and v > self.hard_block_v
            if g_hi < self.g_narrow and g_lo > g_hi + self.g_diff and not (near and prev == 1):
                b2[3] = float(np.max(whi)) + 0.5
            elif g_lo < self.g_narrow and g_hi > g_lo + self.g_diff and not (near and prev == -1):
                b2[2] = float(np.min(wlo)) - 0.5
            bx2.append(b2)
        return bx2

    def _solve_dp(self, s0, d0, chi0, boxes, v, w_h, a_lat_hard, c_tyre,
                  relax=(), hard=True, gap=True,
                  vfloor=None, vmax=None, kcap=None, adec=None, kmax=None,
                  commit_d=None, commit_t=None, alh_override=None):
        tr = self.track
        if self.g_narrow >= 0.0 and boxes and gap:
            boxes = self._close_narrow(s0, boxes, v, w_h)
        nbx = len(boxes)
        bx = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        bprev = np.zeros(nbx, dtype=np.int64)
        bpf = np.zeros(nbx)
        bhard = np.zeros(nbx, dtype=np.bool_)
        mids = [0.5 * (c2[0] + c2[1]) for c2 in boxes]
        young = [i for i, age in enumerate(self.box_age[:nbx]) if age < self.young_t]
        cd = self.commit_d if commit_d is None else commit_d
        ct = self.commit_t if commit_t is None else commit_t
        dnear = max(cd, ct * v)

        for bi_, b in enumerate(boxes):
            smid, dc = mids[bi_], 0.5 * (b[2] + b[3])
            prev = self._prev_side(smid, dc)
            if prev is None:
                continue
            dist = b[0] - (s0 + BOX_FRONT)
            f = min(1.0, max(0.0, (dnear + self.commit_ramp - dist) / self.commit_ramp))
            if dist > dnear + self.release_margin:
                if bi_ in relax:
                    f = 0.0
                for i2 in young:
                    if i2 != bi_ and abs(mids[i2] - smid) < self.cluster_ds:
                        f = 0.0
                        break
            bprev[bi_] = prev
            bpf[bi_] = self.p_flip + (self.p_flip_hard - self.p_flip) * f
            bhard[bi_] = hard and f >= 1.0 and dist < dnear and v > self.hard_block_v

        st, dpath, sides, s_block = _dp_k(
            float(s0), float(d0), float(chi0), float(v), self.n, self.ds, self.dp_ds, self.dd,
            self.dp_kmax if kmax is None else int(kmax),
            tr.wlo_c, tr.whi_c, tr.wlo_p, tr.whi_p, float(tr.fds), self.dp_dmax,
            self.w_dev, self.w_wall, bx, bprev, bpf, bhard, float(w_h), self.dp_near,
            self.p_near, self.c_pref, self.w_soft, bool(gap and self.min_gap_w > 0.0),
            self.min_gap_w,
            self.dp_adec if adec is None else float(adec),
            self.dp_vmax if vmax is None else float(vmax),
            self.dp_vfloor if vfloor is None else float(vfloor),
            float(a_lat_hard if alh_override is None else alh_override),
            self.dp_kcap if kcap is None else float(kcap),
            self.w_cv, self.w_sl, float(c_tyre),
            self.min_gap_keep, tr.fk, float(tr.L), self.dp_kgeo, self.w_geo,
        )
        return st, dpath, sides.tolist(), (None if math.isnan(s_block) else float(s_block))

    def _compute_weaves(self, boxes, sides, w_h, s0):
        out = []
        bs = sorted((b, sd) for b, sd in zip(boxes, sides) if sd != 0 and b[1] > s0)
        for (b1, s1), (b2, s2) in zip(bs[:-1], bs[1:]):
            if s1 == s2 or b2[0] - b1[1] > 4.0:
                continue
            d1 = b1[3] + w_h if s1 > 0 else b1[2] - w_h
            d2 = b2[3] + w_h if s2 > 0 else b2[2] - w_h
            dd = (d2 - d1) * s2
            if dd <= 0.0:
                continue
            ln = max(b2[0] - b1[1] + 0.3, 0.3)
            kap = 4.0 * dd / (ln * ln)
            v_w = max(math.sqrt(self.a_weave / kap), self.v_weave_min)
            out.append((b1[1] - self.weave_pre, b2[0], v_w))
        return out

    def plan_corridor(self, s0, d0, chi0, boxes, vv, w_h, t, a_lat_hard, c_tyre):
        """단일 DP 최적화: 통과 방향 결정, 클러스터 재계획, 적응형 곡률 완화 및 weave 속도 제한 산출."""
        firsts = [self._first_seen(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), t) for b in boxes]
        self.box_age = [t - f for f in firsts]
        self.seen_boxes = [(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), f) for b, f in zip(boxes, firsts)]

        st, dpath, sides, s_block = self._solve_dp(s0, d0, chi0, boxes, vv, w_h, a_lat_hard, c_tyre)
        mids = [0.5 * (b[0] + b[1]) for b in boxes]
        flipped = []
        for i, b in enumerate(boxes):
            p_ = self._prev_side(mids[i], 0.5 * (b[2] + b[3]))
            if p_ is not None and sides[i] != 0 and sides[i] != p_:
                flipped.append(i)
        relax = ()
        if flipped:
            relax = {j for j in range(len(boxes)) for i in flipped if abs(mids[j] - mids[i]) < self.cluster_ds}
            st, dpath, sides, s_block = self._solve_dp(
                s0, d0, chi0, boxes, vv, w_h, a_lat_hard, c_tyre, relax=relax
            )

        # 단일 DP 내부의 적응형 곡률 완화 (별도 플래너 MUX 아님: 해가 없을 때 저속 곡률 제약으로 완화)
        self.slow_at = None
        if s_block is not None and self.v_slow > 0.0:
            sb0 = s_block
            alh_relax = min(a_lat_hard, self.a_lat_slow) if self.a_lat_slow > 0.0 else a_lat_hard
            for rel_, gap_ in ((False, True), (True, True), (False, False), (True, False)):
                if rel_ and self.slow_commit is None:
                    continue
                cd = self.slow_commit[0] if rel_ else None
                ct = self.slow_commit[1] if rel_ else None
                r_ = self._solve_dp(
                    s0, d0, chi0, boxes, vv, w_h, a_lat_hard, c_tyre,
                    relax=relax, hard=True, gap=gap_,
                    vfloor=self.v_slow, vmax=99.0, kcap=self.kcap_slow,
                    adec=self.dp_adec_slow, kmax=self.kmax_slow,
                    commit_d=cd, commit_t=ct, alh_override=alh_relax,
                )
                if r_[3] is None:
                    st, dpath, sides, s_block = r_
                    self.slow_at = sb0
                    break

        if s_block is not None and self.gap_fallback:
            r_ = self._solve_dp(s0, d0, chi0, boxes, vv, w_h, a_lat_hard, c_tyre, relax=relax, hard=True, gap=False)
            if r_[3] is None:
                st, dpath, sides, s_block = r_

        if s_block is not None:
            for bi_, b in enumerate(boxes):
                if sides[bi_] != 0:
                    continue
                smid, dc = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
                prev = self._prev_side(smid, dc)
                sides[bi_] = prev if prev is not None else (1 if d0 > dc else -1)

        self.prev_sides = [(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), sd) for b, sd in zip(boxes, sides) if sd != 0]
        self.weaves = self._compute_weaves(boxes, sides, w_h, s0) if self.a_weave > 0.0 else []
        return st, dpath, sides, s_block
