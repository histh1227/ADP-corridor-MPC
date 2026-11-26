"""최소 시간 속도 프로파일 및 시야 안전 제한 모듈.

기존 cmpc_berlin_tt.py의 _speed_profile, _front_tables, _env 로직을 분리했다.
- TT/OBS 모드 전환 없이 모든 장애물 조건에서 단일 안전 규칙(시야 제한, 좁은 구간 감속, weave 제한) 적용.
"""
import math
import numpy as np

from common.numba_kernels import _cmd_sim, _fwd, _passes, _speed_k
from common.vehicle_params import A_MAX, BOX_FRONT


class SpeedProfiler:
    def __init__(self, track, estimator, p: dict):
        self.n = int(p['n'])
        self.ds = float(p['ds'])
        self.v_max = float(p['v_max'])
        self.a_acc = float(p['a_acc'])
        self.v_end_scale = float(p['v_end_scale'])
        self.vis_tol = float(p['vis_tol'])
        self.vis_start = float(p['vis_start'])
        self.m_vis = float(p['m_vis'])
        self.m_blk = float(p['m_blk'])
        self.t_react = float(p['t_react'])
        self.v_vis_safe = float(p['v_vis_safe'])
        self.v_hz = float(p['v_hz'])
        self.m_hz = float(p['m_hz'])
        self.t_lag = float(p['t_lag'])
        self.w_front = float(p['w_front'])
        self.hz_look = float(p['hz_look'])
        self.front_ramp = float(p['front_ramp'])
        self.em_frac = float(p['em_frac'])
        self.v_slow = float(p['v_slow'])
        self.slow_pre = float(p['slow_pre'])
        self.slow_len = float(p['slow_len'])
        self.m_slow = float(p['m_slow'])

        self.vlim0 = 0.0
        self.sp_cum = None
        self.build_front_tables(track, estimator)

        # Numba 커널 워밍업
        _passes(np.ones(3), np.ones(2), estimator.tab_plan)
        _fwd(np.ones(3), np.ones(2), 1.0, 1.0, estimator.tab_plan)
        _cmd_sim(np.ones(3), np.arange(3.0), np.ones(2), 1.0, 0.0, 1.0, 1.0, estimator.tab_emerg, 0.02)
        z1 = np.zeros(self.n + 1)
        for sblk in (math.nan, 1.0):
            _speed_k(np.arange(self.n + 1) * self.ds, z1, z1, z1, z1, 1.0, np.ones(1080), 0.0, 0.0, 0.0, sblk,
                     track.fx, track.fy, track.fnx, track.fny, track.fpsi, track.fvx, float(track.L), float(track.fds),
                     0.2, 8.0, 7.0, 1.0, 6.0, 0.1, 0.5, 0.1, 0.1, 0.04, 2.5, estimator.tab_plan, estimator.tab_emerg,
                     estimator.tab_dstop, estimator.tab_v)

    def build_front_tables(self, track, estimator):
        self.front_flag = None
        if self.w_front <= 0.0 or self.v_hz < 0.0:
            return
        W = track.whi_tab[0] - track.wlo_tab[0] + 0.06
        nf, fds = track.nf, track.fds
        H = self.n * self.ds
        nar = W < self.w_front
        k0, k1 = int(round((H - 0.5) / fds)), int(round((H + self.hz_look) / fds))
        idx = (np.arange(nf)[:, None] + np.arange(k0, k1 + 1)[None, :]) % nf
        self.front_flag = nar[idx].any(1)
        vs = estimator.tab_v
        reach = H - BOX_FRONT - self.m_hz - vs * (self.t_react + self.t_lag) + float(
            np.interp(self.v_hz, vs, estimator.tab_dstop)
        )
        ok = estimator.tab_dstop <= reach
        v_fc = float(vs[ok][-1]) if ok.any() else self.v_hz
        cap = np.where(self.front_flag, v_fc, np.inf)
        for _ in range(2):
            for i in range(nf - 1, -1, -1):
                cap[i] = min(cap[i], math.sqrt(cap[(i + 1) % nf] ** 2 + 2.0 * self.front_ramp * fds))
        self.front_cap = cap
        self.v_fc = v_fc

    def _env(self, cum, k, v0, vt, m, estimator, frac=1.0):
        reach = cum[k] - BOX_FRONT - m - cum - v0 * (self.t_react + self.t_lag)
        tds = estimator.tab_dstop / frac
        ds_ = float(np.interp(vt, estimator.tab_v, tds))
        env = np.interp(np.maximum(reach, 0.0) + ds_, tds, estimator.tab_v)
        return np.where(reach <= 0.0, np.minimum(env, vt), env)

    def compute_profile(self, st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block,
                        track, estimator, weaves, slow_at, a_lat):
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        vprof, vpred, acc, dpl, kv, v0_lim, self.sp_cum = _speed_k(
            f64(st), f64(d), f64(mu), f64(beta), f64(kr), float(v0), f64(scan), float(lx), float(ly), float(yaw),
            math.nan if s_block is None else float(s_block), track.fx, track.fy, track.fnx, track.fny, track.fpsi,
            track.fvx, float(track.L), float(track.fds), float(self.ds), float(self.v_max), float(a_lat),
            float(self.v_end_scale), float(self.a_acc), float(self.vis_tol), float(self.vis_start),
            float(self.m_vis), float(self.m_blk), float(self.t_react), float(self.v_vis_safe),
            estimator.tab_plan, estimator.tab_emerg, estimator.tab_dstop, estimator.tab_v,
        )
        self.vlim0 = float(v0_lim)
        s_vis = float(st[kv]) if kv >= 0 else None

        cum = self.sp_cum
        envs = []
        if self.v_hz >= 0.0:
            gated = False
            if self.front_flag is not None:
                ii = (np.floor(np.mod(st, track.L) / track.fds).astype(np.int64)) % track.nf
                envs.append(self.front_cap[ii])
                gated = not bool(self.front_flag[ii[0]])
            if not gated:
                envs.append(self._env(cum, cum.size - 1, v0, self.v_hz, self.m_hz, estimator))
        if self.em_frac < 1.0 and s_block is not None:
            kb = min(int(np.searchsorted(st, s_block)), cum.size - 1)
            envs.append(self._env(cum, kb, v0, 0.0, self.m_blk, estimator, self.em_frac))
        for s_w, s_w2, v_w in weaves:
            k = min(int(np.searchsorted(st, s_w)), cum.size - 1)
            k2 = int(np.searchsorted(st, s_w2))
            if k2 > 0:
                e3 = self._env(cum, k, v0, v_w, -BOX_FRONT, estimator)
                e3[k2 + 1:] = np.inf
                envs.append(e3)
        if self.v_slow > 0.0 and slow_at is not None:
            s_t = slow_at - self.slow_pre
            k = min(int(np.searchsorted(st, s_t)), cum.size - 1)
            e2 = self._env(cum, k, v0, self.v_slow, self.m_slow - BOX_FRONT, estimator)
            e2[(st > s_t + self.slow_len)] = np.inf
            envs.append(e2)

        env = np.min(envs, axis=0) if envs else None
        if env is not None and np.any(env < vprof):
            vl = np.minimum(vprof, env)
            vprof, vpred = _fwd(np.ascontiguousarray(vl), np.ascontiguousarray(dpl, dtype=np.float64),
                                float(v0), float(self.a_acc), estimator.tab_emerg)
            acc = np.clip((vpred[1:] ** 2 - vpred[:-1] ** 2) / (2 * dpl), -A_MAX, A_MAX)
            self.vlim0 = min(self.vlim0, float(vl[0]))
        return vprof, vpred, acc, dpl, s_vis
