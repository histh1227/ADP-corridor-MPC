"""온라인 차량 동역학, 명령 지연 및 타이어 강성 추정 모듈.

기존 cmpc_berlin_tt.py의 _observe, _predict, _lat_update, _fg_window, _gain,
_decel_tables, _set_tables, _c_estimate 로직을 분리했다.
"""
import math
import numpy as np

from common.numba_kernels import _lat_sim_k, _mm_prop, _obs_beta_k, _obs_main_k, _predict_k
from common.vehicle_params import A_MAX, WB, k_us, wrap_angle


class OnlineModelEstimator:
    def __init__(self, p: dict):
        self.a_dec = float(p['a_dec'])
        self.c_plan = float(p['c_plan'])
        self.c_emerg = float(p['c_emerg'])
        self.t_react = float(p['t_react'])
        self.k_obs = float(p['k_obs'])
        self.t_lat = float(p['t_lat'])
        self.t_buf = float(p['t_buf'])
        self.lat_est = bool(p['lat_est'])
        self.lat_min = float(p['lat_min'])
        self.lat_max = float(p['lat_max'])
        self.lat_tau = float(p['lat_tau'])
        self.lat_gain = float(p['lat_gain'])
        self.lat_cap = float(p['lat_cap'])
        self.lat_hyst = float(p['lat_hyst'])
        self.lat_noise_floor = bool(p['lat_noise_floor'])
        self.c_tyre = float(p['c_tyre'])
        self.pose_filt = bool(p['pose_filt'])
        self.noise_alpha = float(p['noise_alpha'])
        self.noise_n0 = int(p['noise_n0'])
        self.q_pos = float(p['q_pos'])
        self.q_yaw = float(p['q_yaw'])
        self.q_v = float(p['q_v'])
        self.q_r = float(p['q_r'])
        self.beta_sig = float(p['beta_sig'])
        self.pose_gate = float(p['pose_gate'])
        self.pose_out = float(p['pose_out'])
        self.pose_out_floor = float(p['pose_out_floor'])
        self.yaw_out_floor = float(p['yaw_out_floor'])
        self.c_adapt = bool(p['c_adapt'])
        self.c_grid = tuple(p['c_grid'])
        self.c_tau = float(p['c_tau'])
        self.c_prior = float(p['c_prior'])
        self.c_vmin = float(p['c_vmin'])
        self.l_grid = tuple(p['l_grid'])
        self.l_prior = float(p['l_prior'])
        self.c_tables = bool(p['c_tables'])
        self.t_call = float(p['t_call'])
        self.fg_check = bool(p['fg_check'])
        self.fg_win = float(p['fg_win'])
        self.fg_thr = float(p['fg_thr'])
        self.fg_thr_r = float(p['fg_thr_r'])
        self.fg_k = float(p['fg_k'])
        self.fg_n = int(p['fg_n'])
        self.stall_db = float(p['stall_db'])
        self.stall_min = float(p['stall_min'])
        self.lat_alat = p['lat_alat']

        self.tab_v = np.arange(0.0, 15.0001, 0.05)
        self.tabs_c = np.linspace(*self.c_grid[:2], int(self.c_grid[2]))
        self.tabs = np.array([self._decel_tables(c) for c in self.tabs_c])
        self.tab_c = None
        self.set_tables(self.c_tyre)

        self.lat_c = np.round(np.arange(self.lat_min, self.lat_max + 1e-9, 0.01), 3)
        self.lat_J = np.zeros(self.lat_c.size)
        self.lat_j = int(np.argmin(np.abs(self.lat_c - self.t_lat)))
        self.lat_W = 0.0
        self.n_foreign = 0

        self.t_period = float(self.t_call)
        self.gaps = []
        self.t_last_call = None

        # Numba 커널 워밍업
        z2 = np.zeros(2)
        _obs_main_k(np.zeros((1, 3)), 0.03, 0.02, 0.0, 2, 1.0, 0.0)
        _obs_beta_k(2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, z2, z2, np.zeros(2, dtype=np.bool_), 1.0)
        _mm_prop(np.zeros(3), np.zeros(3), np.zeros(1), np.ones(3), np.zeros(1), 2, np.zeros((1, 3)), 0.03, 0.02, 0.0,
                 1, 1.0, 1.0, np.zeros(1), np.zeros(1), np.ones(1))
        _predict_k(np.zeros((1, 3)), 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0, 0.0, 1.0)
        _lat_sim_k(np.zeros(2), np.ones(2), np.array([0.0, 0.03]), np.ones(2), 0.0, 2)

        self.reset()

    def _decel_tables(self, c):
        ag = np.linspace(-A_MAX, 0.0, 952)
        fac = WB + k_us(ag, c)[None, :] * self.tab_v[:, None] ** 2
        out = []
        for cw in (self.c_plan, self.c_emerg):
            ok = fac >= cw * WB
            amin = np.where(ok.any(1), ag[np.argmax(ok, axis=1)], 0.0)
            out.append(np.maximum(np.minimum(self.a_dec, -amin), 1.0))
        integ = self.tab_v / out[1]
        out.append(np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(self.tab_v))]))
        return out

    def set_tables(self, c):
        if self.tab_c is not None and abs(c - self.tab_c) < 0.005:
            return
        g = self.tabs_c
        i = min(max(int(np.searchsorted(g, c)) - 1, 0), g.size - 2)
        w = min(max((c - g[i]) / (g[i + 1] - g[i]), 0.0), 1.0)
        tp, te, td = (1.0 - w) * self.tabs[i] + w * self.tabs[i + 1]
        self.tab_plan = np.ascontiguousarray(tp)
        self.tab_emerg = np.ascontiguousarray(te)
        self.tab_dstop = np.ascontiguousarray(td)
        self.tab_c = c

    def reset(self):
        sn = getattr(self, 'fg_sn', None)
        if sn is not None:
            self.lat_J, self.lat_j, self.t_lat, self.lat_W = sn[0].copy(), sn[1], sn[2], sn[3]
            self.cg_J, self.c_tyre, self.lat_extra = sn[4].copy(), sn[5], sn[6]
        self.fg_sn = None
        self.fg_ok = 0
        self.hist = []
        self._H = np.zeros((0, 3), dtype=np.float64)
        self.lat_prev = None
        self.lat_v = None
        self.delta_est = 0.0
        self.beta_est = 0.0
        self.last = None
        self.lastf = None
        if not hasattr(self, 'nstat'):
            self.nstat = {k: [0.0, 0.0, None, 0] for k in ('v', 'r', 'yaw', 'pos')}
            self.c_grid_v = np.linspace(*self.c_grid[:2], int(self.c_grid[2]))
            self.l_grid_v = np.linspace(*self.l_grid[:2], int(self.l_grid[2]))
            self.cg_J = np.zeros(self.l_grid_v.size * self.c_grid_v.size)
            self.lat_extra = 0.0
            self.cg_prior = (
                self.c_prior * (self.c_grid_v[None, :] - 1.0) ** 2
                + self.l_prior * self.l_grid_v[:, None] ** 2
            )
        for st in self.nstat.values():
            st[2] = None
        nl_, nc_ = self.l_grid_v.size, self.c_grid_v.size
        self.c_vec = np.concatenate([[self.c_tyre], np.tile(self.c_grid_v, nl_)])
        self.cg_bb = np.zeros(self.c_vec.size)
        self.cg_rr = np.zeros(self.c_vec.size)
        self.cg_b = np.zeros(nl_ * nc_)
        self.cg_r = np.zeros(nl_ * nc_)
        self.cg_d = np.zeros(nl_)
        self.cg_on = False
        self.fg = None
        self.stalls = []
        self.out_pos = False
        self.out_yaw = False

    def trim_history(self, t: float):
        k_old = 0
        while k_old + 1 < len(self.hist) and self.hist[k_old + 1][0] < t - 0.3 - self.lat_max:
            k_old += 1
        self.hist = self.hist[k_old:]
        self._H = np.array(self.hist, dtype=np.float64).reshape(len(self.hist), 3)

    def record_command(self, t: float, steer: float, speed: float):
        self.hist.append((t, steer, speed))

    def update_timing_and_alat(self, t: float, base_a_lat: float, base_a_lat_hard: float):
        gap = None if self.t_last_call is None else t - self.t_last_call
        self.t_last_call = t
        a_lat = base_a_lat
        a_lat_hard = base_a_lat_hard
        if self.stall_db < 1.0:
            if gap is not None and 1.75 * self.t_period < gap:
                self.stalls.append((t, min(gap, 0.5) - self.t_period))
            while self.stalls and self.stalls[0][0] <= t - 5.0:
                self.stalls.pop(0)
            frac = sum(e[1] for e in self.stalls) / 5.0
            a_lat = base_a_lat * max(self.stall_min, 1.0 - max(frac - self.stall_db, 0.0))
        if self.lat_alat is not None:
            l0, l1, a1 = self.lat_alat
            f = min(max((self.t_lat - l0) / max(l1 - l0, 1e-6), 0.0), 1.0)
            ratio = 1.0 + f * (a1 / base_a_lat - 1.0)
            a_lat = min(a_lat if self.stall_db < 1.0 else base_a_lat, base_a_lat * ratio)
            a_lat_hard = base_a_lat_hard * ratio
        if gap is not None and 0.001 < gap < 0.25:
            self.gaps = self.gaps[-14:] + [gap]
            self.t_period = sorted(self.gaps)[len(self.gaps) // 2]
        return a_lat, a_lat_hard

    def _noise_R(self, key):
        st = self.nstat[key]
        return max(-st[1], 0.0) if st[3] >= self.noise_n0 else 0.0

    def update_latency(self, t: float, v: float):
        prev, self.lat_prev = self.lat_prev, (t, v)
        if prev is None or not self.hist:
            return
        t0, v0 = prev
        n = int(round((t - t0) / 0.01))
        if n <= 0 or n > 20 or v0 < 0.5 or v < 0.5:
            self.lat_v = None
            return
        R = self._noise_R('v') if self.lat_noise_floor else 0.0
        cap = max(self.lat_cap, 3.0 * math.sqrt(R))
        H = self._H.T.copy()
        vL = np.full(self.lat_c.size, float(v0)) if self.lat_v is None else self.lat_v
        vL, ok = _lat_sim_k(H[0], H[2], self.lat_c, vL, float(t0), n)
        e = np.minimum(np.maximum(v - vL, -cap), cap)
        self.lat_v = vL + self.lat_gain * (v - vL)
        if not ok:
            return
        lam = math.exp(-(t - t0) / self.lat_tau)
        self.lat_J = lam * self.lat_J + e * e
        self.lat_W = lam * self.lat_W + 1.0
        j = int(np.argmin(self.lat_J))
        cur = self.lat_J[self.lat_j]
        if self.lat_J[j] + 1e-5 < cur - self.lat_hyst * max(cur - R * self.lat_W, 0.0):
            self.lat_j = j
        self.t_lat = float(self.lat_c[self.lat_j])

    def _fg_snap(self):
        return (self.lat_J.copy(), self.lat_j, self.t_lat, self.lat_W, self.cg_J.copy(), self.c_tyre, self.lat_extra)

    def _fg_start(self, t, v, sn=None):
        if sn is None:
            sn = self.fg_sn if getattr(self, 'fg_sn', None) is not None else self._fg_snap()
        self.fg_sn = sn
        self.fg = [t, t, np.full(self.lat_c.size, float(v)), sn]
        self.fg_e2 = np.zeros(self.cg_J.size)
        self.fg_ne = 0

    def check_command_following(self, t: float, v: float):
        g = self.fg
        if not self.hist or v < 0.5:
            self.fg = None
            return
        if g is None:
            self._fg_start(t, v)
            return
        t_s, t_l, vO, sn = g
        n = int(round((t - t_l) / 0.01))
        ok = 0 < n <= 20
        if ok:
            H = self._H.T.copy()
            vO, ok = _lat_sim_k(H[0], H[2], self.lat_c, vO, float(t_l), n)
        if not ok:
            self._fg_start(t, v)
            return
        if t - t_s < self.fg_win - 1e-6:
            self.fg = [t_s, t, vO, sn]
            return
        ev = float(np.min(np.abs(v - vO)))
        er = math.sqrt(float(np.min(self.fg_e2)) / self.fg_ne) if self.fg_ne > 0 else 0.0
        self.fg_diag = (ev, er)
        thr = self.fg_thr + self.fg_k * math.sqrt(self._noise_R('v'))
        thr_r = self.fg_thr_r + self.fg_k * math.sqrt(self._noise_R('r'))
        if ev > thr or er > thr_r:
            self.lat_J, self.lat_j, self.t_lat, self.lat_W = sn[0].copy(), sn[1], sn[2], sn[3]
            self.cg_J, self.c_tyre, self.lat_extra = sn[4].copy(), sn[5], sn[6]
            self.n_foreign += 1
            self.fg_ok = 0
        else:
            self.fg_ok += 1
            if self.fg_ok >= self.fg_n:
                sn = self._fg_snap()
        self._fg_start(t, v, sn)

    def _gain(self, key, e, qmin):
        st = self.nstat[key]
        a = self.noise_alpha
        if st[2] is not None:
            st[1] += a * (sum(u * w for u, w in zip(e, st[2])) / len(e) - st[1])
            st[3] += 1
        st[0] += a * (sum(u * u for u in e) / len(e) - st[0])
        st[2] = e
        R = -st[1]
        if st[3] < self.noise_n0 or R <= 0.0:
            return 1.0
        q = max(st[0] - 2.0 * R, qmin)
        P = 0.5 * (q + math.sqrt(q * q + 4.0 * q * R))
        return P / (P + R)

    def _innov_sd(self, key, qmin):
        st = self.nstat[key]
        R = self._noise_R(key)
        q = max(st[0] - 2.0 * R, qmin)
        P = 0.5 * (q + math.sqrt(q * q + 4.0 * q * R))
        return math.sqrt(P + R)

    @staticmethod
    def _parab_min(g, J):
        k = min(max(int(np.argmin(J)), 1), g.size - 2)
        j0, j1, j2 = J[k - 1], J[k], J[k + 1]
        den = j0 - 2.0 * j1 + j2
        x = g[k] + (0.5 * (g[1] - g[0]) * (j0 - j2) / den if den > 1e-12 else 0.0)
        return float(min(max(x, g[0]), g[-1]))

    def _c_estimate(self):
        J = self.cg_J.reshape(self.l_grid_v.size, self.c_grid_v.size) + self.cg_prior
        self.lat_extra = self._parab_min(self.l_grid_v, J.min(axis=1))
        return self._parab_min(self.c_grid_v, J.min(axis=0))

    def observe(self, t, x, y, yaw, v, r):
        if self.last is None:
            self.last = (t, x, y, yaw, v, r)
            self.lastf = (x, y, yaw, v, r)
            self.cg_b[:] = self.beta_est
            self.cg_r[:] = r
            self.cg_d[:] = self.delta_est
            return x, y, yaw, v, r
        t0, x0, y0, yaw0, v0, r0 = self.last
        xf0, yf0, yawf0, vf0, rf0 = self.lastf
        n = max(int(round((t - t0) / 0.01)), 0)
        dtc = max(t - t0, 1e-3)
        n_gap = n
        if n > 50:
            t0 += 0.01 * (n - 50)
            n = 50
        short = n_gap <= 25
        cv = self.c_vec
        cv[0] = self.c_tyre
        bb = self.cg_bb
        rr = self.cg_rr
        bb[0], rr[0] = self.beta_est, rf0
        bb[1:], rr[1:] = self.cg_b, self.cg_r
        delta, vm, dmain, amain, vms, vck, hvk = _obs_main_k(
            self._H, float(self.t_lat), float(self.t_buf), float(t0), int(n), float(vf0), float(self.delta_est)
        )
        if short and n > 0:
            _mm_prop(bb, rr, self.cg_d, cv, self.l_grid_v, self.c_grid_v.size, self._H,
                     float(self.t_lat), float(self.t_buf), float(t0), int(n), float(vf0), float(v), dmain, amain, vms)
        filt = self.pose_filt and n > 0 and short
        kv = self._gain('v', ((v - v0) - (vm - vf0),), self.q_v) if filt else 1.0
        vf = vm + kv * (v - vm)
        rp = float(rr[0])
        kr = self._gain('r', ((r - r0) - (rp - rf0),), self.q_r) if filt else 1.0
        rf = rp + kr * (r - rp)
        dyaw = 0.5 * (rf0 + rf) * dtc
        inn_y = wrap_angle(yaw - yawf0 - dyaw)
        yaw_keep = yaw
        if filt:
            jump_y = abs(inn_y) > 0.5 * self.pose_gate
            outl_y = (not jump_y and self.pose_out > 0.0
                      and abs(inn_y) > self.pose_out * self._innov_sd('yaw', self.q_yaw) + self.yaw_out_floor)
            if outl_y and not self.out_yaw:
                kyw = 0.0
                self.nstat['yaw'][2] = None
                yaw_keep = float(wrap_angle(yawf0 + dyaw))
            elif jump_y or outl_y:
                kyw = 1.0
                self.nstat['yaw'][2] = None
            else:
                kyw = self._gain('yaw', (wrap_angle(yaw - yaw0) - dyaw,), self.q_yaw)
            self.out_yaw = outl_y and not self.out_yaw
        else:
            kyw = 1.0
        yawf = float(wrap_angle(yawf0 + dyaw + kyw * inn_y))
        dyw = float(wrap_angle(yawf - yawf0))
        beta, has_mid, bm, px_, py_ = _obs_beta_k(
            int(n), float(vf0), float(vf), float(rf0), float(rf), float(yawf0),
            dyw, float(xf0), float(yf0), float(self.beta_est), dmain, vck, hvk, float(self.c_tyre)
        )
        beta_mid = bm if has_mid else None
        x_keep, y_keep = x, y
        p_out = False
        if filt:
            inn = math.hypot(x - px_, y - py_)
            jump = inn > self.pose_gate
            outl = (not jump and self.pose_out > 0.0
                    and inn > self.pose_out * self._innov_sd('pos', self.q_pos) + self.pose_out_floor)
            if outl and not self.out_pos:
                kp = 0.0
                p_out = True
                self.nstat['pos'][2] = None
                x_keep, y_keep = px_, py_
            elif jump or outl:
                kp = 1.0
                self.nstat['pos'][2] = None
            else:
                kp = self._gain('pos', ((x - x0) - (px_ - xf0), (y - y0) - (py_ - yf0)), self.q_pos)
            self.out_pos = outl and not self.out_pos
            xf, yf = px_ + kp * (x - px_), py_ + kp * (y - py_)
        else:
            xf, yf = x, y
        dist = math.hypot(x - x0, y - y0)
        if self.k_obs > 0 and dist > 0.02 and v > 1.0 and beta_mid is not None and short and not p_out:
            chi = math.atan2(y - y0, x - x0)
            ymid = yawf0 + 0.5 * wrap_angle(yawf - yawf0)
            bmeas = wrap_angle(chi - ymid)
            Rp = max(-self.nstat['pos'][1], 0.0) if self.nstat['pos'][3] >= self.noise_n0 else 0.0
            kb = self.k_obs * self.beta_sig ** 2 / (self.beta_sig ** 2 + 2.0 * Rp / (dist * dist))
            beta += kb * (bmeas - beta_mid)
        beta = min(max(beta, -0.5), 0.5)
        self.delta_est, self.beta_est = delta, beta
        if self.c_adapt and n > 0:
            if short and vf0 > self.c_vmin and vf > self.c_vmin and self.cg_on:
                self.cg_b[:], self.cg_r[:] = bb[1:], rr[1:]
                err = r - self.cg_r
                if self.fg is not None:
                    self.fg_e2 += err * err
                    self.fg_ne += 1
                lam = math.exp(-dtc / self.c_tau)
                self.cg_J = lam * self.cg_J + err * err
                self.c_tyre = self._c_estimate()
            else:
                self.cg_b[:], self.cg_r[:] = beta, rf
                self.cg_d[:] = delta
                self.cg_on = vf > self.c_vmin
        self.last = (t, x_keep, y_keep, yaw_keep, v, r)
        self.lastf = (xf, yf, yawf, vf, rf)
        return xf, yf, yawf, vf, rf

    def predict(self, t, x, y, yaw, v, r, t_pred):
        return _predict_k(
            self._H, float(self.t_lat), float(self.t_buf), float(t), float(x), float(y),
            float(yaw), float(v), float(r), float(t_pred), float(self.delta_est),
            float(self.beta_est), float(self.c_tyre),
        )

    def abrake(self, tab, v: float) -> float:
        return tab[min(int(max(v, 0.0) * 20.0), tab.size - 1)]

    def dstop(self, v: float) -> float:
        if v <= 0:
            return 0.0
        return float(np.interp(v, self.tab_v, self.tab_dstop)) + v * self.t_react
