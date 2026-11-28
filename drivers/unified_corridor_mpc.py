import math
import sys
import numpy as np

from common.numba_kernels import _cmd_sim, _plan_prep_k
from common.vehicle_params import BOX_HW, KP_ACC, KP_BRK, LIDAR_X, STEER_MAX, wrap_angle
from control.model_estimator import OnlineModelEstimator
from control.spatial_mpc import SpatialLTVMPC
from control.speed_profiler import SpeedProfiler
from perception.obstacle_detector import ObstacleDetector
from planning.corridor_dp import CorridorDPPlanner
from planning.raceline_loader import RacelineTracker

# 제거된 MUX/후진/모드전환/크기가정 파라미터 목록
REMOVED_PARAMS = (
    'obs_mem', 'obs_kw',
    'back_dist', 'back_after', 'back_max',
    'man_v', 'man_vf', 'man_lens', 'man_m_wall', 'man_m_obs', 'man_tol',
    'man_goal', 'man_w_len', 'man_w_mu', 'man_w_clr', 'man_beam',
    'man_min_prog', 'man_happy', 'man_close',
    'hidden_depth', 'w_max', 'min_width',
)


class UnifiedCorridorMPC:
    def __init__(self, **kw):
        p = dict(
            # 속도 및 종방향
            v_max=8.5, a_lat=7.5, a_lat_hard=7.5, a_acc=6.0, a_dec=9.0, c_plan=0.6, c_emerg=0.3,
            acc_corner_min=1.0,
            m_vis=0.15, m_blk=0.30, jerk=60.0, v_min_plan=0.8, vis_tol=0.12, vis_start=0.6,
            t_react=0.04, v_end_scale=1.0, v_vis_safe=2.0,
            # 기하 여유
            m_hard=0.04, m_hard_v=0.006, c_pref=0.12, wall_clear=0.16, wall_clear_v=0.02,
            wall_clear_vmax=4.0, wall_pref=0.28, obs_pad_s=0.06, obs_pad_d=0.02,
            # 인지 (Perception)
            obs_edt=0.09, r_max=9.0, link=0.22, blind_after_wrap=0.07, carve=True,
            scan_median_win=0,
            # 계획 (DP Corridor)
            ds=0.2, n=40, dd=0.05, dp_ds=0.5, dp_kmax=6, dp_dmax=1.8, dp_vmax=6.0, dp_adec=4.0,
            dp_near=0.6, p_near=30.0, w_dev=1.0, w_sl=2.0, w_cv=20.0, dp_kcap=0.8, dp_vfloor=2.0,
            w_soft=40.0, w_wall=10.0, p_flip=3.0, p_flip_hard=40.0, commit_t=0.45, commit_d=1.0,
            commit_ramp=1.5, young_t=0.15, dp_kgeo=0.95, w_geo=10.0, cluster_ds=3.0,
            release_margin=0.0, min_gap_w=0.10, min_gap_keep=0.0, gap_fallback=True,
            g_narrow=-1.0, g_diff=0.25,
            v_slow=-1.0, kcap_slow=1.1, kmax_slow=6, dp_adec_slow=5.0, slow_pre=2.0, slow_len=4.0,
            m_slow=0.1, slow_commit=None, a_lat_slow=0.0,
            a_weave=0.0, v_weave_min=1.0, weave_pre=0.3,
            # 시야/좁은구간 속도 제한
            v_hz=-1.0, m_hz=0.15, t_lag=0.0, w_front=0.0, hz_look=2.5, front_ramp=3.0, em_frac=1.0,
            # 제어 (Spatial LTV-MPC)
            q_d=2.0, q_mu=0.05, q_a=0.02, r_w=0.02, rho1=200.0, rho2=2000.0, rhoa1=50.0, rhoa2=50.0,
            slack_brake=0.03, lpts=(0.0, 0.22, 0.435), slack_near=20.0, slack_len=1.5, slack_reach=2.0,
            slack_bad=0.15, hp_iter=30, split=True, hp_tol=(1e-4, 1e-5, 1e-5, 1e-5), hp_mode=1,
            reuse_prof=True, ref_dp=True, resolve_da=2.0, resolve_n=10,
            # 온라인 모델 추정 및 필터
            k_obs=0.5, t_lat=0.03, t_buf=0.02, creep_after=1.0, creep_v=0.6,
            lat_est=True, lat_min=0.0, lat_max=0.15, lat_tau=4.0, lat_gain=0.5, lat_cap=0.3,
            lat_hyst=0.05, hold_comp=True, c_tyre=1.0, pose_filt=True, noise_alpha=0.03,
            noise_n0=25, q_pos=0.002 ** 2, q_yaw=0.001 ** 2, q_v=0.02 ** 2, q_r=0.03 ** 2,
            beta_sig=0.02, pose_gate=0.3, c_adapt=True, c_grid=(0.6, 1.4, 9), c_tau=3.0,
            c_prior=1.0, c_vmin=2.0, l_grid=(-0.02, 0.05, 8), l_prior=100.0, c_tables=True,
            t_call=0.02, em_jerk=400.0, soft_em=0.08, soft_em_v=1.0, soft_em_arm=1.5,
            lat_noise_floor=True, fg_check=True, fg_win=0.5, fg_thr=0.3, fg_thr_r=0.35,
            fg_k=4.0, fg_n=3, pose_out=4.0, pose_out_floor=0.03, yaw_out_floor=0.02,
            stall_db=0.1, stall_min=0.85,
            # 저속 안전 가드 (전진 전용)
            creep_slack=-1.0, creep_dist=0.8, creep_chk=0.0, creep_m_obs=0.04, creep_m_wall=0.03,
            prox_v=0.0, prox_t=0.4, mu_max=0.0,
            # 맵 및 레이싱라인
            line_ktrig=0.9, line_kmax=1.0, line_dkmax=2.0, line_clr=0.35, line_wdev=0.001,
            map_name='berlin', raceline_file=None, lat_alat=None,
        )
        for k, v in kw.items():
            if k in REMOVED_PARAMS:
                raise KeyError(
                    f'파라미터 {k}는 단일 플래너/제어기 및 후진 불가 규정 준수를 위해 제거되었습니다.'
                )
            if k not in p:
                raise KeyError(f'알 수 없는 파라미터: {k}')
            p[k] = v

        self.p = p
        for k, v in p.items():
            if k not in ("t_lat", "c_tyre"):
                setattr(self, k, v)
        self.lpts = tuple(self.lpts)

        # 6개 핵심 서브모듈 초기화
        self.track = RacelineTracker(
            map_name=self.map_name, raceline_file=self.raceline_file,
            wall_clear=self.wall_clear, wall_pref=self.wall_pref,
            line_ktrig=self.line_ktrig, line_kmax=self.line_kmax,
            line_dkmax=self.line_dkmax, line_clr=self.line_clr, line_wdev=self.line_wdev,
        )
        self.detector = ObstacleDetector(
            self.track, obs_edt=self.obs_edt, r_max=self.r_max, link=self.link,
            carve=self.carve, obs_pad_s=self.obs_pad_s, obs_pad_d=self.obs_pad_d,
            ds=self.ds, n_stations=self.n, creep_m_obs=self.creep_m_obs,
            creep_m_wall=self.creep_m_wall, scan_median_win=self.scan_median_win,
        )
        self.estimator = OnlineModelEstimator(self.p)
        self.planner = CorridorDPPlanner(self.track, self.p)
        self.mpc = SpatialLTVMPC(self.track, self.estimator, self.p)
        self.speed_profiler = SpeedProfiler(self.track, self.estimator, self.p)

        # _plan_prep_k 워밍업
        stw = np.arange(self.n + 1) * self.ds
        _plan_prep_k(stw, self.track.fk, float(self.track.L), float(self.track.fds), stw,
                     np.zeros((self.n + 1, 5)), stw * 0.0, np.zeros(5), stw, stw[:-1] * 0.0 + self.ds,
                     stw * 0.0 + 1.0, 1.0, 0.0, 6.0, 60.0, self.estimator.tab_emerg, 0.02)

        self.n_bad_scan = 0
        self.reset()

    # ---- LidarLocDriver 호환 프로퍼티 (외부 래퍼가 참조하는 속성 연결)
    @property
    def t_lat(self):
        return self.estimator.t_lat

    @t_lat.setter
    def t_lat(self, val):
        self.estimator.t_lat = float(val)

    @property
    def c_tyre(self):
        return self.estimator.c_tyre

    @property
    def hist(self):
        return self.estimator.hist

    @property
    def t_period(self):
        return self.estimator.t_period

    def _xy(self, s, d):
        return self.track.xy(s, d)

    def reset(self):
        self.estimator.reset()
        self.detector.reset()
        self.planner.reset()
        self.mpc.reset()
        self.fi = None
        self.s_prev = None
        self.prof = None
        self.t_stop = None
        self.dp_res = None
        self.plan = None
        self.phase = 0
        self.a_cmd = 0.0
        self.t_prev_cmd = None
        self.t_off = None
        self.t_int = None
        self.pose_prev = None
        self.n_invalid = 0
        self.odo = 0.0
        self.t_odo = None
        self._creep_blocked = False

    def _clock(self, t):
        if self.t_off is None:
            self.t_off = t
        ti = t - self.t_off
        if self.t_int is not None and ti < self.t_int:
            self.t_off -= self.t_int - ti
            ti = self.t_int
        self.t_int = ti
        return ti

    def _plan(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv):
        N = self.n
        est = self.estimator
        tr = self.track
        if self.mpc.sol is not None and not self.mpc.sol_bad and self.reuse_prof and self.prof is not None:
            sp_, vpp, dplp, vpf = self.prof
            kr, Xbar, dpl, vpred, acc = _plan_prep_k(
                st, tr.fk, float(tr.L), float(tr.fds), self.mpc.sol[0], self.mpc.sol[1],
                dpath, X0, sp_, dplp, vpf, float(pv), float(self.a_cmd), float(self.a_acc),
                float(self.jerk), est.tab_emerg, float(est.t_period),
            )
            return self._plan_rest(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv,
                                   kr, Xbar, dpl, vpred, acc)

        kr = tr.finterp(tr.fk, st)
        if self.mpc.sol is not None and not self.mpc.sol_bad:
            ss_, Xp, wp = self.mpc.sol
            Xbar = np.stack([np.interp(st, ss_, Xp[:, k]) for k in range(5)], 1)
            beyond = st > ss_[-1]
            Xbar[beyond, 0] = dpath[beyond]
            Xbar[beyond, 1:] = 0.0
        else:
            Xbar = np.zeros((N + 1, 5))
            Xbar[:, 0] = dpath
            Xbar[1:, 1] = np.arctan(np.gradient(dpath, st))[1:]
        Xbar[0] = X0

        if self.reuse_prof and self.prof is not None:
            sp_, vpp, dplp, vpf = self.prof
            vpred = np.interp(st, sp_, vpp)
            vpred[0] = pv
            dpl = np.interp(st[:-1], sp_[:-1], dplp)
            vtarget = np.interp(st, sp_, vpf)
        else:
            vtarget, vpred, acc, dpl, _ = self.speed_profiler.compute_profile(
                st, Xbar[:, 0], Xbar[:, 1], Xbar[:, 2], kr, pv, scan, lx, ly, yaw, s_block,
                tr, est, self.planner.weaves, self.planner.slow_at, self.a_lat,
            )

        cum0 = np.concatenate([[0.0], np.cumsum(dpl)])
        vpred, acc = _cmd_sim(
            vtarget.astype(np.float64), cum0, dpl.astype(np.float64), float(pv),
            float(self.a_cmd), float(self.a_acc), float(self.jerk), est.tab_emerg, float(est.t_period),
        )
        return self._plan_rest(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv,
                               kr, Xbar, dpl, vpred, acc)

    def _plan_rest(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv,
                   kr, Xbar, dpl, vpred, acc):
        N = self.n
        est = self.estimator
        tr = self.track
        acc_mpc = acc.copy()
        out = self.mpc.solve(st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, dpath,
                             tr, est.tab_plan, est.c_tyre, self.a_lat_hard)
        slack = None
        if out is not None:
            Xs, w, slack, dts = out
            self.mpc.sol = (st, Xs, w)
            self.mpc.n_fail = 0
        else:
            self.mpc.n_fail += 1
            Xs, dts = Xbar, dpl / np.maximum(vpred[:N], self.v_min_plan)

        vprof, vpred, acc, dpl, s_vis = self.speed_profiler.compute_profile(
            st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv, scan, lx, ly, yaw, s_block,
            tr, est, self.planner.weaves, self.planner.slow_at, self.a_lat,
        )
        if self.resolve_da > 0.0 and out is not None:
            v2, acc2 = _cmd_sim(
                vprof.astype(np.float64), self.speed_profiler.sp_cum, dpl.astype(np.float64), float(pv),
                float(self.a_cmd), float(self.a_acc), float(self.jerk), est.tab_emerg, float(est.t_period),
            )
            kn = self.resolve_n
            if float(np.max(np.abs(acc2[:kn] - acc_mpc[:kn]))) > self.resolve_da:
                out2 = self.mpc.solve(st, X0, Xs, kr, v2, acc2, dpl, boxes, sides, w_h, dpath,
                                      tr, est.tab_plan, est.c_tyre, self.a_lat_hard)
                if out2 is not None:
                    out = out2
                    Xs, w, slack, dts = out2
                    self.mpc.sol = (st, Xs, w)
                    acc_mpc = acc2.copy()
                    vprof, vpred, acc, dpl, s_vis = self.speed_profiler.compute_profile(
                        st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv, scan, lx, ly, yaw, s_block,
                        tr, est, self.planner.weaves, self.planner.slow_at, self.a_lat,
                    )
                    self.mpc.n_resolve += 1

        self.prof = (st, vpred, dpl, vprof)
        cum = self.speed_profiler.sp_cum
        emergency = False
        bad = out is None or (slack is not None and float(np.max(slack)) > self.slack_bad)
        self.mpc.sol_bad = bad
        if slack is not None:
            near = cum <= est.dstop(vv) + self.slack_reach
            if np.any(slack[near] > self.slack_brake):
                emergency = True
        if out is None and self.mpc.n_fail > 2:
            emergency = True
        if bad:
            emergency = True
        if (emergency and self.soft_em > 0.0 and not bad and out is not None and s_block is None
                and vv <= self.soft_em_arm and slack is not None):
            near = cum <= est.dstop(vv) + self.slack_reach
            mx = float(np.max(slack[near])) if np.any(near) else 0.0
            if mx <= self.soft_em:
                emergency = False
                vprof = np.minimum(vprof, self.soft_em_v)
        return dict(t=t, st=st, Xs=Xs, dts=dts, vprof=vprof, cum=cum, emergency=emergency,
                    slack=slack, s_vis=s_vis, bad=bad, acc=acc_mpc, failed=out is None)

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        x, y, yaw, v, yaw_rate, t = float(x), float(y), float(yaw), float(v), float(yaw_rate), float(t)
        scan = np.asarray(scan, dtype=float)
        est = self.estimator
        tr = self.track

        bad_scan = scan.shape != self.detector.ca.shape
        if bad_scan:
            self.n_bad_scan += 1
            if self.n_bad_scan == 1:
                print(f'UnifiedCorridorMPC: scan size {scan.size} != {self.detector.ca.size}',
                      file=sys.stderr, flush=True)
        if bad_scan or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)
                            and math.isfinite(v) and math.isfinite(yaw_rate) and math.isfinite(t)):
            self.n_invalid += 1
            if not est.hist:
                return 0.0, 0.0
            steer_h = est.hist[-1][1]
            speed_h = max(0.0, est.hist[-1][2]) if self.n_invalid <= 5 else 0.0
            if speed_h != est.hist[-1][2]:
                t_h = max(t - self.t_off, est.hist[-1][0]) if (math.isfinite(t) and self.t_off is not None)                     else est.hist[-1][0] + self.n_invalid * est.t_period
                est.record_command(t_h, steer_h, speed_h)
            return steer_h, speed_h

        self.n_invalid = 0
        t = self._clock(t)
        if self.t_odo is not None and t > self.t_odo:
            self.odo += abs(v) * min(t - self.t_odo, 0.2)
        self.t_odo = t

        scan = self.detector.filter_scan_noise(scan)
        N = self.n

        # 텔레포트(포즈 점프) 감지
        teleport = False
        if self.pose_prev is not None:
            tp_, xp_, yp_, vp_ = self.pose_prev
            teleport = math.hypot(x - xp_, y - yp_) > 0.5 + 1.5 * max(abs(v), abs(vp_)) * (t - tp_)
        jump_d = 0.0 if self.pose_prev is None else math.hypot(x - self.pose_prev[1], y - self.pose_prev[2])
        self.pose_prev = (t, x, y, v)
        if teleport:
            self.fi = None
            self.s_prev = None
            est.last = None
            self.mpc.sol_bad = True
            self.prof = None
            self.dp_res = None
            if jump_d > 2.0:
                self.detector.mem_clear()
                self.planner.clear_memory()

        # 1. 온라인 타이밍, 지연, 타이어 강성 추정 및 상태 예측
        est.trim_history(t)
        self.a_lat, self.a_lat_hard = est.update_timing_and_alat(t, self.p['a_lat'], self.p['a_lat_hard'])
        if self.lat_est:
            est.update_latency(t, v)
        v_raw = v
        x, y, yaw, v, yaw_rate = est.observe(t, x, y, yaw, v, yaw_rate)
        if self.fg_check and (self.lat_est or self.c_adapt):
            est.check_command_following(t, v_raw)
        if self.c_tables:
            est.set_tables(est.c_tyre)
        t_bite = est.t_lat + est.t_buf
        px, py, pyaw, pv, pbeta, pr, pdelta, pa = est.predict(t, x, y, yaw, v, yaw_rate, t_bite)
        vv = max(v, 0.0)

        # 2. Frenet 좌표 및 랩 전환 처리
        s_c, d_c, self.fi = tr.frenet(x, y, self.fi)
        wrapped = False
        if self.s_prev is not None and self.s_prev - s_c > tr.L / 2:
            wrapped = True
            self.detector.mem_clear()
            self.detector.blind_until = t + self.blind_after_wrap
            self.planner.clear_memory()
            if self.mpc.sol is not None:
                self.mpc.sol = (self.mpc.sol[0] - tr.L,) + self.mpc.sol[1:]
            if self.prof is not None:
                self.prof = (self.prof[0] - tr.L,) + self.prof[1:]
        self.s_prev = s_c

        s0, d0, _ = tr.frenet(px, py, self.fi)
        if s0 - s_c < -tr.L / 2:
            s0 += tr.L
        elif s0 - s_c > tr.L / 2:
            s0 -= tr.L
        mu0 = float(wrap_angle(pyaw - tr.heading(s0)))
        X0 = np.array([d0, mu0, pbeta, pr, pdelta])

        # 3. 장애물 인지 (Perception)
        lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
        if not teleport:
            self.detector.perceive(scan, lx, ly, yaw, t)

        full = (self.dp_res is None or self.plan is None or not self.split or wrapped or t - self.plan['t'] > 0.2)
        do_dp = full or self.phase == 0
        do_mpc = full or self.phase == 1 or t - self.plan['t'] > 1.75 * est.t_period
        self.phase = 1 - self.phase if not full else 1

        w_h = BOX_HW + self.m_hard + self.m_hard_v * vv
        wc_run = self.wall_clear + self.wall_clear_v * min(vv, self.wall_clear_vmax)
        tr.set_wall(wc_run)

        # 4. 단일 DP 통로 계획 (Planning)
        if do_dp:
            boxes = self.detector.extract_boxes(s0)
            st, dpath, sides, s_block = self.planner.plan_corridor(
                s0, d0, mu0 + pbeta, boxes, vv, w_h, t, self.a_lat_hard, est.c_tyre,
            )
            self.dp_res = (boxes, sides, s_block, st, dpath)
        else:
            boxes, sides, s_block, st_old, dpath_old = self.dp_res
            st = s0 + self.ds * np.arange(N + 1)
            dpath = np.interp(st, st_old, dpath_old)

        # 5. 단일 공간 LTV-MPC 및 속도 프로파일 (Control)
        if do_mpc:
            new = self._plan(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv)
            old = self.plan
            if not (new['failed'] and old is not None and not old['bad'] and self.mpc.n_fail <= 2
                    and not full and t - old['t'] < 0.12):
                self.plan = new

        plan = self.plan
        Xs, dts, vprof, cum, emergency, slack = (
            plan['Xs'], plan['dts'], plan['vprof'], plan['cum'], plan['emergency'], plan['slack']
        )
        shift_t = t - plan['t']
        shift_s = s0 - plan['st'][0]
        s_ahead = max(0.3, vv * 0.10)
        v_tgt = float(np.interp(s_ahead + shift_s, cum, vprof))
        a_em = est.abrake(est.tab_emerg, vv)

        if vv < 0.15 and (emergency or s_block is not None):
            if self.t_stop is None:
                self.t_stop = t
        elif vv > 0.5:
            self.t_stop = None
        creep = self.t_stop is not None and t - self.t_stop > self.creep_after
        self._creep_blocked = bool(plan['bad'])
        if self.creep_slack >= 0.0 and slack is not None:
            near_ = cum <= self.creep_dist
            if np.any(near_) and float(np.max(slack[near_])) > self.creep_slack:
                self._creep_blocked = True
        if creep:
            emergency = False

        if emergency:
            a_des = -a_em
        else:
            a_des = (v_tgt * v_tgt - pv * pv) / (2 * s_ahead)
            if a_des > 0:
                load = min(1.0, abs(vv * yaw_rate) / self.a_lat)
                a_des = min(a_des, self.a_acc * max(self.acc_corner_min, 1.0 - load * load))
            if creep:
                a_des = (self.creep_v - vv) * 4.0
            a_des = min(max(a_des, -a_em), self.a_acc)

        dt_c = 0.02 if self.t_prev_cmd is None else min(0.1, max(0.005, t - self.t_prev_cmd))
        j_dn = self.jerk if (self.em_jerk is None or not emergency) else self.em_jerk
        a_des = min(max(a_des, self.a_cmd - j_dn * dt_c), self.a_cmd + self.jerk * dt_c)
        self.t_prev_cmd = t
        self.a_cmd = a_des

        v_at = vv + pa * est.t_lat
        kp = KP_ACC if a_des >= 0 else KP_BRK
        if self.hold_comp:
            speed = v_at + a_des * est.t_period / (1.0 - math.exp(-kp * est.t_period))
        else:
            speed = v_at + a_des / kp
        speed = max(0.0, speed)
        if vv < 0.05 and a_des <= 0:
            speed = 0.0
        speed = min(speed, self.v_max + 2.0)

        tcum = np.concatenate([[0.0], np.cumsum(dts)])
        steer = float(np.interp(est.t_period + shift_t, tcum, Xs[:, 4]))
        steer = min(max(steer, -STEER_MAX), STEER_MAX)

        # 저속 안전 가드 (전진 전용, 후진 원호 탐색 제거됨)
        veto = self.mu_max > 0.0 and speed > 0.0 and abs(mu0) > self.mu_max
        if not veto and self.prox_v > 0.0 and speed > 0.0 and vv < self.prox_v and not plan['bad']:
            veto = not self.detector.plan_clear(plan, max(self.creep_chk, self.prox_t * vv), shift_s)
        if (self.creep_chk > 0.0 and self.creep_slack >= 0.0 and self._creep_blocked and speed > 0.0
                and vv < 1.5 and not veto):
            veto = not self.detector.fwd_clear(x, y, yaw, steer, max(self.creep_chk, 0.3 * vv))
        if veto:
            self._creep_blocked = True
            speed = 0.0
            self.a_cmd = min(self.a_cmd, 0.0)
            if self.t_stop is None and vv < 0.15:
                self.t_stop = t

        # 후진 불가 규정 최종 보장
        speed = max(0.0, float(speed))
        est.record_command(t, steer, speed)
        return steer, speed


# 기존 import 호환용 별칭
CorridorMPC = UnifiedCorridorMPC
