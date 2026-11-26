"""단일 공간(Spatial) LTV-MPC 및 HPIPM(ctypes) 솔버 모듈.

기존 cmpc_berlin_tt.py의 _Hpipm 래퍼, _build_hpipm, _mpc 로직을 분리했다.
"""
import math
import os
import numpy as np

from common.numba_kernels import _expm_fr_k, _hp_slack_k, _mpc_hp_k
from common.vehicle_params import STEER_RATE


class _Hpipm:
    """CasADi 내장 libhpipm.so를 ctypes로 직접 호출하는 경량 OCP QP IPM 래퍼."""

    def __init__(self, N, nx, nu, nbx, nbu, ng, ns, idxbx, idxbu, idxs, Zl, Zu, zl, zu, R, iter_max=30,
                 tol=(1e-5, 1e-6, 1e-6, 1e-6), mode=1):
        import ctypes as C
        import casadi
        path = os.path.join(os.path.dirname(casadi.__file__), 'libhpipm.so')
        lib = C.CDLL(path)
        self.C, self.lib, self.N = C, lib, N
        vp, ip, sz = C.c_void_p, C.c_int, C.c_size_t
        for name in ('d_ocp_qp_dim_strsize', 'd_ocp_qp_strsize', 'd_ocp_qp_sol_strsize', 'd_ocp_qp_ipm_arg_strsize',
                     'd_ocp_qp_ipm_ws_strsize'):
            getattr(lib, name).restype = sz
        lib.d_ocp_qp_dim_memsize.restype = sz
        lib.d_ocp_qp_dim_memsize.argtypes = [ip]
        for name in ('d_ocp_qp_memsize', 'd_ocp_qp_sol_memsize', 'd_ocp_qp_ipm_arg_memsize'):
            getattr(lib, name).restype = sz
            getattr(lib, name).argtypes = [vp]
        lib.d_ocp_qp_ipm_ws_memsize.restype = sz
        lib.d_ocp_qp_ipm_ws_memsize.argtypes = [vp, vp]
        keep = []

        def buf(n):
            b = C.create_string_buffer(int(n) + 64)
            keep.append(b)
            return C.addressof(b)

        def iarr(lst):
            a = np.ascontiguousarray(np.array(lst, dtype=np.int32))
            keep.append(a)
            return a

        self.dim = buf(lib.d_ocp_qp_dim_strsize())
        lib.d_ocp_qp_dim_create.argtypes = [ip, vp, vp]
        lib.d_ocp_qp_dim_create(N, self.dim, buf(lib.d_ocp_qp_dim_memsize(N)))
        lib.d_ocp_qp_dim_set_all.argtypes = [vp] * 7
        arrs = [iarr(a) for a in (nx, nu, nbx, nbu, ng, ns)]
        lib.d_ocp_qp_dim_set_all(*[a.ctypes.data for a in arrs], self.dim)
        self.qp = buf(lib.d_ocp_qp_strsize())
        lib.d_ocp_qp_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_create(self.dim, self.qp, buf(lib.d_ocp_qp_memsize(self.dim)))
        self.sol = buf(lib.d_ocp_qp_sol_strsize())
        lib.d_ocp_qp_sol_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_sol_create(self.dim, self.sol, buf(lib.d_ocp_qp_sol_memsize(self.dim)))
        self.arg = buf(lib.d_ocp_qp_ipm_arg_strsize())
        lib.d_ocp_qp_ipm_arg_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_ipm_arg_create(self.dim, self.arg, buf(lib.d_ocp_qp_ipm_arg_memsize(self.dim)))
        lib.d_ocp_qp_ipm_arg_set_default.argtypes = [ip, vp]
        lib.d_ocp_qp_ipm_arg_set_default(mode, self.arg)
        for nm, val, typ in (('iter_max', iter_max, C.c_int), ('tol_stat', tol[0], C.c_double),
                             ('tol_eq', tol[1], C.c_double), ('tol_ineq', tol[2], C.c_double),
                             ('tol_comp', tol[3], C.c_double)):
            f = getattr(lib, 'd_ocp_qp_ipm_arg_set_' + nm)
            f.argtypes = [vp, vp]
            v = typ(val)
            keep.append(v)
            f(C.addressof(v), self.arg)
        self.ws = buf(lib.d_ocp_qp_ipm_ws_strsize())
        lib.d_ocp_qp_ipm_ws_create.argtypes = [vp, vp, vp, vp]
        lib.d_ocp_qp_ipm_ws_create(self.dim, self.arg, self.ws, buf(lib.d_ocp_qp_ipm_ws_memsize(self.dim, self.arg)))
        for nm in ('A', 'B', 'b', 'Q', 'S', 'R', 'q', 'r', 'lbx', 'ubx', 'lbu', 'ubu', 'idxbx', 'idxbu', 'C', 'D', 'lg',
                   'ug', 'Zl', 'Zu', 'zl', 'zu', 'idxs', 'lls', 'lus'):
            f = getattr(lib, 'd_ocp_qp_set_' + nm)
            f.argtypes = [ip, vp, vp]
            f.restype = None
        lib.d_ocp_qp_ipm_solve.argtypes = [vp, vp, vp, vp]
        lib.d_ocp_qp_ipm_get_status.argtypes = [vp, vp]
        lib.d_ocp_qp_ipm_get_iter.argtypes = [vp, vp]
        self.keep = keep
        self.nx, self.nu, self.ng, self.ns = list(nx), list(nu), list(ng), list(ns)
        for k in range(N + 1):
            if nbx[k]:
                lib.d_ocp_qp_set_idxbx(k, iarr(idxbx[k]).ctypes.data, self.qp)
            if nbu[k]:
                lib.d_ocp_qp_set_idxbu(k, iarr(idxbu[k]).ctypes.data, self.qp)
            if nu[k]:
                Rk = np.ascontiguousarray(R[k], dtype=float)
                keep.append(Rk)
                lib.d_ocp_qp_set_R(k, Rk.ctypes.data, self.qp)
                Sk = np.zeros(nu[k] * nx[k])
                keep.append(Sk)
                lib.d_ocp_qp_set_S(k, Sk.ctypes.data, self.qp)
                rk = np.zeros(nu[k])
                keep.append(rk)
                lib.d_ocp_qp_set_r(k, rk.ctypes.data, self.qp)
            if ns[k]:
                lib.d_ocp_qp_set_idxs(k, iarr(idxs[k]).ctypes.data, self.qp)
                for nm, val in (('Zl', Zl[k]), ('Zu', Zu[k]), ('zl', zl[k]), ('zu', zu[k]), ('lls', np.zeros(ns[k])),
                                ('lus', np.zeros(ns[k]))):
                    a = np.ascontiguousarray(val, dtype=float)
                    keep.append(a)
                    getattr(lib, 'd_ocp_qp_set_' + nm)(k, a.ctypes.data, self.qp)
            if ng[k] and nu[k]:
                Dk = np.zeros(ng[k] * nu[k])
                keep.append(Dk)
                lib.d_ocp_qp_set_D(k, Dk.ctypes.data, self.qp)
        self.status = C.c_int(0)
        self.iters = C.c_int(0)

    def setup_all(self, nbx, nbu, idxbx, idxbu, idxs, Zl, zl, R):
        C, N, lib = self.C, self.N, self.lib
        nx, nu, ng, ns = self.nx, self.nu, self.ng, self.ns
        P = lambda arrs: (C.c_void_p * (N + 1))(*[a.ctypes.data if a is not None and a.size else None for a in arrs])
        z = lambda n: np.zeros(max(int(n), 1))
        self.bA = np.zeros((N + 1, 5, 5))
        self.bB = np.zeros((N + 1, 1, 5))
        self.bb = np.zeros((N + 1, 5))
        self.bQ = np.zeros((N + 1, 5, 5))
        self.bq = np.zeros((N + 1, 5))
        mg = max(ng)
        self.bC = np.zeros((N + 1, 5, mg))
        self.blg = np.zeros((N + 1, mg))
        self.bug = np.zeros((N + 1, mg))
        mbx = max(nbx)
        self.blbx = np.zeros((N + 1, mbx))
        self.bubx = np.zeros((N + 1, mbx))
        self.keep_all = []
        k_ = self.keep_all
        S_ = [z(nu[k] * nx[k]) for k in range(N + 1)]
        R_ = [np.ascontiguousarray(R[k], dtype=float).ravel() if nu[k] else z(1) for k in range(N + 1)]
        r_ = [z(nu[k]) for k in range(N + 1)]
        D_ = [z(ng[k] * nu[k]) for k in range(N + 1)]
        ibx = [np.array(idxbx[k], dtype=np.int32) for k in range(N + 1)]
        ibu = [np.array(idxbu[k], dtype=np.int32) for k in range(N + 1)]
        lbu = [np.full(max(nbu[k], 1), -STEER_RATE * 0.95) for k in range(N + 1)]
        ubu = [np.full(max(nbu[k], 1), STEER_RATE * 0.95) for k in range(N + 1)]
        Zl_ = [np.array(Zl[k] if ns[k] else [0.0], dtype=float) for k in range(N + 1)]
        zl_ = [np.array(zl[k] if ns[k] else [0.0], dtype=float) for k in range(N + 1)]
        isx = [np.array(idxs[k] if ns[k] else [0], dtype=np.int32) for k in range(N + 1)]
        isr = []
        for k in range(N + 1):
            nb = nbx[k] + nbu[k]
            a = -np.ones(nb + ng[k], dtype=np.int32)
            for j, c in enumerate(idxs[k]):
                a[c] = j
            isr.append(a if a.size else np.zeros(1, dtype=np.int32))
        ls_ = [z(ns[k]) for k in range(N + 1)]
        k_ += [S_, R_, r_, D_, ibx, ibu, lbu, ubu, Zl_, zl_, isx, isr, ls_]
        self.p_all = [
            P([self.bA[k] for k in range(N + 1)]), P([self.bB[k] for k in range(N + 1)]),
            P([self.bb[k] for k in range(N + 1)]), P([self.bQ[k] for k in range(N + 1)]), P(S_), P(R_),
            P([self.bq[k] for k in range(N + 1)]), P(r_), P(ibx), P([self.blbx[k] for k in range(N + 1)]),
            P([self.bubx[k] for k in range(N + 1)]), P(ibu), P(lbu), P(ubu),
            P([self.bC[k] for k in range(N + 1)]), P(D_), P([self.blg[k] for k in range(N + 1)]),
            P([self.bug[k] for k in range(N + 1)]), P(Zl_), P(Zl_), P(zl_), P(zl_), P(isx), P(isr),
            P(ls_), P(ls_),
        ]
        k_ += self.p_all
        lib.d_ocp_qp_set_all.argtypes = [C.c_void_p] * 27
        lib.d_ocp_qp_set_all.restype = None
        self.sx = np.zeros((N + 1, 5))
        self.su = np.zeros((N + 1, 1))
        msz = 64
        self.sbuf = [np.zeros((N + 1, msz)) for _ in range(9)]
        self.p_sol = [P([self.su[k] for k in range(N + 1)]), P([self.sx[k] for k in range(N + 1)])] +                      [P([b[k] for k in range(N + 1)]) for b in self.sbuf]
        k_ += [self.sbuf, self.p_sol]
        lib.d_ocp_qp_sol_get_all.argtypes = [C.c_void_p] * 12
        lib.d_ocp_qp_sol_get_all.restype = None

    def solve_all(self):
        lib, C = self.lib, self.C
        lib.d_ocp_qp_set_all(*self.p_all, self.qp)
        lib.d_ocp_qp_ipm_solve(self.qp, self.sol, self.arg, self.ws)
        lib.d_ocp_qp_ipm_get_status(self.ws, C.addressof(self.status))
        lib.d_ocp_qp_ipm_get_iter(self.ws, C.addressof(self.iters))
        lib.d_ocp_qp_sol_get_all(self.sol, *self.p_sol)
        return self.status.value, self.iters.value, self.sx.copy()


class SpatialLTVMPC:
    def __init__(self, track, estimator, p: dict):
        self.n = int(p['n'])
        self.ds = float(p['ds'])
        self.lpts = tuple(p['lpts'])
        self.lpts_a = np.array(self.lpts, dtype=np.float64)
        self.v_min_plan = float(p['v_min_plan'])
        self.c_pref = float(p['c_pref'])
        self.q_d = float(p['q_d'])
        self.q_mu = float(p['q_mu'])
        self.q_a = float(p['q_a'])
        self.r_w = float(p['r_w'])
        self.rho1 = float(p['rho1'])
        self.rho2 = float(p['rho2'])
        self.rhoa1 = float(p['rhoa1'])
        self.rhoa2 = float(p['rhoa2'])
        self.slack_near = float(p['slack_near'])
        self.slack_len = float(p['slack_len'])
        self.hp_iter = int(p['hp_iter'])
        self.hp_tol = tuple(p['hp_tol'])
        self.hp_mode = int(p['hp_mode'])
        self.ref_dp = bool(p['ref_dp'])

        self.hp = self._build_hpipm()

        # Numba 커널 워밍업
        _expm_fr_k(np.zeros((2, 7, 7)))
        st0 = np.arange(self.n + 1) * self.ds
        z1, zn = np.zeros(self.n + 1), np.zeros(self.n)
        hp = self.hp
        _mpc_hp_k(st0, np.zeros(5), np.zeros((self.n + 1, 5)), z1, z1 + 1.0, zn, zn + self.ds, z1, self.lpts_a,
                  track.wlo_c, track.whi_c, track.nf, track.fds, np.zeros((1, 4)), np.ones(1, dtype=np.int64), 0.1,
                  0.8, 0.1, 1.0, 1.0, 1.0, 1.0, estimator.tab_plan, hp.bA, hp.bB, hp.bb, hp.bQ, hp.bq, hp.bC, hp.blg,
                  hp.bug, hp.blbx, hp.bubx, np.zeros((self.n, len(self.lpts))), np.zeros((self.n, len(self.lpts))),
                  np.zeros((self.n, len(self.lpts))), np.zeros(self.n), 1.0)
        _hp_slack_k(np.zeros((self.n + 1, 5)), np.zeros((self.n, len(self.lpts))),
                    np.zeros((self.n, len(self.lpts))), np.zeros((self.n, len(self.lpts))))

        self.reset()

    def _build_hpipm(self):
        N, nl = self.n, len(self.lpts)
        nx = [5] * (N + 1)
        nu = [1] * N + [0]
        nbx = [5] + [1] * N
        nbu = [1] * N + [0]
        ng = [0] + [nl + 1] * N
        ns = [0] + [nl + 1] * N
        idxbx = [[0, 1, 2, 3, 4]] + [[4]] * N
        idxbu = [[0]] * N + [[]]
        idxs = [[]] + [[2 + j for j in range(nl + 1)]] * (N - 1) + [[1 + j for j in range(nl + 1)]]
        Zl, zl = [[]], [[]]
        for k in range(1, N + 1):
            fk = 1.0 + self.slack_near * math.exp(-k * self.ds / self.slack_len)
            Zl.append([2 * self.rho2 * fk] * nl + [2 * self.rhoa2])
            zl.append([self.rho1 * fk] * nl + [self.rhoa1])
        R = [np.array([[2 * self.r_w]])] * N + [np.zeros((0, 0))]
        hp = _Hpipm(N, nx, nu, nbx, nbu, ng, ns, idxbx, idxbu, idxs, Zl, Zl, zl, zl, R,
                    iter_max=self.hp_iter, tol=self.hp_tol, mode=self.hp_mode)
        hp.setup_all(nbx, nbu, idxbx, idxbu, idxs, Zl, zl, R)
        return hp

    def reset(self):
        self.sol = None
        self.sol_bad = False
        self.qp_info = None
        self.n_fail = 0
        self.n_resolve = 0

    def solve(self, st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, dpath,
              track, tab_plan, c_tyre, a_lat_hard):
        """통로 제약 내에서 공간 LTV-MPC를 풀어 (Xs, steering_rate, slack, dt) 또는 None을 반환."""
        N, nl = self.n, len(self.lpts)
        hp = self.hp
        bx = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        bsd = np.array(sides, dtype=np.int64).reshape(-1)
        base = dpath if self.ref_dp else np.zeros(N + 1)
        cmu = np.empty((N, nl))
        lgc = np.empty((N, nl))
        ugc = np.empty((N, nl))
        dt = np.empty(N)
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        _mpc_hp_k(f64(st), f64(X0), f64(Xbar), f64(kr), f64(vpred), f64(acc), f64(dpl), f64(base), self.lpts_a,
                  track.wlo_c, track.whi_c, track.nf, track.fds, bx, bsd, float(w_h), float(self.v_min_plan),
                  float(self.c_pref), float(self.q_d), float(self.q_mu), float(self.q_a), float(a_lat_hard),
                  tab_plan, hp.bA, hp.bB, hp.bb, hp.bQ, hp.bq, hp.bC, hp.blg, hp.bug, hp.blbx, hp.bubx,
                  cmu, lgc, ugc, dt, float(c_tyre))
        status, iters, Xs = hp.solve_all()
        self.qp_info = (iters, status)
        if status not in (0, 1) or not np.all(np.isfinite(Xs)):
            return None
        return Xs, np.diff(Xs[:, 4]) / dt, _hp_slack_k(Xs, cmu, lgc, ugc), dt
