"""LiDAR 위치 추정(Iterated EKF + SDF 정합)용 Numba JIT 커널 모음."""
import math
import numpy as np

from common.vehicle_params import (
    A_MAX, CSF_, CSR_, G_, HCG_, I_, KP_ACC, KP_BRK, LF_, LR_, LIDAR_X,
    M_, MU_, STEER_MAX, STEER_RATE, V_SWITCH, WB,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def _targets(Ht, Hs, Hv, nh, tt, tl, tb):
    ts0 = tt + 1e-6 - tl - tb
    tv0 = tt + 1e-6 - tl
    hs, st, hv, vc = False, 0.0, False, 0.0
    for j in range(nh - 1, -1, -1):
        if Ht[j] <= ts0:
            hs, st = True, Hs[j]
            break
    for j in range(nh - 1, -1, -1):
        if Ht[j] <= tv0:
            hv, vc = True, Hv[j]
            break
    return hs, st, hv, vc


@njit(cache=True)
def _p_accel(v_cmd, v):
    dv = v_cmd - v
    if v > 0.0:
        a = (KP_ACC if dv > 0 else KP_BRK) * dv
    else:
        a = (2.0 * 9.51 / 20.0 if dv > 0 else 2.0 * 9.51 / 5.0) * dv
    pos = A_MAX * V_SWITCH / v if v > V_SWITCH else A_MAX
    return max(-A_MAX, min(pos, a))


@njit(cache=True)
def _st_rhs(v, r, beta, psi, delta, a, c):
    aa = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
    cf = c * MU_ * CSF_ * (G_ * LR_ - aa * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + aa * HCG_)
    vl = v * WB
    a11 = -(cr + cf) / vl
    a12 = (cr * LR_ - cf * LF_) / (v * vl) - 1.0
    b1 = cf / vl
    a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
    a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB)
    b2 = M_ * LF_ * cf / (I_ * WB)
    return (v * math.cos(psi + beta), v * math.sin(psi + beta), r, a,
            a21 * beta + a22 * r + b2 * delta, a11 * beta + a12 * r + b1 * delta)


@njit(cache=True)
def _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, s, delta, c):
    x, y, psi, v, r, beta = s[0], s[1], s[2], s[3], s[4], s[5]
    h = 0.01
    for k in range(n):
        hs, tgt, hv, vc = _targets(Ht, Hs, Hv, nh, t0 + 0.01 * k, tl, tb)
        d0 = delta
        if hs:
            diff = tgt - delta
            if abs(diff) > 1e-4:
                delta += max(-STEER_RATE * h, min(STEER_RATE * h, diff))
            delta = max(-STEER_MAX, min(STEER_MAX, delta))
        dm = 0.5 * (d0 + delta)
        a = _p_accel(vc, v) if hv else 0.0
        if v >= 0.5:
            k1 = _st_rhs(v, r, beta, psi, dm, a, c)
            k2 = _st_rhs(v + 0.5 * h * k1[3], r + 0.5 * h * k1[4], beta + 0.5 * h * k1[5], psi + 0.5 * h * k1[2], dm, a, c)
            k3 = _st_rhs(v + 0.5 * h * k2[3], r + 0.5 * h * k2[4], beta + 0.5 * h * k2[5], psi + 0.5 * h * k2[2], dm, a, c)
            k4 = _st_rhs(v + h * k3[3], r + h * k3[4], beta + h * k3[5], psi + h * k3[2], dm, a, c)
            x += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            y += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            psi += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            v += h / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            r += h / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
            beta += h / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        else:
            x += v * math.cos(psi) * h
            y += v * math.sin(psi) * h
            r = v * math.tan(dm) / WB
            psi += r * h
            beta = 0.0
            v += a * h
    out = np.empty(6)
    out[0], out[1], out[2], out[3], out[4], out[5] = x, y, psi, v, r, beta
    return out, delta


@njit(cache=True)
def _predict_ekf(Ht, Hs, Hv, nh, tl, tb, t0, n, s, P, delta, c, Qd, Pmax):
    s1, d1 = _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, s, delta, c)
    F = np.eye(6)
    eps = np.array([1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5])
    for j in range(6):
        sp = s.copy()
        sg = 1.0
        if j == 3 and (s[3] <= 0.0 or (s[3] < 0.5 and s[3] + eps[3] >= 0.5)):
            sg = -1.0
        sp[j] += sg * eps[j]
        s2, _ = _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, sp, delta, c)
        for i in range(6):
            F[i, j] = (s2[i] - s1[i]) / (sg * eps[j])
    P1 = F @ P @ F.T
    for i in range(6):
        P1[i, i] += Qd[i] * n
    P1 = 0.5 * (P1 + P1.T)
    for i in range(6):
        if P1[i, i] > Pmax[i]:
            f = math.sqrt(Pmax[i] / P1[i, i])
            for j in range(6):
                P1[i, j] *= f
                P1[j, i] *= f
    return s1, P1, d1


@njit(cache=True)
def _sdf_at(sdf, ox, oy, res, W, H, px, py):
    u = (px - ox) / res - 0.5
    v = (py - oy) / res - 0.5
    fu = math.floor(u)
    fv = math.floor(v)
    c0 = int(fu)
    r0 = int(fv)
    if c0 < 0 or r0 < 0 or c0 + 1 >= W or r0 + 1 >= H:
        return 0.0, 0.0, 0.0, False
    fx = u - fu
    fy = v - fv
    v00 = sdf[r0, c0]
    v01 = sdf[r0, c0 + 1]
    v10 = sdf[r0 + 1, c0]
    v11 = sdf[r0 + 1, c0 + 1]
    val = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11)
    gx = ((1.0 - fy) * (v01 - v00) + fy * (v11 - v10)) / res
    gy = ((1.0 - fx) * (v10 - v00) + fx * (v11 - v01)) / res
    return val, gx, gy, True


@njit(cache=True)
def _register(s_pred, P, ranges, cb, sb, stride, r_min, r_max, sdf, ox, oy, res, W, H, sig_b, sig_ang, off, off_b,
              hub, gate, gate2, inl_thr, info_scale, n_iter, use_prior, link, obs_ext):
    Pinv = np.linalg.inv(P)
    s = s_pred.copy()
    M = Pinv.copy()
    nval = 0
    ninl = 0
    nfree = 0
    ss = 0.0
    A = np.zeros((3, 3))
    b = np.zeros(3)
    p0 = s_pred[:3].copy()
    it_done = 0
    for it in range(n_iter):
        A[:, :] = 0.0
        b = np.zeros(3)
        p0 = s[:3].copy()
        nval = 0
        ninl = 0
        nfree = 0
        ss = 0.0
        gt_ = gate if it == 0 else gate2
        cy = math.cos(s[2])
        sy = math.sin(s[2])
        lx = s[0] + LIDAR_X * cy
        ly = s[1] + LIDAR_X * sy
        for i in range(0, ranges.shape[0], stride):
            rg = ranges[i]
            if not (rg > r_min and rg < r_max):
                continue
            bx = rg * cb[i]
            by = rg * sb[i]
            px = lx + cy * bx - sy * by
            py = ly + sy * bx + cy * by
            val, gx, gy, ok = _sdf_at(sdf, ox, oy, res, W, H, px, py)
            if not ok:
                continue
            nval += 1
            gn = math.sqrt(gx * gx + gy * gy)
            ci = abs(gx * (px - lx) + gy * (py - ly)) / (gn * rg) if gn > 1e-9 else 1.0
            e = val - (off + off_b * ci)
            ae = abs(e)
            if ae < inl_thr:
                ninl += 1
                ss += e * e
            elif e > 0.0:
                nfree += 1
            if ae > gt_:
                continue
            sg = math.sqrt(sig_b * sig_b + (sig_ang * rg) ** 2)
            w = 1.0 if ae <= hub * sg else hub * sg / ae
            w /= sg * sg * info_scale
            jt = -gx * (py - s[1]) + gy * (px - s[0])
            A[0, 0] += w * gx * gx
            A[0, 1] += w * gx * gy
            A[0, 2] += w * gx * jt
            A[1, 1] += w * gy * gy
            A[1, 2] += w * gy * jt
            A[2, 2] += w * jt * jt
            b[0] += w * gx * e
            b[1] += w * gy * e
            b[2] += w * jt * e
        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        M = Pinv.copy()
        for i in range(3):
            for j in range(3):
                M[i, j] += A[i, j]
        dsv = s - s_pred
        dsv[2] = _wrap(dsv[2])
        g = Pinv @ dsv if use_prior else np.zeros(6)
        for i in range(3):
            g[i] += b[i]
        dx = -np.linalg.solve(M, g)
        for i in range(6):
            s[i] += dx[i]
        it_done = it + 1
        if abs(dx[0]) < 1e-5 and abs(dx[1]) < 1e-5 and abs(dx[2]) < 1e-6:
            break
    Pp = np.linalg.inv(M)
    Pp = 0.5 * (Pp + Pp.T)
    s[2] = _wrap(s[2])
    rms = math.sqrt(ss / ninl) if ninl > 0 else 1.0
    jcost = 0.0
    cy = math.cos(s[2])
    sy = math.sin(s[2])
    lx = s[0] + LIDAR_X * cy
    ly = s[1] + LIDAR_X * sy
    nval = 0
    ninl = 0
    nfree = 0
    nsmall = 0
    run_n = 0
    x0r = 0.0
    y0r = 0.0
    xl = 0.0
    yl = 0.0
    for i in range(0, ranges.shape[0], stride):
        rg = ranges[i]
        if not (rg > r_min and rg < r_max):
            continue
        bx = rg * cb[i]
        by = rg * sb[i]
        px = lx + cy * bx - sy * by
        py = ly + sy * bx + cy * by
        val, gx, gy, ok = _sdf_at(sdf, ox, oy, res, W, H, px, py)
        if not ok:
            continue
        nval += 1
        gn = math.sqrt(gx * gx + gy * gy)
        ci = abs(gx * (px - lx) + gy * (py - ly)) / (gn * rg) if gn > 1e-9 else 1.0
        e = val - (off + off_b * ci)
        sg = math.sqrt(sig_b * sig_b + (sig_ang * rg) ** 2)
        ae = min(abs(e), gate2)
        if ae <= hub * sg:
            jcost += (ae / sg) ** 2 / info_scale
        else:
            jcost += (2.0 * hub * ae / sg - hub * hub) / info_scale
        free = False
        if abs(e) < inl_thr:
            ninl += 1
        elif e > 0.0:
            nfree += 1
            free = True
        if free and run_n > 0 and math.hypot(px - xl, py - yl) <= link:
            run_n += 1
        else:
            if run_n > 0 and math.hypot(xl - x0r, yl - y0r) <= obs_ext:
                nsmall += run_n
            run_n = 0
            if free:
                run_n = 1
                x0r = px
                y0r = py
        if free:
            xl = px
            yl = py
    if run_n > 0 and math.hypot(xl - x0r, yl - y0r) <= obs_ext:
        nsmall += run_n
    if use_prior:
        dsv = s - s_pred
        dsv[2] = _wrap(dsv[2])
        jcost += dsv @ (Pinv @ dsv)
        sg1, ld1 = np.linalg.slogdet(M)
        sg2, ld2 = np.linalg.slogdet(P)
        jcost += ld1 + ld2
    return s, Pp, nval, ninl, nfree, nsmall, rms, A, it_done, p0, b, jcost


@njit(cache=True)
def _mm_predict(Ht, Hs, Hv, nh, lats, tb, t0, n, S, PP, D, c, Qd, Pmax):
    K = lats.shape[0]
    S1 = np.empty_like(S)
    P1 = np.empty_like(PP)
    D1 = np.empty_like(D)
    for k in range(K):
        s1, p1, d1 = _predict_ekf(Ht, Hs, Hv, nh, lats[k], tb, t0, n, S[k], PP[k], D[k], c, Qd, Pmax)
        S1[k] = s1
        P1[k] = p1
        D1[k] = d1
    return S1, P1, D1


@njit(cache=True)
def _mm_update(S, PP, p0, A, b):
    K = S.shape[0]
    S1 = np.empty_like(S)
    P1 = np.empty_like(PP)
    J = np.empty(K)
    for k in range(K):
        Pi = np.linalg.inv(PP[k])
        M = Pi.copy()
        for i in range(3):
            for j in range(3):
                M[i, j] += A[i, j]
        u = np.empty(3)
        u[0] = S[k, 0] - p0[0]
        u[1] = S[k, 1] - p0[1]
        u[2] = _wrap(S[k, 2] - p0[2])
        gq = b + A @ u
        g = np.zeros(6)
        g[0], g[1], g[2] = gq[0], gq[1], gq[2]
        Minv = np.linalg.inv(M)
        dx = -(Minv @ g)
        S1[k] = S[k] + dx
        S1[k, 2] = _wrap(S1[k, 2])
        P1[k] = 0.5 * (Minv + Minv.T)
        f0 = 2.0 * (b @ u) + u @ (A @ u)
        sgn1, ld1 = np.linalg.slogdet(M)
        sgn2, ld2 = np.linalg.slogdet(PP[k])
        J[k] = f0 - g @ (Minv @ g) + ld1 + ld2
    return S1, P1, J


@njit(cache=True)
def _corr_scores(lf, ox, oy, res, W, H, qx, qy, cx, cy, out):
    nk = cx.shape[0]
    nj = qx.shape[0]
    nb = qx.shape[1]
    for k in range(nk):
        x0 = (cx[k] - ox) / res
        y0 = (cy[k] - oy) / res
        for j in range(nj):
            acc = 0.0
            for i in range(nb):
                c = int(x0 + qx[j, i])
                r = int(y0 + qy[j, i])
                if 0 <= c < W and 0 <= r < H:
                    acc += lf[r, c]
            out[k, j] = acc
