"""Numba JIT 고속 연산 커널 모음 (Perception, Planning, Control 공용).

기존 cmpc_berlin_tt.py 내부의 @njit 커널을 분리하여 모듈 간 재사용성과
install_final_demo.sh 단계에서의 사전 컴파일(warmup)을 지원한다.
"""


import math


import numpy as np


from common.vehicle_params import (
    A_MAX, ANG_INC, ANG_MIN, BOX_FRONT, CSF_, CSR_, G_, HCG_, I_, K_US0,
    KP_ACC, KP_BRK, LF_, LR_, M_, MU_, STEER_MAX, STEER_RATE, V_SWITCH, WB,
)


try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _dp_core(cost0, node_cost, blocked, kv, cst, cv_cost, sl_cost, dlo, dhi, w_geo):
    M1, J = node_cost.shape
    K = kv.shape[0]
    M = M1 - 1
    back = np.zeros((M, J, K), np.int8)
    cost = cost0.copy()
    new = np.empty((J, K))
    reach = M
    cmax = (cv_cost.shape[0] - 1) // 2
    for i in range(1, M + 1):
        anyf = False
        c = cst[i]
        for j in range(J):
            if blocked[i, j]:
                for k in range(K):
                    new[j, k] = np.inf
                continue
            nc = node_cost[i, j]
            for k in range(K):
                jp = j - kv[k]
                best = np.inf
                bd = 0
                if jp >= 0 and jp < J:
                    glo = dlo[i - 1, jp]
                    ghi = dhi[i - 1, jp]
                    for dl in range(-c, c + 1):
                        kp = k - dl
                        if kp >= 0 and kp < K:
                            val = cost[jp, kp] + cv_cost[dl + cmax]
                            ex = glo - dl if glo - dl > dl - ghi else dl - ghi
                            if ex > 0.0:
                                val += w_geo * ex * ex
                            if val < best:
                                best = val
                                bd = dl + cmax
                if best < np.inf:
                    new[j, k] = best + nc + sl_cost[k]
                    back[i - 1, j, k] = bd
                    anyf = True
                else:
                    new[j, k] = np.inf
        if not anyf:
            reach = i - 1
            break
        tmp = cost
        cost = new
        new = tmp
    return back, cost, reach


@njit(cache=True)
def _cluster_k(xs, ys, link):
    n = xs.shape[0]
    par = np.arange(n)
    l2 = link * link
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy < l2:
                a = i
                while par[a] != a:
                    a = par[a]
                b = j
                while par[b] != b:
                    b = par[b]
                if a != b:
                    par[b] = a
    lab = np.empty(n, np.int64)
    for i in range(n):
        a = i
        while par[a] != a:
            a = par[a]
        lab[i] = a
    return lab


@njit(cache=True)
def _hits_k(scan, lx, ly, cy, sy, ca, sa, r_max, ox, oy, res, W, H, edt, obs_edt, mres, MW):
    n = scan.shape[0]
    out = np.empty(n, np.int64)
    m = 0
    thr = np.float32(obs_edt)
    for i in range(n):
        r = scan[i]
        if not (r > 0.05 and r < r_max):
            continue
        ux = cy * ca[i] - sy * sa[i]
        uy = sy * ca[i] + cy * sa[i]
        px = lx + r * ux
        py = ly + r * uy
        col = int(math.floor((px - ox) / res))
        row = int(math.floor((py - oy) / res))
        if col < 0 or col >= W or row < 0 or row >= H:
            continue
        if not (edt[row, col] > thr):
            continue
        mc = int(math.floor((px - ox) / mres))
        mr = int(math.floor((py - oy) / mres))
        out[m] = mr * MW + mc
        m += 1
    srt = np.sort(out[:m])
    k = 0
    for i in range(m):
        if i == 0 or srt[i] != srt[i - 1]:
            srt[k] = srt[i]
            k += 1
    return srt[:k].copy()


@njit(cache=True)
def _merge_k(active, amem, ax, ay, a_s, ad, aok, cells, cell_x, cell_y, cell_s, cell_d, cell_ok):
    na, nc = active.shape[0], cells.shape[0]
    n = na + nc
    o_id = np.empty(n, np.int64)
    o_mem = np.empty(n, np.int16)
    o_x = np.empty(n, np.float32)
    o_y = np.empty(n, np.float32)
    o_s = np.empty(n, np.float32)
    o_d = np.empty(n, np.float32)
    o_ok = np.empty(n, np.bool_)
    i = j = k = 0
    while i < na or j < nc:
        if j >= nc or (i < na and active[i] < cells[j]):
            o_id[k], o_mem[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = active[i], amem[i], ax[i], ay[i], a_s[i], ad[i], aok[i]
            i += 1
        elif i < na and active[i] == cells[j]:
            o_id[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = active[i], ax[i], ay[i], a_s[i], ad[i], aok[i]
            o_mem[k] = min(amem[i] + 2, 12)
            i += 1
            j += 1
        else:
            c = cells[j]
            o_id[k], o_mem[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = c, 2, cell_x[c], cell_y[c], cell_s[c], cell_d[c], cell_ok[c]
            j += 1
        k += 1
    return o_id[:k].copy(), o_mem[:k].copy(), o_x[:k].copy(), o_y[:k].copy(), o_s[:k].copy(), o_d[:k].copy(), o_ok[:k].copy()


@njit(cache=True)
def _carve_k(amem, rng, b, scan):
    n = amem.shape[0]
    keep = np.ones(n, np.bool_)
    lo, hi, mg = np.float32(0.4), np.float32(7.0), np.float32(0.2)
    for i in range(n):
        bi = b[i]
        if bi < 2 or bi > 1077 or not (rng[i] > lo) or not (rng[i] < hi):
            continue
        mn = min(min(min(min(scan[bi - 2], scan[bi - 1]), scan[bi]), scan[bi + 1]), scan[bi + 2])
        if mn != mn or scan[bi - 2] != scan[bi - 2] or scan[bi - 1] != scan[bi - 1] or scan[bi] != scan[bi]                 or scan[bi + 1] != scan[bi + 1] or scan[bi + 2] != scan[bi + 2]:
            continue
        if mn > np.float64(np.float32(rng[i] + mg)):
            amem[i] -= 1
            if amem[i] <= 0:
                keep[i] = False
    return keep


@njit(cache=True)
def _frenet_k(fx, fy, x, y, guess):
    nf = fx.shape[0]
    best = np.inf
    bi = 0
    if guess < 0:
        for i in range(nf):
            dd = (fx[i] - x) ** 2 + (fy[i] - y) ** 2
            if dd < best:
                best = dd
                bi = i
    else:
        for c in range(guess - 12, guess + 40):
            i = c % nf
            dd = (fx[i] - x) ** 2 + (fy[i] - y) ** 2
            if dd < best:
                best = dd
                bi = i
    return bi


@njit(cache=True)
def _psum(a):
    n = a.shape[0]
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    i = 8
    m = n - (n % 8)
    while i < m:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@njit(cache=True)
def _dp_boxes_k(sd, dg, s0, bx, bprev, bpf, bhard, half, w_h, dp_near, p_near, c_pref, w_soft, Ds, gap_rule,
                min_gap_w, min_gap_keep, wlo, whi, cost_node, blocked):
    M1 = sd.shape[0]
    J = dg.shape[0]
    ks = np.empty(M1, np.int64)
    for b in range(bx.shape[0]):
        nks = 0
        for k in range(M1):
            if sd[k] >= bx[b, 0] - half and sd[k] <= bx[b, 1] + half:
                ks[nks] = k
                nks += 1
        if nks == 0:
            continue
        lo = bx[b, 2] - w_h
        hi = bx[b, 3] + w_h
        nfar = 0
        mwhi = np.inf
        mwlo = -np.inf
        for t in range(nks):
            k = ks[t]
            if sd[k] - s0 >= dp_near:
                nfar += 1
                mwhi = min(mwhi, whi[k])
                mwlo = max(mwlo, wlo[k])
                for j in range(J):
                    if dg[j] > lo and dg[j] < hi:
                        blocked[k, j] = True
        if nfar > 0 and gap_rule:
            g_hi = min_gap_keep if bprev[b] == 1 else min_gap_w
            g_lo = min_gap_keep if bprev[b] == -1 else min_gap_w
            if mwhi - hi < g_hi:
                for t in range(nks):
                    k = ks[t]
                    if sd[k] - s0 >= dp_near:
                        for j in range(J):
                            if dg[j] >= hi:
                                blocked[k, j] = True
            if lo - mwlo < g_lo:
                for t in range(nks):
                    k = ks[t]
                    if sd[k] - s0 >= dp_near:
                        for j in range(J):
                            if dg[j] <= lo:
                                blocked[k, j] = True
        for t in range(nks):
            k = ks[t]
            if sd[k] - s0 < dp_near:
                for j in range(J):
                    if dg[j] > lo and dg[j] < hi:
                        cost_node[k, j] += p_near
        for t in range(nks):
            k = ks[t]
            for j in range(J):
                clr = max(lo - dg[j], dg[j] - hi)
                soft = max(0.0, c_pref - clr)
                cost_node[k, j] += w_soft * (soft * soft) * Ds
        if bprev[b] != 0:
            dc = 0.5 * (bx[b, 2] + bx[b, 3])
            kmid = ks[nks // 2]
            for j in range(J):
                wrong = (dg[j] < dc) if bprev[b] > 0 else (dg[j] > dc)
                if not wrong:
                    continue
                if bhard[b]:
                    for t in range(nks):
                        blocked[ks[t], j] = True
                else:
                    cost_node[kmid, j] += bpf[b]


@njit(cache=True)
def _dp_sides_k(sd, dpath_dp, reach, bx, half):
    nb = bx.shape[0]
    out = np.zeros(nb, np.int64)
    buf = np.empty(sd.shape[0])
    for b in range(nb):
        n = 0
        for k in range(min(reach + 1, sd.shape[0])):
            if sd[k] >= bx[b, 0] - half and sd[k] <= bx[b, 1] + half:
                buf[n] = dpath_dp[k]
                n += 1
        if n == 0:
            continue
        out[b] = 1 if _psum(buf[:n]) / n > 0.5 * (bx[b, 2] + bx[b, 3]) else -1
    return out


@njit(cache=True)
def _clusters_k(labels, nc, rel, dd):
    n = labels.shape[0]
    rmin = np.full(nc, np.inf, np.float32)
    rmax = np.full(nc, -np.inf, np.float32)
    dmin = np.full(nc, np.inf, np.float32)
    dmax = np.full(nc, -np.inf, np.float32)
    for i in range(n):
        c = labels[i]
        rmin[c] = min(rmin[c], rel[i])
        rmax[c] = max(rmax[c], rel[i])
        dmin[c] = min(dmin[c], dd[i])
        dmax[c] = max(dmax[c], dd[i])
    return rmin, rmax, dmin, dmax


@njit(cache=True)
def _interp1(x, xp, fp):
    n = xp.shape[0]
    if x < xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xp[lo] == x:
        return fp[lo]
    slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo])
    return slope * (x - xp[lo]) + fp[lo]


@njit(cache=True)
def _dp_k(s0, d0, chi0, v, N, ds, Ds, dd, Kmax, wlo_c, whi_c, wlo_p, whi_p, fds, dp_dmax, w_dev, w_wall, bx, bprev,
          bpf, bhard, w_h, dp_near, p_near, c_pref, w_soft, gap_rule, min_gap_w, dp_adec, dp_vmax, dp_vfloor, a_lat_hard,
          dp_kcap, w_cv, w_sl, c_ty, min_gap_keep, fk, L, dp_kgeo, w_geo):
    st = np.empty(N + 1)
    for i in range(N + 1):
        st[i] = s0 + ds * i
    M = int(round(N * ds / Ds))
    sd = np.empty(M + 1)
    for i in range(M + 1):
        sd[i] = s0 + Ds * i
    sq = dd / Ds
    K = 2 * Kmax + 1
    kv = np.arange(-Kmax, Kmax + 1)
    nf = wlo_c.shape[0]
    wlo = np.empty(M + 1)
    whi = np.empty(M + 1)
    wlo2 = np.empty(M + 1)
    whi2 = np.empty(M + 1)
    for i in range(M + 1):
        k = int(np.rint(sd[i] / fds)) % nf
        wlo[i], whi[i], wlo2[i], whi2[i] = wlo_c[k], whi_c[k], wlo_p[k], whi_p[k]
    jlo = int(math.floor((min(max(wlo.min(), -dp_dmax), d0) + 3.5) / dd)) - 1
    jhi = int(math.ceil((max(min(whi.max(), dp_dmax), d0) + 3.5) / dd)) + 1
    j_a, j_b = max(jlo, 0), min(jhi, int(7.0 / dd))
    J = j_b - j_a + 1
    dg = np.empty(J)
    for j in range(J):
        dg[j] = -3.5 + dd * (j_a + j)
    blocked = np.empty((M + 1, J), np.bool_)
    cost_node = np.empty((M + 1, J))
    for i in range(M + 1):
        for j in range(J):
            blocked[i, j] = dg[j] < wlo[i] or dg[j] > whi[i]
            pen = max(0.0, wlo2[i] - dg[j]) + max(0.0, dg[j] - whi2[i])
            cost_node[i, j] = w_dev * (dg[j] * dg[j]) * Ds + w_wall * (pen * pen) * Ds
    half = 0.5 * Ds
    _dp_boxes_k(sd, dg, s0, bx, bprev, bpf, bhard, half, w_h, dp_near, p_near, c_pref, w_soft, Ds, gap_rule,
                min_gap_w, min_gap_keep, wlo, whi, cost_node, blocked)
    cst = np.empty(M + 1, np.int64)
    c = 0
    for i in range(M + 1):
        vr = min(math.sqrt(max(v * v - 2.0 * dp_adec * (sd[i] - s0), 0.0)), 100.0)
        vdp = max(min(vr, dp_vmax), dp_vfloor)
        kap = min(min(STEER_MAX / (WB + K_US0 / c_ty * vdp * vdp), a_lat_hard / (vdp * vdp)), dp_kcap)
        cst[i] = min(max(np.int64(math.floor(kap * Ds / sq + 1e-9)), 1), Kmax)
        c = max(c, cst[i])
    cv_cost = np.empty(2 * c + 1)
    for i in range(2 * c + 1):
        g = (i - c) * sq / Ds
        cv_cost[i] = w_cv * (g * g) * Ds
    sl_cost = np.empty(K)
    for k in range(K):
        g = kv[k] * sq
        sl_cost[k] = w_sl * (g * g) * Ds
    dlo = np.full((M + 1, J), -100.0)
    dhi = np.full((M + 1, J), 100.0)
    if dp_kgeo > 0.0:
        kg = dp_kgeo * STEER_MAX / (WB + K_US0 / c_ty * dp_vfloor ** 2)
        nfk = fk.shape[0]
        for i in range(M + 1):
            u = (sd[i] % L) / fds
            fu = math.floor(u)
            i0 = int(fu) % nfk
            f = u - fu
            kr_d = fk[i0] * (1 - f) + fk[(i0 + 1) % nfk] * f
            for j in range(J):
                Dm = max(1.0 - kr_d * dg[j], 0.05)
                dlo[i, j] = (-kg * Dm - kr_d) * Dm * (Ds / sq)
                dhi[i, j] = (kg * Dm - kr_d) * Dm * (Ds / sq)
    cost0 = np.full((J, K), np.inf)
    j0 = min(max(int(round((d0 - dg[0]) / dd)), 0), J - 1)
    k0 = min(max(int(round(math.tan(max(-1.2, min(1.2, chi0))) / sq)), -Kmax), Kmax) + Kmax
    for k in range(max(k0 - 1, 0), min(k0 + 2, K)):
        cost0[j0, k] = 0.0
    back, cost, reach = _dp_core(cost0, cost_node, blocked, kv, cst, cv_cost, sl_cost, dlo, dhi, w_geo)
    if reach < M:
        back, cost, reach2 = _dp_core(cost0, cost_node[:reach + 1], blocked[:reach + 1], kv, cst[:reach + 1], cv_cost,
                                      sl_cost, dlo, dhi, w_geo)
    flat = 0
    for q in range(J * K):
        if cost.flat[q] < cost.flat[flat]:
            flat = q
    jb, kb = flat // K, flat % K
    path_j = np.zeros(M + 1, np.int64)
    path_j[reach] = jb
    for i in range(reach, 0, -1):
        dl = np.int64(back[i - 1, jb, kb]) - c
        jp = jb - kv[kb]
        kb = kb - dl
        jb = jp
        path_j[i - 1] = jb
    for i in range(reach + 1, M + 1):
        path_j[i] = path_j[reach]
    dpath_dp = np.empty(M + 1)
    for i in range(M + 1):
        dpath_dp[i] = dg[path_j[i]]
    dpath = np.empty(N + 1)
    for i in range(N + 1):
        dpath[i] = _interp1(st[i], sd, dpath_dp)
    sides = _dp_sides_k(sd, dpath_dp, reach, bx, half)
    s_block = sd[reach + 1] if reach < M else np.nan
    return st, dpath, sides, s_block


@njit(cache=True)
def _boxes_k(active, act_ok, act_s, act_x, act_y, act_d, MW, W, s_car, L, win_hi, link):
    n = active.shape[0]
    pos = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if act_ok[i]:
            pos[m] = i
            m += 1
    pos = pos[:m]
    if m > 400:
        blk = np.empty(m, np.int64)
        for i in range(m):
            a = active[pos[i]]
            blk[i] = (a // MW // 2) * W + (a % MW) // 2
        order = np.argsort(blk, kind='mergesort')
        keep = np.zeros(m, np.bool_)
        for t in range(m):
            if t == 0 or blk[order[t]] != blk[order[t - 1]]:
                keep[order[t]] = True
        pos = pos[keep]
        m = pos.shape[0]
    sc, h, Lf, lo, hi = np.float32(s_car), np.float32(L / 2), np.float32(L), np.float32(-1.5), np.float32(win_hi)
    rel = np.empty(m, np.float32)
    k = 0
    for i in range(m):
        v = np.float32(np.float32(act_s[pos[i]] - sc) + h)
        r = np.float32(np.float32(v % Lf) - h)
        if r > lo and r < hi:
            pos[k] = pos[i]
            rel[k] = r
            k += 1
    pos = pos[:k]
    rel = rel[:k]
    if k == 0:
        return _clusters_k(np.zeros(0, np.int64), 0, rel, rel)
    xs = np.empty(k)
    ys = np.empty(k)
    dd = np.empty(k, np.float32)
    for i in range(k):
        j = pos[i]
        xs[i] = act_x[j]
        ys[i] = act_y[j]
        dd[i] = act_d[j]
    roots = _cluster_k(xs, ys, link) if k > 1 else np.zeros(1, np.int64)
    rs = np.unique(roots)
    nc = rs.shape[0]
    labels = np.empty(k, np.int64)
    for i in range(k):
        labels[i] = np.searchsorted(rs, roots[i])
    return _clusters_k(labels, nc, rel, dd)



@njit(cache=True)
def _lat_sim_k(Ti, Vc, Lc, vL, t0, n):
    v = vL.copy()
    ok = True
    for k in range(n):
        tk = t0 + 0.01 * k + 1e-6
        for j in range(Lc.shape[0]):
            i = Ti.shape[0] - 1
            while i >= 0 and Ti[i] > tk - Lc[j]:
                i -= 1
            if i < 0:
                ok = False
                i = 0
            vj = v[j]
            dv = Vc[i] - vj
            if vj > 0.0:
                kp = KP_ACC if dv > 0 else KP_BRK
            else:
                kp = 0.951 if dv > 0 else 3.804
            a = kp * dv
            pos = A_MAX * V_SWITCH / vj if vj > V_SWITCH else A_MAX
            if a > pos:
                a = pos
            if a < -A_MAX:
                a = -A_MAX
            v[j] = vj + 0.01 * a
    return v, ok


@njit(cache=True)
def _expm_fr_k(M):
    nb = M.shape[0]
    out = np.zeros((nb, 7, 7))
    A = np.zeros((7, 7))
    T = np.zeros((7, 7))
    T2 = np.zeros((7, 7))
    E = np.zeros((7, 7))
    for b in range(nb):
        nrm = 0.0
        for i in range(5):
            ssum = 0.0
            for j in range(7):
                ssum += abs(M[b, i, j])
            if ssum > nrm:
                nrm = ssum
        s = 0
        if nrm > 0.5:
            s = int(math.ceil(math.log(nrm / 0.5) / math.log(2.0)))
        sc = 1.0 / (2.0 ** s)
        for i in range(7):
            for j in range(7):
                A[i, j] = M[b, i, j] * sc
                T[i, j] = A[i, j]
                E[i, j] = A[i, j] + (1.0 if i == j else 0.0)
        a01, a02, a06 = A[0, 1], A[0, 2], A[0, 6]
        a10, a11, a12, a13, a16 = A[1, 0], A[1, 1], A[1, 2], A[1, 3], A[1, 6]
        a22, a23, a24, a32, a33, a34, a45 = A[2, 2], A[2, 3], A[2, 4], A[3, 2], A[3, 3], A[3, 4], A[4, 5]
        for p in range(2, 11):
            for i in range(5):
                t0, t1, t2, t3, t4 = T[i, 0], T[i, 1], T[i, 2], T[i, 3], T[i, 4]
                T2[i, 0] = (t1 * a10) / p
                T2[i, 1] = (t0 * a01 + t1 * a11) / p
                T2[i, 2] = (t0 * a02 + t1 * a12 + t2 * a22 + t3 * a32) / p
                T2[i, 3] = (t1 * a13 + t2 * a23 + t3 * a33) / p
                T2[i, 4] = (t2 * a24 + t3 * a34) / p
                T2[i, 5] = (t4 * a45) / p
                T2[i, 6] = (t0 * a06 + t1 * a16) / p
            for i in range(5):
                for j in range(7):
                    T[i, j] = T2[i, j]
                    E[i, j] += T2[i, j]
        for _ in range(s):
            for i in range(5):
                for j in range(7):
                    acc = 0.0
                    for q in range(5):
                        acc += E[i, q] * E[q, j]
                    if j == 5:
                        acc += E[i, 5]
                    elif j == 6:
                        acc += E[i, 6]
                    T2[i, j] = acc
            for i in range(5):
                for j in range(7):
                    E[i, j] = T2[i, j]
        for i in range(7):
            for j in range(7):
                out[b, i, j] = E[i, j]
    return out


@njit(cache=True)
def _corr1(s, wlo, whi, nf, fds, bx, bsd, w_h):
    i = int(np.rint(s / fds)) % nf
    lo = wlo[i]
    hi = whi[i]
    for b in range(bx.shape[0]):
        if bsd[b] == 0 or s < bx[b, 0] or s > bx[b, 1]:
            continue
        if bsd[b] > 0:
            v = bx[b, 3] + w_h
            if v > lo:
                lo = v
        else:
            v = bx[b, 2] - w_h
            if v < hi:
                hi = v
    return lo, hi


@njit(cache=True)
def _mpc_hp_k(st, X0, Xbar, kr, vpred, acc, dpl, dpath, lpts, wlo, whi, nf, fds, bx, bsd, w_h, v_min_plan, c_pref,
              q_d, q_mu, q_a, a_lat_hard, tab_plan, bA, bB, bb, bQ, bq, bC, blg, bug, blbx, bubx, cmu, lgc, ugc, dt,
              c_ty):
    N = dpl.shape[0]
    nl = lpts.shape[0]
    vp = np.empty(N + 1)
    for k in range(N + 1):
        vp[k] = vpred[k] if vpred[k] > v_min_plan else v_min_plan
    M = np.zeros((N, 7, 7))
    for k in range(N):
        vi = vp[k]
        a = acc[k]
        a = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
        cf = c_ty * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
        cr = c_ty * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
        a11 = -(cr + cf) / (vi * WB)
        a12 = (cr * LR_ - cf * LF_) / (vi * vi * WB) - 1.0
        b1 = cf / (vi * WB)
        a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
        a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (vi * I_ * WB)
        b2 = M_ * LF_ * cf / (I_ * WB)
        dbar = Xbar[k, 0]
        chib = Xbar[k, 1] + Xbar[k, 2]
        cb = math.cos(chib)
        sb = math.sin(chib)
        Dk = 1.0 - kr[k] * dbar
        if Dk < 0.2:
            Dk = 0.2
        krr = kr[k]
        cd = -krr * krr * vi * cb / Dk ** 2
        cm = krr * vi * sb / Dk
        dt[k] = dpl[k] / vi
        h = dt[k]
        M[k, 0, 1] = vi * cb * h
        M[k, 0, 2] = vi * cb * h
        M[k, 0, 6] = vi * (sb - cb * chib) * h
        M[k, 1, 0] = cd * h
        M[k, 1, 1] = cm * h
        M[k, 1, 2] = cm * h
        M[k, 1, 3] = h
        M[k, 1, 6] = (-krr * vi * cb / Dk - cd * dbar - cm * chib) * h
        M[k, 2, 2] = a11 * h
        M[k, 2, 3] = a12 * h
        M[k, 2, 4] = b1 * h
        M[k, 3, 2] = a21 * h
        M[k, 3, 3] = a22 * h
        M[k, 3, 4] = b2 * h
        M[k, 4, 5] = h
    E = _expm_fr_k(M)
    for k in range(N):
        for i in range(5):
            for j in range(5):
                bA[k, i, j] = E[k, j, i]
            bB[k, 0, i] = E[k, i, 5]
            bb[k, i] = E[k, i, 6]
    v0_ = vp[0]
    x = v0_ if v0_ > 0.0 else 0.0
    ab = tab_plan[min(int(x * 20.0), tab_plan.shape[0] - 1)]
    cum = 0.0
    for k in range(N + 1):
        if k > 0:
            cum += dpl[k - 1]
        la, ha = _corr1(st[k], wlo, whi, nf, fds, bx, bsd, w_h)
        lb, hb = _corr1(st[k] + BOX_FRONT, wlo, whi, nf, fds, bx, bsd, w_h)
        loc = la if la > lb else lb
        hic = ha if ha < hb else hb
        dr = dpath[k]
        if dr < loc + c_pref:
            dr = loc + c_pref
        if dr > hic - c_pref:
            dr = hic - c_pref
        if hic - loc < 2 * c_pref:
            dr = 0.5 * (loc + hic)
        vsq = v0_ * v0_ - 2.0 * ab * cum
        vreach = math.sqrt(vsq if vsq > 1.0 else 1.0)
        vlat = vp[k] if vp[k] < vreach else vreach
        wa = q_a * (vp[k] * vp[k])
        for i in range(5):
            for j in range(5):
                bQ[k, i, j] = 0.0
            bq[k, i] = 0.0
            for j in range(nl + 1):
                bC[k, i, j] = 0.0
        if k == 0:
            for i in range(5):
                bQ[0, i, i] = 2e-6
            for j in range(nl + 1):
                blg[0, j] = 0.0
                bug[0, j] = 0.0
            for i in range(5):
                blbx[0, i] = X0[i]
                bubx[0, i] = X0[i]
            continue
        bQ[k, 0, 0] = 2 * q_d
        bQ[k, 1, 1] = 2 * q_mu
        bQ[k, 2, 2] = 2e-6
        bQ[k, 3, 3] = 2 * wa
        bQ[k, 4, 4] = 2e-6
        bq[k, 0] = -2 * q_d * dr
        bq[k, 3] = -2 * wa * vp[k] * kr[k]
        mb = Xbar[k, 1]
        cmy = math.cos(mb)
        smy = math.sin(mb)
        Dk = 1.0 - kr[k] * Xbar[k, 0]
        if Dk < 0.2:
            Dk = 0.2
        keff = kr[k] / Dk
        for j in range(nl):
            lp = lpts[j]
            lo, hi = _corr1(st[k] + lp * cmy / Dk, wlo, whi, nf, fds, bx, bsd, w_h)
            const = lp * (smy - cmy * mb) - keff * lp * lp / 2
            lo = lo - const
            hi = hi - const
            if lo > hi:
                lo = hi = 0.5 * (lo + hi)
            cmu[k - 1, j] = lp * cmy
            lgc[k - 1, j] = lo
            ugc[k - 1, j] = hi
            bC[k, 0, j] = 1.0
            bC[k, 1, j] = lp * cmy
            blg[k, j] = lo
            bug[k, j] = hi
        bC[k, 3, nl] = vlat
        blg[k, nl] = -a_lat_hard
        bug[k, nl] = a_lat_hard
        blbx[k, 0] = -STEER_MAX * 0.98
        bubx[k, 0] = STEER_MAX * 0.98


@njit(cache=True)
def _targets_k(H, tt, tl, tb):
    hd, dl, hv, vc = False, 0.0, False, 0.0
    ts0 = tt + 1e-6 - tl - tb
    tv0 = tt + 1e-6 - tl
    for i in range(H.shape[0]):
        if H[i, 0] <= ts0:
            dl = H[i, 1]
            hd = True
        if H[i, 0] <= tv0:
            vc = H[i, 2]
            hv = True
    return hd, dl, hv, vc


@njit(cache=True)
def _servo_k(delta, has, tgt):
    if not has:
        return delta
    diff = tgt - delta
    step = STEER_RATE * 0.01
    if abs(diff) > 1e-4:
        delta += max(-step, min(step, diff))
    return max(-STEER_MAX, min(STEER_MAX, delta))


@njit(cache=True)
def _p_accel_k(v_cmd, v):
    dv = v_cmd - v
    if v > 0.0:
        a = (KP_ACC if dv > 0 else KP_BRK) * dv
    else:
        a = (2.0 * 9.51 / 20.0 if dv > 0 else 2.0 * 9.51 / 5.0) * dv
    pos = A_MAX * V_SWITCH / v if v > V_SWITCH else A_MAX
    return max(-A_MAX, min(pos, a))


@njit(cache=True)
def _bicycle_s_k(v, a, c):
    a = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
    cf = c * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
    vl = v * WB
    return (
        -(cr + cf) / vl,
        (cr * LR_ - cf * LF_) / (v * vl) - 1.0,
        cf / vl,
        M_ * (cr * LR_ - cf * LF_) / (I_ * WB),
        -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB),
        M_ * LF_ * cf / (I_ * WB),
    )


@njit(cache=True)
def _obs_main_k(H, tl, tb, t0, n, vf0, delta):
    m = max(n, 1)
    dmain = np.empty(m)
    amain = np.empty(m)
    vms = np.empty(m)
    vck = np.zeros(m)
    hvk = np.zeros(m, np.bool_)
    vm = vf0
    for k in range(n):
        hd, tgt, hv, vc = _targets_k(H, t0 + 0.01 * k, tl, tb)
        delta = _servo_k(delta, hd, tgt)
        a = _p_accel_k(vc, vm) if hv else 0.0
        dmain[k] = delta
        amain[k] = a
        vms[k] = vm
        vck[k] = vc
        hvk[k] = hv
        vm = max(0.0, vm + a * 0.01)
    return delta, vm, dmain, amain, vms, vck, hvk


@njit(cache=True)
def _obs_beta_k(n, vf0, vf, rf0, rf, yawf0, dyw, xf0, yf0, beta, dk, vck, hvk, c):
    has_mid = False
    beta_mid = 0.0
    px_ = xf0
    py_ = yf0
    for k in range(n):
        b_before = beta
        f = (k + 0.5) / max(n, 1)
        vk = vf0 + (vf - vf0) * f
        rk = rf0 + (rf - rf0) * f
        a = _p_accel_k(vck[k], vk) if hvk[k] else 0.0
        yk = yawf0 + dyw * f
        if vk >= 0.5:
            a11, a12, b1, _, _, _ = _bicycle_s_k(vk, a, c)
            e = math.exp(a11 * 0.01)
            bss = -(a12 * rk + b1 * dk[k]) / a11
            bmid_ = bss + (beta - bss) * math.exp(a11 * 0.005)
            beta = bss + (beta - bss) * e
        else:
            bmid_ = 0.0
            beta = 0.0
        px_ += vk * math.cos(yk + bmid_) * 0.01
        py_ += vk * math.sin(yk + bmid_) * 0.01
        h2 = 2 * (k + 1)
        if h2 == n:
            beta_mid = beta
            has_mid = True
        elif h2 == n + 1:
            beta_mid = 0.5 * (b_before + beta)
            has_mid = True
    return beta, has_mid, beta_mid, px_, py_


@njit(cache=True)
def _mm_prop(bb, rr, dg, cv, lg, nc, H, tl, tb, t0, n, vf0, v, dmain, amain, vms):
    nl = lg.shape[0]
    NH = H.shape[0]
    NM = bb.shape[0]
    dvec = np.empty(NM)
    avec = np.empty(NM)
    ag = np.empty(nl)
    h = 0.01
    for k in range(n):
        tt = t0 + 0.01 * k
        vk = vf0 + (v - vf0) * (k + 0.5) / max(n, 1)
        for l in range(nl):
            tq = tt - lg[l] + 1e-6
            ts0 = tq - tl - tb
            tv0 = tq - tl
            tg = dg[l]
            vc = vk
            for j in range(NH - 1, -1, -1):
                if H[j, 0] <= ts0:
                    tg = H[j, 1]
                    break
            for j in range(NH - 1, -1, -1):
                if H[j, 0] <= tv0:
                    vc = H[j, 2]
                    break
            diff = tg - dg[l]
            if abs(diff) > 1e-4:
                dg[l] += min(max(diff, -STEER_RATE * 0.01), STEER_RATE * 0.01)
            dg[l] = min(max(dg[l], -STEER_MAX), STEER_MAX)
            dvc = vc - vk
            a = ((KP_ACC if dvc > 0 else KP_BRK) * dvc) if vk > 0.0 else 0.0
            amax = A_MAX * V_SWITCH / vk if vk > V_SWITCH else A_MAX
            ag[l] = min(max(a, -A_MAX), amax)
        dvec[0] = dmain[k]
        avec[0] = amain[k]
        for l in range(nl):
            for i in range(nc):
                dvec[1 + l * nc + i] = dg[l]
                avec[1 + l * nc + i] = ag[l]
        vm = vms[k]
        if vm >= 0.5:
            for i in range(NM):
                cf = cv[i] * MU_ * CSF_ * (G_ * LR_ - avec[i] * HCG_)
                cr = cv[i] * MU_ * CSR_ * (G_ * LF_ + avec[i] * HCG_)
                a11 = -(cr + cf) / (vm * WB)
                a12 = (cr * LR_ - cf * LF_) / (vm * vm * WB) - 1.0
                a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
                a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (vm * I_ * WB)
                ub = cf / (vm * WB) * dvec[i]
                ur = M_ * LF_ * cf / (I_ * WB) * dvec[i]
                b0, r0 = bb[i], rr[i]
                k1b = a11 * b0 + a12 * r0 + ub
                k1r = a21 * b0 + a22 * r0 + ur
                b_, r_ = b0 + 0.5 * h * k1b, r0 + 0.5 * h * k1r
                k2b = a11 * b_ + a12 * r_ + ub
                k2r = a21 * b_ + a22 * r_ + ur
                b_, r_ = b0 + 0.5 * h * k2b, r0 + 0.5 * h * k2r
                k3b = a11 * b_ + a12 * r_ + ub
                k3r = a21 * b_ + a22 * r_ + ur
                b_, r_ = b0 + h * k3b, r0 + h * k3r
                k4b = a11 * b_ + a12 * r_ + ub
                k4r = a21 * b_ + a22 * r_ + ur
                bb[i] = b0 + h / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
                rr[i] = r0 + h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        else:
            for i in range(NM):
                bb[i] = 0.0
                rr[i] = vm * math.tan(dvec[i]) / WB


@njit(cache=True)
def _predict_k(H, tl, tb, t, x, y, yaw, v, r, t_pred, delta, beta, c):
    n = int(round(t_pred / 0.01))
    a = 0.0
    for k in range(n):
        tt = t + 0.01 * k
        hd, tgt, hv, vc = _targets_k(H, tt, tl, tb)
        delta = _servo_k(delta, hd, tgt)
        a = _p_accel_k(vc, v) if hv else 0.0
        if v >= 0.5:
            a11, a12, b1, a21, a22, b2 = _bicycle_s_k(v, a, c)
            m = max(1, int(math.ceil(-0.01 * min(a11, a22) / 0.7)))
            h = 0.01 / m
            for _ in range(m):
                bd = a11 * beta + a12 * r + b1 * delta
                rd = a21 * beta + a22 * r + b2 * delta
                x += v * math.cos(yaw + beta) * h
                y += v * math.sin(yaw + beta) * h
                yaw += r * h
                beta += bd * h
                r += rd * h
        else:
            x += v * math.cos(yaw) * 0.01
            y += v * math.sin(yaw) * 0.01
            r = v * math.tan(delta) / WB
            yaw += r * 0.01
            beta = 0.0
        v = max(0.0, v + a * 0.01)
    return x, y, yaw, v, beta, r, delta, a


@njit(cache=True)
def _hp_slack_k(Xs, cmu, lgc, ugc):
    N, nl = cmu.shape
    sl = np.zeros(N + 1)
    for k in range(N):
        m = 0.0
        for j in range(nl):
            val = Xs[k + 1, 0] + cmu[k, j] * Xs[k + 1, 1]
            if lgc[k, j] - val > m:
                m = lgc[k, j] - val
            if val - ugc[k, j] > m:
                m = val - ugc[k, j]
        sl[k + 1] = m
    return sl

