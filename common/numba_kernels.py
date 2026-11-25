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

