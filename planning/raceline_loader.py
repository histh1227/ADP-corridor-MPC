"""맵, 레이싱라인 로딩 및 Frenet 좌표 변환 모듈.

기존 cmpc_berlin_tt.py의 _load_occ, _load_track, _custom_line, _set_line,
_repair_line, _load_map 및 Frenet 좌표 변환 유틸리티를 분리했다.
- 하드코딩된 ~/forza_ws 경로 대신 common.paths.resolve_maps_dir() 사용
"""
import json
import math
import os

import numpy as np
import scipy.sparse as sp
import yaml
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from common.numba_kernels import _frenet_k
from common.paths import resolve_maps_dir, resolve_project_root
from common.vehicle_params import STEER_MAX, WB, wrap_angle


class RacelineTracker:
    def __init__(self, map_name: str = "berlin", raceline_file: str = None,
                 wall_clear: float = 0.16, wall_pref: float = 0.28,
                 line_ktrig: float = 0.9, line_kmax: float = 1.0,
                 line_dkmax: float = 2.0, line_clr: float = 0.35, line_wdev: float = 0.001):
        self.map_name = map_name
        self.raceline_file = raceline_file
        self.wall_clear = wall_clear
        self.wall_pref = wall_pref
        self.wc_eff = wall_clear
        self.line_ktrig = line_ktrig
        self.line_kmax = line_kmax
        self.line_dkmax = line_dkmax
        self.line_clr = line_clr
        self.line_wdev = line_wdev

        self.mapdir = os.path.join(str(resolve_maps_dir()), self.map_name)
        self._load_occ()
        self._load_track()
        self._load_map()

        kc = int(round(min(max(self.wall_clear, 0.0), 0.6) / 0.01))
        self.wlo_c = np.ascontiguousarray(self.wlo_tab[kc])
        self.whi_c = np.ascontiguousarray(self.whi_tab[kc])
        kp = int(round(min(max(self.wall_pref, 0.0), 0.6) / 0.01))
        self.wlo_p = np.ascontiguousarray(self.wlo_tab[kp])
        self.whi_p = np.ascontiguousarray(self.whi_tab[kp])

        # Numba 워밍업
        _frenet_k(self.fx, self.fy, 0.0, 0.0, -1)

    def _load_occ(self):
        y = f'{self.mapdir}/{self.map_name}_orig.yaml'
        if not os.path.exists(y):
            y = f'{self.mapdir}/{self.map_name}.yaml'
        meta = yaml.safe_load(open(y))
        img = np.array(
            Image.open(f"{self.mapdir}/{meta['image']}").convert('L').transpose(Image.FLIP_TOP_BOTTOM)
        )
        occ = img <= 128
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.H, self.W = occ.shape
        self.edt = (ndimage.distance_transform_edt(~occ) * self.res).astype(np.float32)

    def _load_track(self):
        w = json.load(open(f'{self.mapdir}/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts']
        s = np.array([q['s_m'] for q in w])
        x = np.array([q['x_m'] for q in w])
        y = np.array([q['y_m'] for q in w])
        vx = np.array([q['vx_mps'] for q in w])
        x[-1], y[-1] = x[0], y[0]
        if self.raceline_file:
            s, x, y, vx = self._custom_line(s, x, y, vx, float(w[0]['psi_rad']))
        self._set_line(s, x, y)
        self.fvx = np.interp(self.fs, s, vx)
        self.line_repaired = False
        if self.line_kmax > 0.0 and float(np.max(np.abs(self.fk))) > self.line_ktrig * math.tan(STEER_MAX) / WB:
            xo, yo = self.fx.copy(), self.fy.copy()
            xr, yr = self._repair_line()
            sr = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xr), np.diff(yr)))])
            self._set_line(sr, xr, yr)
            _, io = cKDTree(np.stack([xo, yo], 1)).query(np.stack([self.fx, self.fy], 1))
            self.fvx = np.interp(io * (s[-1] / xo.size), s, vx)
            self.line_repaired = True
        self.tree = cKDTree(np.stack([self.fx, self.fy], 1))

    def _custom_line(self, s_m, x_m, y_m, vx_m, psi0):
        path = self.raceline_file
        if not os.path.isabs(path):
            path = os.path.join(str(resolve_project_root()), path)
        d = json.load(open(path))
        if isinstance(d, dict) and 'global_traj_wpnts_iqp' in d:
            d = d['global_traj_wpnts_iqp']
        if isinstance(d, dict) and 'wpnts' in d:
            d = d['wpnts']
        if isinstance(d, list):
            x = np.array([q['x_m'] for q in d], dtype=float)
            y = np.array([q['y_m'] for q in d], dtype=float)
            v = np.array([q['vx_mps'] for q in d], dtype=float) if 'vx_mps' in d[0] else None
        else:
            x, y = np.asarray(d['x'], dtype=float), np.asarray(d['y'], dtype=float)
            v = np.asarray(d['vx'], dtype=float) if 'vx' in d else None
        if math.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
            x, y = x[:-1], y[:-1]
            v = v[:-1] if v is not None else None
        hx, hy = math.cos(psi0), math.sin(psi0)
        xc, yc = np.append(x, x[0]), np.append(y, y[0])
        f = (xc - x_m[0]) * hx + (yc - y_m[0]) * hy
        best = None
        for i in np.flatnonzero((f[:-1] < 0.0) & (f[1:] >= 0.0)):
            a = -f[i] / (f[i + 1] - f[i])
            px, py = xc[i] + a * (xc[i + 1] - xc[i]), yc[i] + a * (yc[i + 1] - yc[i])
            dist = math.hypot(px - x_m[0], py - y_m[0])
            if best is None or dist < best[0]:
                best = (dist, i, a, px, py)
        if best is None or best[0] > 3.0:
            raise ValueError(f'raceline_file {path}: does not cross the finish line near map s = 0')
        _, i, a, px, py = best
        n = x.size
        order = (np.arange(n) + i + 1) % n
        x2 = np.concatenate([[px], x[order], [px]])
        y2 = np.concatenate([[py], y[order], [py]])
        seg = np.hypot(np.diff(x2), np.diff(y2))
        keep = np.concatenate([[True], seg > 1e-6])
        x2, y2 = x2[keep], y2[keep]
        s2 = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x2), np.diff(y2)))])
        if v is not None:
            vc = np.append(v, v[0])
            v0 = vc[i] + a * (vc[i + 1] - vc[i])
            v2 = np.concatenate([[v0], v[order], [v0]])[keep]
        else:
            _, io = cKDTree(np.stack([x_m, y_m], 1)).query(np.stack([x2, y2], 1))
            v2 = vx_m[io]
        self.line_file = path
        return s2, x2, y2, v2

    def _set_line(self, s, x, y):
        L = float(s[-1])
        csx = CubicSpline(s, x, bc_type='periodic')
        csy = CubicSpline(s, y, bc_type='periodic')
        nf = int(round(L / 0.05))
        fs = np.arange(nf) * (L / nf)
        dx, dy = csx(fs, 1), csy(fs, 1)
        ddx, ddy = csx(fs, 2), csy(fs, 2)
        self.L, self.nf, self.fds = L, nf, L / nf
        self.fs = fs
        self.fx, self.fy = csx(fs), csy(fs)
        self.fpsi = np.arctan2(dy, dx)
        self.fk = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        self.ftx, self.fty = np.cos(self.fpsi), np.sin(self.fpsi)
        self.fnx, self.fny = -self.fty, self.ftx

    def clearance(self, x, y):
        return ndimage.map_coordinates(
            self.edt,
            [(y - self.oy) / self.res - 0.5, (x - self.ox) / self.res - 0.5],
            order=1, mode='constant', cval=0.0,
        )

    def _repair_line(self):
        h = 0.2
        n = int(round(self.L / h))
        h = self.L / n
        i0 = np.rint(np.arange(n) * h / self.fds).astype(int) % self.nf
        x0, y0 = self.fx[i0], self.fy[i0]
        kmax = self.line_kmax
        clr = np.minimum(self.line_clr, self.clearance(x0, y0))

        def res(z):
            x, y = z[:n], z[n:]
            ax, ay = x - np.roll(x, 1), y - np.roll(y, 1)
            bx, by = np.roll(x, -1) - x, np.roll(y, -1) - y
            la = np.hypot(ax, ay)
            k = 2.0 * (ax * by - ay * bx) / np.maximum(la * np.hypot(bx, by) * np.hypot(ax + bx, ay + by), 1e-9)
            dk = (k - np.roll(k, 1)) / h
            return np.concatenate([
                math.sqrt(h) * np.maximum(np.abs(k) - kmax, 0.0),
                math.sqrt(h) * 0.2 * np.maximum(np.abs(dk) - self.line_dkmax, 0.0),
                3.0 * np.maximum(clr - self.clearance(x, y), 0.0),
                math.sqrt(self.line_wdev) * (x - x0),
                math.sqrt(self.line_wdev) * (y - y0),
                la - h,
            ])

        rows, cols = [], []
        ii = np.arange(n)
        for dj in (-1, 0, 1):
            rows += [ii, ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        for dj in (-2, -1, 0, 1):
            rows += [n + ii, n + ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        rows += [2 * n + ii, 2 * n + ii, 3 * n + ii, 4 * n + ii]
        cols += [ii, n + ii, ii, n + ii]
        for dj in (-1, 0):
            rows += [5 * n + ii, 5 * n + ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        jac = sp.csr_matrix(
            (np.ones(sum(r.size for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
            shape=(6 * n, 2 * n),
        )
        r = least_squares(res, np.concatenate([x0, y0]), jac_sparsity=jac, method='trf',
                          max_nfev=600, xtol=1e-8, ftol=1e-10)
        x, y = r.x[:n], r.x[n:]
        return np.append(x, x[0]), np.append(y, y[0])

    def _load_map(self):
        self.mres = self.res / 2.0
        self.MH, self.MW = self.H * 2, self.W * 2
        cc, rr = np.meshgrid(np.arange(self.MW), np.arange(self.MH))
        mx = self.ox + (cc.ravel() + 0.5) * self.mres
        my = self.oy + (rr.ravel() + 0.5) * self.mres
        dist, idx = self.tree.query(np.stack([mx, my], 1))
        ddx, ddy = mx - self.fx[idx], my - self.fy[idx]
        self.cell_x = mx.astype(np.float32)
        self.cell_y = my.astype(np.float32)
        self.cell_s = (self.fs[idx] + ddx * self.ftx[idx] + ddy * self.fty[idx]).astype(np.float32)
        self.cell_d = (ddx * self.fnx[idx] + ddy * self.fny[idx]).astype(np.float32)
        self.cell_ok = dist < 4.0

        dg = np.arange(-4.0, 4.0001, 0.01)
        px = self.fx[:, None] + dg[None, :] * self.fnx[:, None]
        py = self.fy[:, None] + dg[None, :] * self.fny[:, None]
        rows = (py - self.oy) / self.res - 0.5
        cols = (px - self.ox) / self.res - 0.5
        clr = ndimage.map_coordinates(self.edt, [rows.ravel(), cols.ravel()], order=1, mode='constant', cval=0.0)
        clr = clr.reshape(px.shape) - 0.03
        cl = np.arange(0.0, 0.61, 0.01)
        j0 = int(np.argmin(np.abs(dg)))
        lo = np.zeros((len(cl), self.nf))
        hi = np.zeros((len(cl), self.nf))
        for k, c in enumerate(cl):
            bad = clr < c
            badl = bad[:, :j0][:, ::-1]
            badr = bad[:, j0:]
            il = np.where(badl.any(1), np.argmax(badl, axis=1), badl.shape[1])
            ir = np.where(badr.any(1), np.argmax(badr, axis=1), badr.shape[1])
            lo[k] = dg[np.clip(j0 - il, 0, len(dg) - 1)]
            hi[k] = dg[np.clip(j0 + ir - 1, 0, len(dg) - 1)]
        self.wlo_tab, self.whi_tab = lo, hi

    def set_wall(self, wc: float):
        self.wc_eff = wc
        kc = int(round(min(max(wc, 0.0), 0.6) / 0.01))
        self.wlo_c = np.ascontiguousarray(self.wlo_tab[kc])
        self.whi_c = np.ascontiguousarray(self.whi_tab[kc])

    def fidx(self, s):
        return np.mod(np.rint(np.asarray(s) / self.fds).astype(int), self.nf)

    def finterp(self, arr, s):
        u = np.mod(np.asarray(s, dtype=float), self.L) / self.fds
        i0 = np.floor(u).astype(int) % self.nf
        f = u - np.floor(u)
        return arr[i0] * (1 - f) + arr[(i0 + 1) % self.nf] * f

    def xy(self, s, d):
        u = np.mod(np.asarray(s, dtype=float), self.L) / self.fds
        i0 = np.floor(u).astype(int) % self.nf
        i1 = (i0 + 1) % self.nf
        f = u - np.floor(u)
        bx = self.fx[i0] * (1 - f) + self.fx[i1] * f
        by = self.fy[i0] * (1 - f) + self.fy[i1] * f
        nx = self.fnx[i0] * (1 - f) + self.fnx[i1] * f
        ny = self.fny[i0] * (1 - f) + self.fny[i1] * f
        return bx + d * nx, by + d * ny

    def frenet(self, x: float, y: float, guess):
        i = int(_frenet_k(self.fx, self.fy, float(x), float(y), -1 if guess is None else int(guess)))
        if guess is not None and (i == (guess - 12) % self.nf or i == (guess + 39) % self.nf):
            return self.frenet(x, y, None)
        dx, dy = x - self.fx[i], y - self.fy[i]
        return self.fs[i] + dx * self.ftx[i] + dy * self.fty[i], dx * self.fnx[i] + dy * self.fny[i], i

    def wall_bounds(self, s, c: float):
        k = int(round(min(max(c, 0.0), 0.6) / 0.01))
        i = self.fidx(s)
        return self.wlo_tab[k, i], self.whi_tab[k, i]

    def heading(self, s: float) -> float:
        u = (s % self.L) / self.fds
        fu = math.floor(u)
        i0 = int(fu) % self.nf
        i1 = (i0 + 1) % self.nf
        f = u - fu
        return float(self.fpsi[i0] + wrap_angle(float(self.fpsi[i1] - self.fpsi[i0])) * f)
