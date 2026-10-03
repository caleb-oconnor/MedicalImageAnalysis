"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

_raster.py - shared internals for the conversion modules (not public API).

  _group / _region_volumes        contour list -> per-region field volumes   (to_mesh, to_mask)
  volume_to_grid                  field volume -> uint8 0..255 on the CT grid (to_mesh, to_mask)
  _loops / _cut / _flat           mesh plane cuts, VTK segments -> rings     (to_contours, to_mask)
  fill_contours / rings_to_mask   rings on CT slices -> binary / 0..255 mask (to_mask)
"""
import numpy as np
import vtk
from scipy import ndimage
from scipy.spatial import cKDTree
from vtk.util import numpy_support as nps

from ._contourmesh import (_Volume, _S2D, ZSEP, _links, _prepare, _seg_dist,
                           auto_h, group_contours, split_regions)

SENTINEL = 1e3          # |field| above this is BIG (1e4) or mixed with it


def _to_u8(f, w):
    """signed distance (negative inside) -> 0..255, 128 at the boundary."""
    return np.rint(np.clip(0.5 - f / w, 0.0, 1.0) * 255).astype(np.uint8)


# =============================================================================== contours -> field volumes
def _group(contours, origin, spacing, direction):
    """list of (n,3) world rings -> (slices dict, contour frame, contour slice spacing)."""
    return group_contours(contours, direction, origin, spacing)


def _region_volumes(slices, frame, slice_spacing, m=1, c=None, gap_factor=1.5, pad=0.5,
                    workers=-1, mid="auto", flat_caps=False, cap_radius=None, regions=True):
    """Same regions / h / _Volume settings as contours_to_mesh_regions, yielded one at a time."""
    groups = split_regions(slices, frame, slice_spacing, gap_factor, pad) if regions else [slices]
    for g in groups:
        h = auto_h(g, frame)
        _, s, loc = _prepare(g, frame)
        if loc:
            yield _Volume(loc, s, slice_spacing, h, m, h if c is None else c, 3, 0,
                          gap_factor, workers, mid, flat_caps, cap_radius)


# =============================================================================== field -> CT grid
def volume_to_grid(vol, frame, shape_zyx, origin, spacing, direction, ramp=None, out=None):
    """Sample a _Volume field at every CT voxel centre -> uint8 (nz, ny, nx), 0..255.
    Trilinear in (x, y, plane index), plane index -> z piecewise linear: the same
    interpolation Flying Edges uses along lattice edges. Unions into `out` by max.

    BIG sentinels (and values mixed with them) are replaced by +-max(h, w) first: FE only
    interpolates along edges that cross zero (exact at both ends by construction), but a
    trilinear cell interior sees all 8 corners and a 1e4 corner would drag the boundary."""
    nz, ny, nx = shape_zyx
    Dc, sc, Oc = (np.asarray(a, float) for a in (direction, spacing, origin))
    if out is None:
        out = np.zeros(shape_zyx, np.uint8)
    w = float(min(sc[0], sc[1]) if ramp is None else ramp)
    R = np.float32(max(vol.h, w))
    h, i0, j0, pz = vol.h, vol.i0, vol.j0, vol.pz
    P = len(pz)
    pidx = np.arange(P, dtype=float)

    # CT index (i, j, k) -> contour-frame local (x, y, z):  q = g + G @ ijk
    Fr = np.c_[frame.u, frame.v, frame.n]
    G = Fr.T @ (Dc * sc)
    g = Fr.T @ (Oc - frame.origin)

    # bounding box of the volume in CT index space
    lo_l = (i0 * h, j0 * h, pz[0])
    hi_l = ((i0 + vol.nx - 1) * h, (j0 + vol.ny - 1) * h, pz[-1])
    corners = np.array([[a, b, z] for a in (lo_l[0], hi_l[0])
                        for b in (lo_l[1], hi_l[1]) for z in (lo_l[2], hi_l[2])])
    ijk = np.linalg.solve(G, (corners - g).T).T
    lo = np.maximum(np.floor(ijk.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + 1, (nx, ny, nz))
    if np.any(hi <= lo):
        return out

    Vc = vol.V.copy()
    s = np.abs(Vc) > SENTINEL
    Vc[s] = np.copysign(R, Vc[s])

    def put(k, f):
        sl = out[k, lo[1]:hi[1], lo[0]:hi[0]]
        np.maximum(sl, _to_u8(f, w).reshape(sl.shape), out=sl)

    if abs(G[2, 0]) < 1e-6 and abs(G[2, 1]) < 1e-6:
        # contour planes parallel to CT slices: one z per CT slice -> blend two planes,
        # then a 2D affine resample (no coordinate arrays)
        A2 = G[:2, :2] / h
        M = np.array([[A2[1, 1], A2[1, 0]], [A2[0, 1], A2[0, 0]]])
        shp = (hi[1] - lo[1], hi[0] - lo[0])
        for k in range(lo[2], hi[2]):
            z = g[2] + G[2, 2] * k
            if not pz[0] < z < pz[-1]:
                continue
            p = np.interp(z, pz, pidx)
            p0 = min(int(p), P - 2)
            t = p - p0
            f = (1 - t) * Vc[p0] + t * Vc[p0 + 1]
            cc = (G[:2, 2] * k + g[:2]) / h - (i0, j0)
            off = M @ (lo[1], lo[0]) + (cc[1], cc[0])
            put(k, ndimage.affine_transform(f, M, offset=off, output_shape=shp, order=1,
                                            mode="constant", cval=R, prefilter=False))
    else:
        # oblique / sagittal / coronal contours on an axial grid: z varies within a CT slice
        jj, ii = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        base = g[:, None] + G[:, :2] @ np.stack([ii.ravel(), jj.ravel()])
        for k in range(lo[2], hi[2]):
            q = base + G[:, 2:3] * k
            p = np.interp(q[2], pz, pidx, left=-10.0, right=P + 10.0)
            put(k, ndimage.map_coordinates(Vc, [p, q[1] / h - j0, q[0] / h - i0], order=1,
                                           mode="constant", cval=R, prefilter=False))
    return out


# =============================================================================== mesh cuts / loops
def _loops(pd):
    """vtkPolyData of line segments -> list of closed (n,3) point loops."""
    st = vtk.vtkStripper()
    st.SetInputData(pd)
    st.JoinContiguousSegmentsOn()
    st.Update()
    out = st.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return []
    P = nps.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    lines = out.GetLines()
    off = nps.vtk_to_numpy(lines.GetOffsetsArray())
    con = nps.vtk_to_numpy(lines.GetConnectivityArray())
    loops = []
    for a, b in zip(off[:-1], off[1:]):
        ids = con[a:b]
        if len(ids) > 1 and ids[0] == ids[-1]:
            ids = ids[:-1]
        if len(ids) >= 3:
            loops.append(P[ids])
    return loops


def _cut(poly, origin, direction, slice_positions, eps=1e-5):
    """Cut a closed mesh with planes normal to the image k axis in one pass.
    slice_positions: distances along k (mm) from `origin`.
    Returns {index into slice_positions: [(n,3) world rings]}."""
    D = np.asarray(direction, float)
    n = D[:, 2]
    origin = np.asarray(origin, float)
    P = nps.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    q = vtk.vtkPolyData()
    q.ShallowCopy(poly)
    q.GetPointData().SetScalars(nps.numpy_to_vtk((P - origin) @ n, deep=True))
    cf = vtk.vtkContourFilter()
    cf.SetInputData(q)
    vals = np.asarray(slice_positions, float)
    for i, v in enumerate(vals):
        cf.SetValue(i, v + eps)          # tiny offset: meshes may have vertices exactly on slice planes
    cf.UseScalarTreeOn()
    cf.ComputeScalarsOff(); cf.ComputeNormalsOff(); cf.ComputeGradientsOff()
    cf.Update()
    out = {}
    for ring in _loops(cf.GetOutput()):
        k = int(np.argmin(np.abs(vals - ((ring - origin) @ n).mean())))
        out.setdefault(k, []).append(ring)
    return out


def _flat(rings_by_slice):
    """{slice_index: [rings]} -> flat list of rings in slice order."""
    return [r for k in sorted(rings_by_slice) for r in rings_by_slice[k]]


# =============================================================================== rings -> mask
def fill_contours(rings_by_slice, shape_zyx, origin, spacing, direction):
    """{slice_index: [(n,3) world rings]} -> binary uint8 mask (nz, ny, nx). Even-odd scanline
    fill of every ring of every slice at once; a pixel is inside when its centre is inside."""
    nz, ny, nx = shape_zyx
    D = np.asarray(direction, float)
    sp = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    P, nxt, ks = [], [], []
    base = 0
    for k, rings in rings_by_slice.items():
        if not 0 <= k < nz:
            continue
        for r in rings:
            ij = (((np.asarray(r, float) - origin) @ D) / sp)[:, :2]   # continuous (i, j)
            m = len(ij)
            P.append(ij)
            nxt.append(np.r_[np.arange(1, m), 0] + base)
            ks.append(np.full(m, k))
            base += m
    mask = np.zeros((nz, ny, nx), np.uint8)
    if not P:
        return mask
    a = np.vstack(P); b = a[np.concatenate(nxt)]; kk = np.concatenate(ks)
    # work only inside the rings' bounding box (slices, rows, cols), then paste
    k0, k1 = kk.min(), kk.max() + 1
    j0 = max(int(np.floor(a[:, 1].min())), 0); j1 = min(int(np.ceil(a[:, 1].max())) + 1, ny)
    i0 = max(int(np.floor(a[:, 0].min())), 0); i1 = min(int(np.ceil(a[:, 0].max())) + 1, nx)
    if j1 <= j0 or i1 <= i0:
        return mask
    cz, cy, cx = k1 - k0, j1 - j0, i1 - i0
    a = a - (i0, j0); b = b - (i0, j0); kk = kk - k0
    ya, yb = a[:, 1], b[:, 1]
    r0 = np.ceil(np.minimum(ya, yb)).astype(np.int64)
    n = np.ceil(np.maximum(ya, yb)).astype(np.int64) - r0
    keep = n > 0
    nk = n[keep]
    e = np.repeat(np.flatnonzero(keep), nk)
    rows = np.repeat(r0[keep], nk) + (np.arange(e.size) - np.repeat(np.cumsum(nk) - nk, nk))
    ok = (rows >= 0) & (rows < cy)
    e, rows = e[ok], rows[ok]
    t = (rows - ya[e]) / (yb[e] - ya[e])
    x = a[e, 0] + t * (b[e, 0] - a[e, 0])
    col = np.clip(np.floor(x).astype(np.int64) + 1, 0, cx)
    lin = (kk[e] * cy + rows) * (cx + 1) + col
    par = (np.bincount(lin, minlength=cz * cy * (cx + 1)) & 1).astype(np.uint8)
    mask[k0:k1, j0:j1, i0:i1] = np.bitwise_xor.accumulate(par.reshape(cz, cy, cx + 1), axis=2)[:, :, :cx]
    return mask


def rings_to_mask(rings_by_slice, shape_zyx, origin, spacing, direction, ramp=None):
    """{slice_index: [(n,3) world rings]} -> uint8 (nz, ny, nx), 0..255, >= 128 inside.
    Exact in-plane signed distance near the rings (same quantity as the field's slice
    planes), even-odd fill for the sign and for the saturated interior."""
    nz, ny, nx = shape_zyx
    D, sp, O = (np.asarray(a, float) for a in (direction, spacing, origin))
    w = float(min(sp[0], sp[1]) if ramp is None else ramp)
    inside = fill_contours(rings_by_slice, shape_zyx, origin, spacing, direction)
    out = inside * np.uint8(255)

    P, rid, ks, n = [], [], [], 0
    for k, rings in rings_by_slice.items():
        if not 0 <= k < nz:
            continue
        for r in rings:
            P.append(((np.asarray(r, float) - O) @ D)[:, :2])      # mm along i, j
            rid.append(np.full(len(P[-1]), n)); ks.append(np.full(len(P[-1]), k)); n += 1
    if not P:
        return out
    P, rid, ks = np.vstack(P), np.concatenate(rid), np.concatenate(ks)

    # densify to <= half a pixel
    _, nxt = _links(rid)
    cnt = np.maximum(1, np.ceil(np.linalg.norm(P[nxt] - P, axis=1)
                                / (0.5 * min(sp[0], sp[1]))).astype(np.int64))
    e = np.repeat(np.arange(len(P)), cnt)
    fr = (np.arange(e.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)) / cnt[e]
    Q = P[e] + (P[nxt][e] - P[e]) * fr[:, None]
    qk = ks[e]
    qprv, qnxt = _links(rid[e])

    # band = 8-connected dilation of the pixels the boundary passes through
    ti = np.clip(np.rint(Q[:, 0] / sp[0]).astype(np.int64), 0, nx - 1)
    tj = np.clip(np.rint(Q[:, 1] / sp[1]).astype(np.int64), 0, ny - 1)
    k0, k1 = qk.min(), qk.max() + 1
    j0, j1 = max(tj.min() - 1, 0), min(tj.max() + 2, ny)
    i0, i1 = max(ti.min() - 1, 0), min(ti.max() + 2, nx)
    band = np.zeros((k1 - k0, j1 - j0, i1 - i0), bool)
    band[qk - k0, tj - j0, ti - i0] = True
    band = ndimage.binary_dilation(band, structure=_S2D)
    bk, bj, bi = np.nonzero(band)
    bk += k0; bj += j0; bi += i0

    # exact point-to-polyline distance (nearest sample, then its two segments)
    q = np.c_[bi * sp[0], bj * sp[1]]
    _, nn = cKDTree(np.c_[Q, qk * ZSEP]).query(np.c_[q, bk * ZSEP], workers=-1)
    d = np.minimum(_seg_dist(q, Q[qprv[nn]], Q[nn]), _seg_dist(q, Q[nn], Q[qnxt[nn]]))
    f = np.where(inside[bk, bj, bi] > 0, -d, d)
    out[bk, bj, bi] = _to_u8(f, w)
    return out