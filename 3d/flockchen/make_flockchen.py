"""
Flöckchen (tonies Adventskalender 2025) - v7, full redesign from the original gallery photo.
Measured on the gallery image (1160 px figure height = 119 units, k = 0.1034 units/px, x centred on the figure):
- head ~600 px wide, ~400 px tall (chin to top of head): radii 31 x 22 x 21, centre z 63
- body is wide and round (~440 px at chest, ~600 px at the feet), sits between the feet, top under the head:
  radii 22 x 20 x 21, centre z 22
- big round feet at the front-sides, bottom at z=0, top at z~24: radii 10 x 12 x 12
- arms thick, crossed in front of the belly, paws at z~20
- ears: left long (tilted back-left), right bent out to the side (same as v5c)
- light string: yellow bulb on top of head, teal on right ear, red on the cheek, green on the chest
Side view is not in the photos, so depth (y) is set symmetric to the front photo.
Single color, one watertight mesh. Z up, base z=0.
"""
import numpy as np
import trimesh
from skimage import measure

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)

def smax(a, b, k=0.0):
    return -smin(-a, -b, k) if k else np.maximum(a, b)

def ellipsoid(P, c, r):
    p = P - np.asarray(c, np.float32)
    r = np.asarray(r, np.float32)
    k0 = np.linalg.norm(p / r, axis=-1)
    k1 = np.maximum(np.linalg.norm(p / (r * r), axis=-1), 1e-9)
    return k0 * (k0 - 1.0) / k1

def ear_basis(direction):
    """Ear frame: local z = base->tip, local y = thickness (toward the camera), local x = width."""
    dz = np.asarray(direction, float); dz /= np.linalg.norm(dz)
    t = np.array([0.0, 1.0, 0.0]) - dz * dz[1]
    t /= np.linalg.norm(t)
    w = np.cross(t, dz)
    return np.stack([w, t, dz], axis=1).astype(np.float32)

def ellipsoid_rot(P, c, r, R):
    p = (P - np.asarray(c, np.float32)) @ R
    return ellipsoid(p, (0, 0, 0), r)

def capsule(P, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    r = ra + (rb - ra) * h
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r

def sphere(P, c, r):
    return np.linalg.norm(P - np.asarray(c, np.float32), axis=-1) - r

def polyline(P, pts, r):
    out = None
    for a, b in zip(pts[:-1], pts[1:]):
        s = capsule(P, a, b, r)
        out = s if out is None else np.minimum(out, s)
    return out

# ---------- grid ----------
VOX = 0.45
xs = np.arange(-40, 62 + VOX, VOX, dtype=np.float32)
ys = np.arange(-38, 30 + VOX, VOX, dtype=np.float32)
zs = np.arange(-3, 126 + VOX, VOX, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
P = np.stack([X, Y, Z], axis=-1)
del X, Y, Z

# ---------- body ----------
d = ellipsoid(P, (0, -1, 66), (28, 21, 21))                       # head
d = smin(d, ellipsoid(P, (0, 2, 25), (24, 22, 25)), 7)             # wide round body, sits under the head
d = smin(d, sphere(P, (-20, -9, 55), 9.5), 5)                     # fluffy cheeks
d = smin(d, sphere(P, (20, -9, 55), 9.5), 5)
d = smin(d, ellipsoid(P, (0, -3, 91), (6.0, 5.0, 4.5)), 3)         # head tuft between the ears

# ears (same measured placement as v5c)
d = smin(d, ellipsoid_rot(P, (-19, -0.5, 103), (10.0, 3.6, 25.0), ear_basis((-22, -1, 30))), 3)
d = smin(d, ellipsoid_rot(P, (29, 0, 93), (10.0, 3.6, 25.0), ear_basis((32, -1, 14))), 3)

# feet: big round paws at the front-sides, bottom at z=0
d = smin(d, ellipsoid(P, (-15, -19, 7), (9.5, 12.0, 7.0)), 4)
d = smin(d, ellipsoid(P, (15, -19, 7), (9.5, 12.0, 7.0)), 4)
# arms: thick, crossed paws in front of the belly
d = smin(d, capsule(P, (-15, -14, 38), (-6, -22, 27), 6.5, 5.5), 3)
d = smin(d, capsule(P, (15, -14, 38), (6, -22, 27), 6.5, 5.5), 3)
# small round tail at the back
d = smin(d, sphere(P, (0, 21, 16), 8.5), 3)

# ---------- face ----------
d = smin(d, ellipsoid(P, (-10.5, -22, 67), (4.6, 2.8, 4.8)), 1.5)  # left eye
d = smin(d, ellipsoid(P, (10.5, -22, 67), (4.6, 2.8, 4.8)), 1.5)   # right eye
d = smin(d, ellipsoid(P, (0, -22.5, 60), (3.6, 2.0, 1.6)), 1.0)    # nose
d = smax(d, -polyline(P, np.array([(-11, -23, 57), (-5, -24.5, 52), (0, -25, 50.5),
                                   (5, -24.5, 52), (11, -23, 57)], np.float32), 1.0), 0.5)  # smile
d = smax(d, -sphere(P, (-12.2, -25.0, 70.0), 1.3), 0.3)
d = smax(d, -sphere(P, (8.8, -25.0, 70.0), 1.3), 0.3)
# toe pads on the front of each foot
for sx in (-1, 1):
    cx = sx * 15
    for dx, dz in [(-3.5, 9.5), (0.0, 11.0), (3.5, 9.5), (0.0, 4.5)]:
        d = smax(d, -sphere(P, (cx + dx, -30.6, dz), 1.9), 0.3)

# ---------- light string (measured positions) ----------
strandA = np.array([(12, -17, 90), (22, -15, 86), (33, -8, 88), (38, -9, 80),
                    (31, -12, 70), (27, -15, 58), (25, -17, 49)], np.float32)
strandB = np.array([(-20, -14, 42), (-8, -18, 36), (6, -17, 27)], np.float32)
for pts in (strandA, strandB):
    d = smin(d, polyline(P, pts, 1.1), 1.2)
for c in [(12, -17, 90), (34, -5, 90), (25, -17, 46.5), (-8, -18, 35.5)]:
    d = smin(d, sphere(P, c, 2.6), 1.0)

# ---------- surface ----------
verts, faces, _, _ = measure.marching_cubes(d.astype(np.float32), level=0.0,
                                            spacing=(VOX, VOX, VOX))
verts += np.array([xs[0], ys[0], zs[0]], dtype=np.float32)
del d, P

mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
trimesh.repair.fix_normals(mesh)
trimesh.repair.fill_holes(mesh)
parts = mesh.split(only_watertight=False)
for p in parts:
    print("part vol=%.0f centre=%s" % (p.volume, np.round(p.centroid, 1)))
mesh = max(parts, key=lambda m: m.volume)
mesh.apply_translation([-(mesh.bounds[0][0] + mesh.bounds[1][0]) / 2,
                        -(mesh.bounds[0][1] + mesh.bounds[1][1]) / 2,
                        -mesh.bounds[0][2]])
mesh.apply_scale(0.75)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flockchen.stl")
mesh.export(out)
print("file:", out)
print("watertight:", mesh.is_watertight, "| volume_cm3:", round(mesh.volume / 1000, 1))
print("size_mm (x,y,z):", [round(float(v), 1) for v in mesh.extents])
print("parts:", len(parts))
