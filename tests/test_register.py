"""Registration recovers wrong angles; symmetry detection finds the plane; screening catches a changed pose."""
import numpy as np

from i2s import geom as G
from i2s import hull as HL
from i2s import register as RG
from i2s import synth

C = np.array([0.0, 2.3, 0.0])
W, H = 240, 320
K = (0.5 * H / np.tan(np.radians(18)),) * 2 + (W / 2, H / 2)
_CACHE = {}


def _mesh(subject, yaw=0.0, **kw):
    key = (subject, yaw, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = synth.truth_mesh(synth.SUBJECTS[subject], yaw, C, vox=1 / 90, **kw)
    return _CACHE[key]


def _view(V, Fc, p):
    Rm = G.raster(V, Fc, G.make_cam(p, C, K), W, H, VN=G.vertex_normals(V, Fc))
    return Rm["mask"], Rm["ncam"]


def test_recovers_a_view_far_from_the_requested_angle():
    V, Fc = _mesh("creature", 15.0)
    true = {"az": 72.0, "el": 9.0, "s": 1.03, "du": 4.0, "dv": -3.0}
    mask, ncam = _view(V, Fc, true)
    p, sc, io = RG.register(V, Fc, mask, ncam, C, K, {"az": 30, "el": 0})  # asked for 30, got 72
    p, _ = RG.refine(V, Fc, mask, ncam, C, K, p)
    daz = abs(((p["az"] - true["az"] + 180) % 360) - 180)
    assert daz < 3 and abs(p["el"] - true["el"]) < 3, p
    return f"az {p['az']:.1f} (true 72), el {p['el']:.1f} (true 9)"


def test_back_view_is_not_confused_with_the_front():
    V, Fc = _mesh("creature", 15.0)
    true = {"az": 172.0, "el": 0.0}
    mask, ncam = _view(V, Fc, true)
    p, _, _ = RG.register(V, Fc, mask, ncam, C, K, {"az": 180, "el": 0})
    daz = abs(((p["az"] - 172 + 180) % 360) - 180)
    assert daz < 6, p
    return f"az {p['az']:.1f} (true 172)"


def test_symmetry_plane_is_found_at_any_yaw():
    V, Fc = _mesh("figure", 25.0)
    lo = V.min(0) - 0.05
    vox = 1 / 80
    n = np.ceil((V.max(0) + 0.05 - lo) / vox).astype(int)
    g = lo + (np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3) + 0.5) * vox
    occ = (synth.figure((g - C) @ G.rz(np.radians(-25)).T) < 0).reshape(n)
    sym = HL.symmetry(occ, lo, vox, C)
    yaw = sym["yaw_deg"]
    assert sym["symmetric"] and min(abs(yaw - 25), abs(yaw + 155), abs(yaw - 205)) < 2.5, sym
    occ2 = (synth.creature((g - C)) < 0).reshape(n)
    sym2 = HL.symmetry(occ2, lo, vox, C)
    return f"figure yaw {yaw:.1f} (true 25) IoU {sym['iou']:.3f}; creature IoU {sym2['iou']:.3f}"


def test_changed_pose_scores_low_against_the_true_shape():
    """the screening signal: a view of a different pose cannot reach the IoU of a consistent view"""
    V, Fc = _mesh("figure", 0.0)
    Vb, Fb = _mesh("figure", 0.0, arms_up=True)
    good, gn = _view(V, Fc, {"az": -30})
    bad, bn = _view(Vb, Fb, {"az": -30})
    _, info_g = RG.refine(V, Fc, good, gn, C, K, RG.register(V, Fc, good, gn, C, K, {"az": -30})[0])
    _, info_b = RG.refine(V, Fc, bad, bn, C, K, RG.register(V, Fc, bad, bn, C, K, {"az": -30})[0])
    io_g, io_b = info_g["after"][1], info_b["after"][1]
    assert io_g > 0.95 and io_b < 0.8, (io_g, io_b)
    return f"consistent view IoU {io_g:.2f}, changed pose IoU {io_b:.2f}"


def test_symmetric_subject_back_view_from_above_stays_a_back_view():
    """(az, el) and (180 - az, -el) outline the same on a symmetric subject; the requested angle breaks the tie"""
    V, Fc = _mesh("figure", 0.0)
    true = {"az": 160.0, "el": 22.0}
    mask, ncam = _view(V, Fc, true)
    p, _, _ = RG.register(V, Fc, mask, ncam, C, K, {"az": 180, "el": 30})
    daz = abs(((p["az"] - 160 + 180) % 360) - 180)
    assert daz < 8 and p["el"] > 10, p
    return f"az {p['az']:.1f} el {p['el']:.1f} (true 160, 22)"
