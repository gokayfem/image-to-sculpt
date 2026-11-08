"""End to end on a synthetic subject: wrong view angles, noisy normals, one view with a changed pose.

The fit must reject the bad view, recover the cameras, and end closer to the true shape than the hull, with a
held-out view (never fitted) that agrees.
"""
import json
import os
import shutil
import tempfile

import numpy as np

from i2s import fit, hull, synth
from i2s import maps as MP


def slow(fn):
    fn.slow = True
    return fn


@slow
def test_synthetic_figure_end_to_end():
    d = tempfile.mkdtemp(prefix="i2s_test_")
    try:
        case, truth = synth.make_case(os.path.join(d, "case"), "figure", size=(240, 320), n_views="minimal",
                                      bad_view=True, res=110, hull_res=70, truth_vox=1 / 110)
        case.cfg["settings"]["fit_scale"] = 1.0
        case.save()
        hull.build(case, log=lambda *a: None)
        roles = {n: v["role"] for n, v in case.cfg["views"].items()}
        assert roles["az-030_badpose"] == "rejected", roles
        for n, v in case.cfg["views"].items():
            if v["role"] == "rejected":
                continue
            t = truth["views"][n]["az"]
            assert abs(((v["params"]["az"] - t + 180) % 360) - 180) < 8, (n, v["params"], t)
        Z = np.load(case.p("synthetic_truth.npz"))
        Vt, Ft = Z["verts"].astype(float), Z["faces"].astype(np.int64)
        Vh, Fh = case.mesh()
        ch_hull = synth.chamfer(Vh, Fh, Vt, Ft)[0]
        for st in ("forms", "surface"):
            fit.run_stage(case, st, passes=3, log=lambda *a: None)
        fit.run_op(case, "cut", which="front", log=lambda *a: None)
        V, Fc = case.mesh()
        ch_fit = synth.chamfer(V, Fc, Vt, Ft)[0]
        res, _ = MP.evaluate(V, Fc, MP.load_views(case, 1.0))
        ho = res["holdout_mean"]
        assert ch_fit < ch_hull, (ch_fit, ch_hull)
        assert ho["iou"] > 0.85, ho
        assert res["front"]["iou"] > 0.95, res["front"]
        assert len(case.read_log()) >= 8
        return (f"surface distance hull {ch_hull * 1000:.1f} -> fit {ch_fit * 1000:.1f} /1000; held-out IoU "
                f"{ho['iou']:.3f}; photo IoU {res['front']['iou']:.3f}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
