"""REPORT.md, the per-pass chart and a static review page.

Every number in the report is measured the same way for the hull and for the final sculpt, with the same cameras.
The held-out view's camera was frozen before any fitting, and the view never moved the surface.
"""
import glob
import html
import json
import os

import numpy as np

from . import geom as G
from . import maps as MP
from . import register as RG

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"


def _mesh(path):
    z = np.load(path)
    return z["verts"].astype(float), z["faces"].astype(np.int64)


def hull_step(case):
    for p in sorted(glob.glob(os.path.join(case.steps_dir(), "*.npz"))):
        z = np.load(p)
        if str(z["op"]) == "remesh":
            return p
    return None


def chart(case):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    log = [e for e in case.read_log() if e.get("op") in ("fit", "cut", "smooth", "finish") and "metrics" in e]
    if len(log) < 2:
        return None
    k = list(range(1, len(log) + 1))

    def series(group, key):
        return [e["metrics"].get(group, {}).get(key, np.nan) for e in log]
    panels = [("silhouette IoU (higher is better)", "iou"), ("normal angle, median ° (lower is better)", "normal_med"),
              ("roughness vs reference (1 = as smooth)", "rough_ratio"), ("outline distance, px (lower is better)", "sil_px")]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.0), facecolor=SURF)
    stages, prev = [], None
    for i, e in enumerate(log):
        s = e.get("stage") or e.get("op")
        if s != prev:
            stages.append((s, i + 1))
            prev = s
    for ax, (title, key) in zip(axs.ravel(), panels):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for _, start in stages[1:]:
            ax.axvline(start - 0.5, color=GRID, lw=1.2)
        for group, label, col in (("train_mean", "views used to fit", S1), ("holdout_mean", "held-out view", S2)):
            y = series(group, key)
            if np.all(np.isnan(y)):
                continue
            ax.plot(k, y, color=col, lw=2, label=label)
            ax.annotate(f"{y[-1]:.3g}", (k[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=9)
        if key == "rough_ratio":
            ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("recorded fit step")
        ax.legend(frameon=False, fontsize=9, loc="best")
    ymax = axs[0, 0].get_ylim()[1]
    last = -99
    for name, start in stages:
        if start - last >= 5:
            axs[0, 0].text(start, ymax, " " + str(name), color=INK2, fontsize=8, va="top")
            last = start
    fig.suptitle("map fit, step by step (held-out view: fixed camera, never used to fit)", x=0.01, ha="left",
                 color=INK, fontsize=13)
    fig.tight_layout()
    out = case.p("checks", "fit_passes.png")
    fig.savefig(out, dpi=120, facecolor=SURF)
    plt.close(fig)
    return out


def holdout_bestfit(case, V, Fc, views):
    """re-register the held-out view against the final shape (measures shape, not camera)"""
    out = {}
    for vw in views:
        if vw.role != "holdout":
            continue
        p, _ = RG.refine(V, Fc, vw.mask, vw.ncam, case.center, case.K, case.cfg["views"][vw.name]["params"], f=2)
        vw.set_cam(G.scale_cam(G.make_cam(p, case.center, case.K), vw.scale))
        res, _ = MP.evaluate(V, Fc, [vw])
        out[vw.name] = res[vw.name]
        vw.set_cam(case.cam(vw.name, vw.scale))
    return out


def acceptance(case, final):
    """final metrics against the case's acceptance.json -> [(check, target, value, ok)]"""
    p = case.p("acceptance.json")
    if not os.path.exists(p) or not final:
        return []
    A = json.load(open(p))
    out = []
    groups = {"photo": final.get("front", {}), "train_mean": final.get("train_mean", {}),
              "holdout": final.get("holdout_mean", {})}
    keys = {"iou_min": ("iou", ">="), "sil_px_max": ("sil_px", "<="), "normal_med_max_deg": ("normal_med", "<="),
            "depth_per_mille_max": ("depth_mm", "<=")}
    for g, m in groups.items():
        for k, t in (A.get(g) or {}).items():
            if k in keys and keys[k][0] in m:
                key, op = keys[k]
                v = m[key]
                out.append((f"{g} {key}", f"{op} {t}", v, v >= t if op == ">=" else v <= t))
            elif k == "rough_ratio_range" and "rough_ratio" in m:
                v = m["rough_ratio"]
                out.append((f"{g} rough_ratio", f"{t[0]}–{t[1]}", v, t[0] <= v <= t[1]))
    return out


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def build(case, log=print):
    views = MP.load_views(case, 0.5)
    hp = hull_step(case)
    fp = case.p("build", "final_mesh.npz")
    if not os.path.exists(fp):
        fp = case.p("build", "mesh.npz")
    rows = {}
    for tag, p in (("hull", hp), ("final", fp)):
        if p:
            V, Fc = _mesh(p)
            rows[tag], _ = MP.evaluate(V, Fc, views)
    V, Fc = _mesh(fp)
    hofit = holdout_bestfit(case, V, Fc, views)
    json.dump({"hull": rows.get("hull"), "final": rows.get("final"), "holdout_bestfit": hofit},
              open(case.p("checks", "metrics_report.json"), "w"), indent=1)
    ch = chart(case)
    cfg = case.cfg
    logs = case.read_log()
    ops = {}
    for e in logs:
        if "step" in e:
            ops[e["op"]] = ops.get(e["op"], 0) + 1
    L = []
    subj = cfg.get("subject") or "subject"
    L.append(f"# image-to-sculpt report: {subj}\n")
    L.append(f"Input `{cfg['input']['file']}` ({cfg['input']['size'][0]}×{cfg['input']['size'][1]}, kind "
             f"{cfg['input']['kind']}), sha256 `{cfg['input']['sha256'][:16]}…`. Working size "
             f"{cfg['work']['size'][0]}×{cfg['work']['size'][1]} (padded to {cfg['work']['aspect']}).\n")
    fr = cfg.get("front", {})
    L.append(f"Front camera: focal {fr.get('K', [0])[0]:.0f} px, mask from {fr.get('mask_source', '?')}, depth from "
             f"{fr.get('depth_source', '?')}. Units: the subject's larger front extent = 1.\n")
    L.append("![hero](hero.png)\n")
    L.append("## Build\n")
    L.append(f"{sum(ops.values())} recorded steps: " + ", ".join(f"{v} {k}" for k, v in ops.items()) +
             ". Every step is a mesh in `build/steps/`, replayed in Blender (`blender/sculpt.blend`) and shown in "
             "`timelapse.mp4`.\n")
    sym = cfg.get("symmetry", {})
    if sym:
        L.append(f"Symmetry: mirror IoU of the hull {fmt(sym.get('iou'))} at yaw {fmt(sym.get('yaw_deg'), 1)}° → "
                 f"{'used' if sym.get('used') else 'not used'}.\n")
    L.append("## Views\n")
    L.append("| view | role | asked az / el | registered az / el | IoU vs hull | source |\n|---|---|---|---|---|---|")
    for n, v in cfg.get("views", {}).items():
        q, p = v.get("requested", {}), v.get("params", {})
        L.append(f"| {n} | {v.get('role')}{' (' + v['reason'] + ')' if v.get('reason') else ''} | "
                 f"{q.get('az')} / {q.get('el')} | {fmt(p.get('az'), 1)} / {fmt(p.get('el'), 1)} | "
                 f"{fmt(v.get('iou_hull'))} | {v.get('source', '')} |")
    L.append("\n![views](views.png)\n")
    L.append("## Measured\n")
    L.append("Same cameras for both rows. *train* = mean over the views used to fit (photo included); "
             "*held-out* = the view never used to fit, camera frozen before fitting; *best-fit* = the held-out view "
             "re-registered to the final shape.\n")
    L.append("| | IoU | outline px | normals median ° | normals < 15° | roughness × | photo depth /1000 |\n"
             "|---|---|---|---|---|---|---|")
    for tag in ("hull", "final"):
        r = rows.get(tag) or {}
        for g, nm in (("train_mean", "train"), ("holdout_mean", "held-out")):
            m = r.get(g)
            if not m:
                continue
            L.append(f"| {tag} · {nm} | {fmt(m.get('iou'))} | {fmt(m.get('sil_px'), 1)} | {fmt(m.get('normal_med'), 1)} | "
                     f"{fmt(m.get('normal_p15'), 0)}% | {fmt(m.get('rough_ratio'), 2)} | "
                     f"{fmt((r.get('front') or {}).get('depth_mm'), 1) if nm == 'train' else '–'} |")
    for n, m in hofit.items():
        L.append(f"| final · held-out best-fit | {fmt(m.get('iou'))} | {fmt(m.get('sil_px'), 1)} | "
                 f"{fmt(m.get('normal_med'), 1)} | {fmt(m.get('normal_p15'), 0)}% | {fmt(m.get('rough_ratio'), 2)} | – |")
    L.append("")
    acc = acceptance(case, rows.get("final") or {})
    if acc:
        L.append("## Acceptance\n")
        L.append("Targets from `acceptance.json`, set before fitting and never relaxed.\n")
        L.append("| check | target | measured | |\n|---|---|---|---|")
        for name, target, val, ok in acc:
            L.append(f"| {name} | {target} | {fmt(val, 3)} | {'pass' if ok else '**fail**'} |")
        L.append("")
    if ch:
        L.append(f"![per-step chart](../checks/{os.path.basename(ch)})\n")
    if os.path.exists(case.p("checks", "sheet_final.png")):
        L.append("![map sheet](../checks/sheet_final.png)\n")
    L.append("## fal calls\n")
    L.append(f"{len(cfg.get('fal', []))} calls, estimated ${case.fal_spent():.2f} (request ids in `case.json`).\n")
    L.append("## What this is and is not\n")
    L.append("- The photo fixes what the camera sees: silhouette, normals, metric depth. Everything the photo does "
             "not see comes from generated views, which are the generator's guesses, made consistent by "
             "registration and screening. The held-out view is generated too: it measures consistency, not truth.\n"
             "- The sculpt is a watertight mesh fitted in numpy (normal integration on a signed distance grid); "
             "Blender replays and renders the recorded steps and holds the final file.\n")
    if os.path.exists(case.p("synthetic_truth.npz")):
        from .synth import chamfer
        Z = np.load(case.p("synthetic_truth.npz"))
        out = []
        for tag, p in (("hull", hp), ("final", fp)):
            if p:
                Va, Fa = _mesh(p)
                out.append((tag, chamfer(Va, Fa, Z["verts"].astype(float), Z["faces"].astype(np.int64))))
        L.append("## Ground truth (synthetic case)\n")
        L.append("| | mean surface distance | 95th percentile |\n|---|---|---|")
        for tag, (m, p95) in out:
            L.append(f"| {tag} | {m * 1000:.1f}/1000 | {p95 * 1000:.1f}/1000 |")
        L.append("")
    md = "\n".join(L)
    open(case.p("out", "REPORT.md", mk=True), "w").write(md)
    page(case, md)
    case.status("report")
    log(f"  report: {case.p('out', 'REPORT.md')}")
    return rows


def page(case, md):
    """a static review page: the film, the sheets, the report"""
    body = []
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("!["):
            src = line[line.index("(") + 1:line.rindex(")")]
            body.append(f'<img src="{html.escape(src)}" alt="">')
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    out, rows = [], []
    for b in body:
        if b.startswith("<tr>"):
            rows.append(b)
            continue
        if rows:
            out.append("<table>" + "".join(rows) + "</table>")
            rows = []
        out.append(b)
    if rows:
        out.append("<table>" + "".join(rows) + "</table>")
    vid = '<video src="timelapse.mp4" controls autoplay muted loop playsinline></video>' if os.path.exists(
        case.p("out", "timelapse.mp4")) else ""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sculpt review</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--ink:#ececec;--ink2:#a9a9a6;--line:#2c2c30}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}}
img,video{{max-width:100%;border-radius:6px;margin:8px 0}} table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}
td{{border-bottom:1px solid var(--line);padding:4px 8px;white-space:nowrap}} tr:first-child td{{color:var(--ink2)}} p{{color:var(--ink2)}}
</style></head><body>{vid}{''.join(out)}</body></html>"""
    open(case.p("out", "index.html"), "w").write(doc)
