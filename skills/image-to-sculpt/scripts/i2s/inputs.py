"""Front maps, generated or imported views, part masks, and text-to-image for idea-only inputs.

The front image fixes the world: its camera is the origin, MoGe-3 gives its focal length and metric depth,
Marigold gives its normals, BiRefNet (or the alpha channel, or a SAM-3 text prompt) gives its mask. Units are
then normalised so the subject's larger front extent is 1.0.
"""
import concurrent.futures as cf
import json
import os
import re
import threading

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import falio
from .case import normalise_input, safe_label

LOCK = threading.Lock()

# ----------------------------------------------------------------------------- decoding helpers


def read_ply(path):
    """minimal binary/ascii PLY vertex reader -> dict of float arrays (x, y, z, nx, ny, nz, ...)"""
    with open(path, "rb") as f:
        head, props, n, fmt = [], [], 0, "binary_little_endian"
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            head.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[2])
                cur = "vertex"
            elif line.startswith("element"):
                cur = line.split()[1]
            elif line.startswith("property") and cur == "vertex":
                t, name = line.split()[1], line.split()[2]
                props.append((name, {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
                                     "int": "i4", "int32": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}[t]))
            elif line == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            return {p[0]: data[:, i].astype(float) for i, p in enumerate(props)}
        dt = np.dtype([(p, ("<" if fmt.endswith("little_endian") else ">") + t) for p, t in props])
        arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return {p: arr[p].astype(float) for p, _ in props}


def mask_png(path, size):
    im = Image.open(path)
    a = np.asarray(im.convert("RGBA"))
    m = a[..., 3] if (a[..., 3] < 250).mean() > 0.01 else np.asarray(im.convert("L"))
    return np.asarray(Image.fromarray(m).resize(size, Image.BILINEAR)) > 127


def normals_png(path, size):
    a = np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR)).astype(np.float32)
    n = a / 127.5 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def largest_blob(m):
    lab, nl = ndi.label(m)
    if nl > 1:
        s = ndi.sum(np.ones_like(lab), lab, range(1, nl + 1))
        m = lab == 1 + int(np.argmax(s))
    return ndi.binary_fill_holes(m)


def sam_mask(case, path_png, prompt, out_png, raw):
    r = falio.call(case, "fal-ai/sam-3-1/image", {"image_url": falio.upload(path_png), "prompt": prompt},
                   f"sam-3 '{prompt}' on {os.path.basename(path_png)}", raw, LOCK)
    masks = r.get("masks") or ([r["image"]] if r.get("image") else [])
    if not masks:
        raise RuntimeError(f"SAM-3 found nothing for '{prompt}'")
    W, H = Image.open(path_png).size
    acc = np.zeros((H, W), bool)
    for k, m in enumerate(masks[:4]):
        p = out_png.replace(".png", f"_{k}.png")
        falio.download(m["url"], p)
        acc |= mask_png(p, (W, H))
    Image.fromarray(acc.astype(np.uint8) * 255).save(out_png)
    return acc


# ----------------------------------------------------------------------------- front


def prep(case, select=None, restyle=False, fov_deg=40.0):
    W, H = case.size
    front = case.p("ref", "front.png")
    raw = case.p("ref", "raw", mk=True)
    if restyle:
        orig = case.p("ref", "front_original.png")
        if not os.path.exists(orig):
            Image.open(front).save(orig)
        r = falio.call(case, "google/nano-banana-2.1/edit", {
            "prompt": f"Turn this into a clean 3D render of the same {case.subject}: the same pose, proportions, "
                      "silhouette and details, a solid sculpted form with clear shading, plain light grey seamless "
                      "studio background, soft studio lighting, whole subject in frame with margin, no text.",
            "image_urls": [falio.upload(orig)], "aspect_ratio": case.cfg["work"]["aspect"], "resolution": "1K",
            "num_images": 1, "output_format": "png"}, "restyle drawing -> 3D render",
            os.path.join(raw, "restyle.json"))
        falio.download(r["images"][0]["url"], os.path.join(raw, "restyle.png"))
        Image.open(os.path.join(raw, "restyle.png")).convert("RGB").resize((W, H), Image.LANCZOS).save(front)
    online = falio.have_key()
    out = {}
    jobs = {}
    if online:
        url = falio.upload(front)
        with cf.ThreadPoolExecutor(3) as ex:
            if select:
                jobs["mask"] = ex.submit(sam_mask, case, front, select, os.path.join(raw, "select_mask.png"),
                                         os.path.join(raw, "sam_select.json"))
            elif not os.path.exists(case.p("ref", "alpha_mask.png")):
                jobs["mask"] = ex.submit(falio.call, case, "fal-ai/birefnet/v2", {"image_url": url, "output_mask": True},
                                         "front mask", os.path.join(raw, "birefnet.json"), LOCK)
            jobs["moge"] = ex.submit(falio.call, case, "fal-ai/moge-3", {"image_url": url}, "front depth + camera",
                                     os.path.join(raw, "moge.json"), LOCK)
            jobs["normals"] = ex.submit(falio.call, case, "fal-ai/marigold-v2",
                                        {"image_url": url, "prediction_type": "normals",
                                         "image_size": {"width": W, "height": H}},
                                        "front normals", os.path.join(raw, "marigold.json"), LOCK)
            res = {k: j.result() for k, j in jobs.items()}
    else:
        res = {}
    # -- mask
    if os.path.exists(case.p("ref", "alpha_mask.png")) and not select:
        mask = mask_png(case.p("ref", "alpha_mask.png"), (W, H))
        out["mask_source"] = "alpha"
    elif select:
        mask = res["mask"]
        out["mask_source"] = f"sam-3 '{select}'"
    elif "mask" in res:
        falio.download(res["mask"]["mask_image"]["url"], os.path.join(raw, "birefnet_mask.png"))
        mask = mask_png(os.path.join(raw, "birefnet_mask.png"), (W, H))
        out["mask_source"] = "birefnet"
    else:
        raise SystemExit("no mask: give an image with a transparent background, or set FAL_KEY")
    mask = largest_blob(mask)
    # -- normals
    if "normals" in res:
        falio.download(res["normals"]["image"]["url"], os.path.join(raw, "marigold_normals.png"))
        ncam = normals_png(os.path.join(raw, "marigold_normals.png"), (W, H))
        Image.open(os.path.join(raw, "marigold_normals.png")).convert("RGB").resize((W, H)).save(
            case.p("ref", "front_normals.png"))
    else:
        ncam = np.zeros((H, W, 3), np.float32)
    # -- depth + intrinsics
    if "moge" in res:
        r = res["moge"]
        falio.download(r["point_cloud"]["url"], os.path.join(raw, "moge_points.ply"))
        Kn = np.array(r["intrinsics"], float)
        fx, fy, cx, cy = Kn[0, 0] * W, Kn[1, 1] * H, Kn[0, 2] * W, Kn[1, 2] * H
        pts = read_ply(os.path.join(raw, "moge_points.ply"))
        P = np.stack([pts["x"], pts["y"], pts["z"]], 1)
        u = -fx * P[:, 0] / P[:, 2] + cx
        v = fy * P[:, 1] / P[:, 2] + cy
        ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (P[:, 2] < 0)
        ok[ok] &= mask[vi[ok], ui[ok]]
        idx = vi[ok] * W + ui[ok]
        cnt = np.bincount(idx, minlength=W * H)
        dsum = np.bincount(idx, weights=-P[ok, 2], minlength=W * H)
        depth = np.where(cnt > 0, dsum / np.maximum(cnt, 1), np.nan).reshape(H, W)
        hole = mask & ~np.isfinite(depth)  # fill small gaps from the nearest measured pixel
        if hole.any() and np.isfinite(depth).any():
            _, (iy, ix) = ndi.distance_transform_edt(~np.isfinite(depth), return_indices=True)
            depth = np.where(hole, depth[iy, ix], depth)
        depth = np.where(mask, depth, np.nan)
        out.update(fov_x=r.get("fov_x"), fov_y=r.get("fov_y"), depth_source="moge-3")
    else:
        fy = fx = 0.5 * H / np.tan(np.radians(fov_deg) / 2)
        cx, cy = W / 2, H / 2
        depth = np.full((H, W), np.nan)
        out["depth_source"] = f"none (assumed vertical fov {fov_deg} deg)"
    return write_front(case, mask, depth, ncam, (fx, fy, cx, cy), out)


def write_front(case, mask, depth, ncam, K, info=None):
    """normalise units (the subject's larger front extent = 1.0), estimate the subject centre, save front.npz"""
    fx, fy, cx, cy = K
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if np.isfinite(depth).any():
        d = np.where(np.isfinite(depth[ys, xs]), depth[ys, xs], np.nanmedian(depth))
        X = (xs + 0.5 - cx) / fx * d
        Z = -(ys + 0.5 - cy) / fy * d
        ext = max(np.percentile(X, 99.5) - np.percentile(X, 0.5), np.percentile(Z, 99.5) - np.percentile(Z, 0.5))
        scale = 1.0 / ext
        depth = depth * scale
        dnear = float(np.nanpercentile(depth, 3))
        r_units = float(ndi.distance_transform_edt(mask).max()) * dnear / fx
    else:
        scale = 1.0
        D = fx / max(np.ptp(xs), np.ptp(ys))  # 1 unit spans the larger mask extent at the centre
        r_units = float(ndi.distance_transform_edt(mask).max()) * D / fx
        dnear = D - r_units
        depth = np.full((H, W), np.nan)
    xc = ((xs.min() + xs.max()) / 2 + 0.5 - cx) / fx * (dnear + r_units)
    zc = -((ys.min() + ys.max()) / 2 + 0.5 - cy) / fy * (dnear + r_units)
    center = [float(xc), float(dnear + r_units), float(zc)]
    np.savez_compressed(case.p("ref", "front.npz"), mask=mask, depth=depth.astype(np.float32),
                        ncam=ncam.astype(np.float32), K=np.array([fx, fy, cx, cy]),
                        has_depth=np.isfinite(depth).any(), has_normals=bool(np.abs(ncam).sum() > 0))
    Image.fromarray(mask.astype(np.uint8) * 255).save(case.p("ref", "front_mask.png"))
    case.cfg["front"] = {"K": [float(fx), float(fy), float(cx), float(cy)], "unit_scale": float(scale),
                         "center": center, **(info or {})}
    case._front = None
    case.save()
    case.status("prep")
    return case.cfg["front"]


# ----------------------------------------------------------------------------- views

ANGLE_SETS = {
    "minimal": [(45, 0), (-45, 0), (90, 0), (-90, 0), (180, 0)],
    "standard": [(45, 0), (-45, 0), (90, 0), (-90, 0), (180, 0), (-135, 0), (0, 30), (180, 30), (-45, 30)],
    "full": [(45, 0), (-45, 0), (90, 0), (-90, 0), (180, 0), (-135, 0), (0, 30), (180, 30), (-45, 30), (45, 30),
             (0, -20), (65, 0), (-65, 0)],
}
HOLDOUT = (135, 0)


def view_name(az, el):
    az = ((az + 180) % 360) - 180 if az != 180 else 180
    n = "az180" if az in (180, -180) else f"az{az:+04d}"
    return n + ("" if el == 0 else ("_hi" if el == 30 else "_lo" if el == -20 else f"_el{el:+03d}"))


def describe(az, el, subject):
    s = subject or "subject"
    a = abs(az)
    side = "RIGHT" if az > 0 else "LEFT"
    if a < 5:
        d = "from the front, like the input"
    elif a >= 175:
        d = f"from directly behind: the back view of the {s}"
    else:
        d = (f"the camera has orbited {a} degrees around the {s} to the {side}, so we now see more of the side that "
             f"was on the {side.lower()} of the original image")
        if 80 <= a <= 100:
            d += " (a side profile view)"
        elif a > 100:
            d += " (a back three-quarter view)"
    if el > 0:
        d += f", and the camera is raised, looking down at the {s} at about {el} degrees"
    elif el < 0:
        d += f", and the camera is lowered, looking up at the {s} at about {-el} degrees"
    return d


PROMPT = ("Show the exact same {s} from a different camera angle: {d}. Keep everything else identical to the input "
          "image: the same {s}, same proportions, same pose, same shapes and details, same colours and materials. "
          "Same camera distance and lens, so the {s} has about the same size in the frame. The whole {s} in frame "
          "with margin, plain light grey seamless studio background, soft even studio lighting, no ground shadow, "
          "a clean 3D turnaround render, no text, no extra objects, no props.")


def _helpers(case, name):
    W, H = case.size
    img = case.p("views", f"{name}.png")
    raw = case.p("views", "raw", mk=True)
    url = falio.upload(img)
    with cf.ThreadPoolExecutor(2) as ex:
        jm = ex.submit(falio.call, case, "fal-ai/birefnet/v2", {"image_url": url, "output_mask": True},
                       f"mask {name}", os.path.join(raw, f"{name}_birefnet.json"), LOCK)
        jn = ex.submit(falio.call, case, "fal-ai/marigold-v2", {"image_url": url, "prediction_type": "normals",
                                                                "image_size": {"width": W, "height": H}},
                       f"normals {name}", os.path.join(raw, f"{name}_marigold.json"), LOCK)
        rm, rn = jm.result(), jn.result()
    falio.download(rm["mask_image"]["url"], os.path.join(raw, f"{name}_mask.png"))
    m = largest_blob(mask_png(os.path.join(raw, f"{name}_mask.png"), (W, H)))
    Image.fromarray(m.astype(np.uint8) * 255).save(case.p("views", f"{name}_mask.png"))
    falio.download(rn["image"]["url"], os.path.join(raw, f"{name}_normals.png"))
    Image.open(os.path.join(raw, f"{name}_normals.png")).convert("RGB").resize((W, H), Image.BILINEAR).save(
        case.p("views", f"{name}_normals.png"))


def _gen_one(case, url, az, el, role):
    name = view_name(az, el)
    W, H = case.size
    raw = case.p("views", "raw", mk=True)
    prompt = PROMPT.format(s=case.subject, d=describe(az, el, case.subject))
    r = falio.call(case, "google/nano-banana-2.1/edit", {
        "prompt": prompt, "image_urls": [url], "aspect_ratio": case.cfg["work"]["aspect"], "resolution": "1K",
        "num_images": 1, "output_format": "png", "limit_generations": False},
        f"view {name}", os.path.join(raw, f"{name}.json"), LOCK)
    falio.download(r["images"][0]["url"], os.path.join(raw, f"{name}.png"))
    Image.open(os.path.join(raw, f"{name}.png")).convert("RGB").resize((W, H), Image.LANCZOS).save(
        case.p("views", f"{name}.png"))
    _helpers(case, name)
    with LOCK:
        case.cfg.setdefault("views", {})[name] = {"requested": {"az": az, "el": el}, "role": role,
                                                  "source": "google/nano-banana-2.1/edit", "prompt": prompt}
        case.save()
    return name


def generate(case, angle_set="standard", angles=None, holdout=True, workers=4):
    todo = [(az, el, "train") for az, el in (angles or ANGLE_SETS[angle_set])]
    if holdout and not any(v.get("role") == "holdout" for v in case.cfg.get("views", {}).values()):
        todo.append((*HOLDOUT, "holdout"))
    todo = [t for t in todo if view_name(t[0], t[1]) not in case.cfg.get("views", {})]
    if not todo:
        return []
    url = falio.upload(case.p("ref", "front.png"))
    done = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_gen_one, case, url, az, el, role): (az, el) for az, el, role in todo}
        for f in cf.as_completed(futs):
            try:
                done.append(f.result())
                print("view", done[-1], flush=True)
            except falio.BudgetError as e:
                print("stopped:", e, flush=True)
            except Exception as e:  # one failed generation never stops the others
                print("view", futs[f], "failed:", e, flush=True)
    case.status("views")
    return done


def parse_angle(name):
    m = re.search(r"az([+-]?\d+)", name)
    e = re.search(r"el([+-]?\d+)", name)
    az = int(m.group(1)) if m else None
    el = int(e.group(1)) if e else (30 if "_hi" in name else -20 if "_lo" in name else 0)
    return az, el


def import_views(case, folder, holdout=None):
    """bring your own views: other photos or renders of the same subject. Angles from the file names (az+090,
    az-045_el+30, _hi/_lo) or folder/views.json {file: {az, el}}; unknown angles are found by registration."""
    W, H = case.size
    meta = {}
    if os.path.exists(os.path.join(folder, "views.json")):
        meta = json.load(open(os.path.join(folder, "views.json")))
    names = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) or f.endswith(("_mask.png", "_normals.png")):
            continue
        stem = os.path.splitext(f)[0]
        az, el = (meta[f]["az"], meta[f].get("el", 0)) if f in meta else parse_angle(stem)
        name = stem if stem.startswith("az") else f"imp_{stem}"
        img, alpha, _, _ = normalise_input(os.path.join(folder, f), max(W, H))
        img.resize((W, H), Image.LANCZOS).save(case.p("views", f"{name}.png"))
        mfile = os.path.join(folder, stem + "_mask.png")
        nfile = os.path.join(folder, stem + "_normals.png")
        have_mask = os.path.exists(mfile) or alpha is not None
        if os.path.exists(mfile):
            m = mask_png(mfile, (W, H))
        elif alpha is not None:
            m = np.asarray(Image.fromarray(alpha.astype(np.uint8) * 255).resize((W, H))) > 127
        if have_mask:
            Image.fromarray(largest_blob(m).astype(np.uint8) * 255).save(case.p("views", f"{name}_mask.png"))
        if os.path.exists(nfile):
            Image.open(nfile).convert("RGB").resize((W, H)).save(case.p("views", f"{name}_normals.png"))
        if (not have_mask or not os.path.exists(nfile)) and falio.have_key():
            _helpers(case, name)
        case.cfg.setdefault("views", {})[name] = {
            "requested": {"az": az, "el": el}, "source": f"import {os.path.basename(folder)}/{f}",
            "role": "holdout" if holdout and (name == holdout or f == holdout) else "train"}
        names.append(name)
    case.save()
    case.status("views")
    return names


# ----------------------------------------------------------------------------- parts, idea


def parts(case, prompts=(), boxes=()):
    """front-image region masks: SAM-3 text prompts ('hands', 'face') or boxes 'label:x0,y0,x1,y1' (fractions of
    the working image or pixels)"""
    W, H = case.size
    mask = case.front()["mask"]
    out = {}
    for p in prompts:
        label = safe_label(p)
        m = sam_mask(case, case.p("ref", "front.png"), p, case.p("parts", f"{label}.png", mk=True),
                     case.p("parts", f"{label}_sam.json"))
        out[label] = m & ndi.binary_dilation(mask, iterations=2)
    for b in boxes:
        label, coords = b.split(":", 1)
        label = safe_label(label)
        x0, y0, x1, y1 = [float(c) for c in coords.split(",")]
        if max(x0, y0, x1, y1) <= 1.0:
            x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
        m = np.zeros((H, W), bool)
        m[int(y0):int(y1), int(x0):int(x1)] = True
        out[label] = m & mask
    for label, m in out.items():
        Image.fromarray(m.astype(np.uint8) * 255).save(case.p("parts", f"{label}.png", mk=True))
    return {k: int(v.sum()) for k, v in out.items()}


def imagine(prompt, out_png, aspect="3:4"):
    """an input image for an idea-only request: full subject, plain background, readable shading"""
    import fal_client
    full = (f"{prompt}. The whole subject in frame with margin, clear readable forms, soft even lighting, plain light "
            "grey seamless background, no ground shadow, no text, no other objects.")
    r = fal_client.subscribe("fal-ai/nano-banana-pro", arguments={"prompt": full, "aspect_ratio": aspect,
                                                                  "num_images": 1, "output_format": "png"})
    falio.download(r["images"][0]["url"], out_png)
    json.dump({"endpoint": "fal-ai/nano-banana-pro", "prompt": full, "est_usd": falio.PRICE["fal-ai/nano-banana-pro"]},
              open(os.path.splitext(out_png)[0] + ".json", "w"), indent=1)
    return out_png
