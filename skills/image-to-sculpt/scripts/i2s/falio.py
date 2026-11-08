"""Every paid call goes through `call`: budget check, request id written to case.json BEFORE waiting, raw result
saved next to the files it produced. FAL_KEY is read from the environment by fal_client and never printed.

Endpoints (fal.ai):
  google/nano-banana-2.1/edit   new views of the same subject; drawing -> 3D render restyle
  fal-ai/nano-banana-pro        an input image from a text idea
  fal-ai/birefnet/v2            subject mask
  fal-ai/sam-3-1/image          a mask from a text prompt (pick one subject in a busy image; part masks)
  fal-ai/moge-3                 metric point cloud + intrinsics of the photo (depth and focal length)
  fal-ai/marigold-v2            normal maps (camera frame), the main shape signal
"""
import json
import os
import time
import urllib.parse
import urllib.request

# estimated USD per call, for the budget guard (check the fal dashboard for exact billing)
PRICE = {"google/nano-banana-2.1/edit": 0.15, "fal-ai/nano-banana-pro": 0.15, "fal-ai/birefnet/v2": 0.005,
         "fal-ai/sam-3-1/image": 0.01, "fal-ai/moge-3": 0.02, "fal-ai/marigold-v2": 0.03}


class BudgetError(RuntimeError):
    pass


def have_key():
    return bool(os.environ.get("FAL_KEY") or os.environ.get("FAL_KEY_ID"))


def _client():
    try:
        import fal_client
    except ImportError as e:
        raise SystemExit("fal-client is not installed: pip install fal-client") from e
    if not have_key():
        raise SystemExit("FAL_KEY is not set: export it (it is never printed or written to files)")
    return fal_client


def upload(path):
    return _client().upload_file(path)


def download(url, path, tries=3):
    """fetch a result file; only https links are followed (urllib would also open file:// and ftp://)"""
    if urllib.parse.urlparse(str(url)).scheme != "https":
        raise ValueError(f"refusing to download a non-https link: {str(url)[:80]}")
    for k in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def call(case, endpoint, args, purpose, raw_json=None, lock=None):
    """submit, record the request id, then wait for the result (never a blind resubmit)"""
    fc = _client()
    est = PRICE.get(endpoint, 0.05)
    if lock:
        lock.acquire()
    try:
        budget = float(case.cfg.get("settings", {}).get("budget_usd", 4.0))
        if case.fal_spent() + est > budget + 1e-9:
            raise BudgetError(f"fal budget ${budget:.2f} would be exceeded by {purpose} (${case.fal_spent():.2f} "
                              f"spent, est. ${est:.2f}). Raise settings.budget_usd in case.json if intended.")
        h = fc.submit(endpoint, arguments=args)
        entry = {"endpoint": endpoint, "request_id": h.request_id, "purpose": purpose, "est_usd": est,
                 "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
        case.cfg.setdefault("fal", []).append(entry)
        case.save()
    finally:
        if lock:
            lock.release()
    r = h.get()
    if raw_json:
        json.dump({"endpoint": endpoint, "request_id": entry["request_id"], "arguments":
                   {k: v for k, v in args.items() if k not in ("image_url", "image_urls")}, "result": r},
                  open(raw_json, "w"), indent=1)
    return r
