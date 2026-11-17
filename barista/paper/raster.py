"""Digitise bitmap figures (scanned papers, PNG/JPG/TIFF figures, raster images inside PDFs).

Pipeline: axes frame from long dark lines (morphology) → tick labels by OCR (tesseract)
→ linear/log calibration (also for 10^n labels that OCR returns as '10-3', '103', '10°')
→ series by colour clustering in CIELAB → markers (connected components) or lines
(column-wise tracing).  Black data are recovered after removing axes, grid lines and text.

Accuracy depends on resolution; render PDFs at ≥ 300 dpi (default 400).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .vector import Axis, Plot, Series, _calibrate


def _have_tesseract():
    return shutil.which("tesseract") is not None


def _ocr(img, psm=11):
    """Run tesseract on a numpy RGB/gray image; returns list of words with boxes."""
    import cv2
    if not _have_tesseract():
        return []
    tmp = tempfile.mkdtemp(prefix="barista_ocr_")
    try:
        p = os.path.join(tmp, "a.png")
        cv2.imwrite(p, img)
        r = subprocess.run(["tesseract", p, "stdout", "--psm", str(psm), "-c",
                            "tessedit_char_whitelist=0123456789.-−eE+x×^()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/µ% ",
                            "tsv"], capture_output=True, text=True, timeout=60)
        rows = []
        for line in r.stdout.splitlines()[1:]:
            f = line.split("\t")
            if len(f) == 12 and f[11].strip():
                rows.append({"text": f[11].strip(), "x": int(f[6]), "y": int(f[7]), "w": int(f[8]), "h": int(f[9]),
                             "conf": float(f[10])})
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _frame(gray):
    import cv2
    h, w = gray.shape
    dark = (gray < 110).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 5, 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 5, 20)))
    H = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    V = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    rows = np.where(H.sum(axis=1) > w * 0.25)[0]
    cols = np.where(V.sum(axis=0) > h * 0.25)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    def groups(idx):
        g, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= 2:
                cur.append(i)
            else:
                g.append(cur); cur = [i]
        g.append(cur)
        return [int(np.mean(x)) for x in g]

    rg, cg = groups(rows), groups(cols)
    bottom = max(rg); left = min(cg)
    top = min(rg) if len(rg) > 1 else int(np.where(V[:, left] > 0)[0].min())
    right = max(cg) if len(cg) > 1 else int(np.where(H[bottom, :] > 0)[0].max())
    if right - left < 40 or bottom - top < 40:
        return None
    return left, top, right, bottom


def _parse_label(t):
    s = t.replace("−", "-").replace("O", "0").replace("o", "0").replace("°", "0").replace(",", ".")
    s = re.sub(r"[^\d.\-eE+x×^]", "", s)
    cands = []
    try:
        cands.append(("plain", float(s)))
    except ValueError:
        pass
    m = re.fullmatch(r"10\^?(-?\d{1,2})", s)
    if m:
        cands.append(("pow", 10.0 ** int(m.group(1))))
    m = re.fullmatch(r"(\d(?:\.\d+)?)[eE]([-+]?\d+)", s)
    if m:
        cands.append(("plain", float(m.group(1)) * 10 ** int(m.group(2))))
    return cands


def _ocr_digit(glyph):
    """OCR one isolated glyph (binary, dark=1) as a digit."""
    import cv2
    g = (255 * (1 - glyph)).astype(np.uint8)
    h = g.shape[0]
    f = max(1, int(round(60 / max(h, 1))))
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    g = cv2.copyMakeBorder(g, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    tmp = tempfile.mkdtemp(prefix="barista_d_")
    try:
        pth = os.path.join(tmp, "d.png")
        cv2.imwrite(pth, g)
        r = subprocess.run(["tesseract", pth, "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=0123456789"],
                           capture_output=True, text=True, timeout=30)
        t = re.sub(r"\D", "", r.stdout)
        return t[:1] if t else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _sup_exponent(img, wd):
    """Read n in a '10^n' label by glyph segmentation: superscript glyphs are the small, raised
    components to the right of the '1' and '0'; a flat component is a minus sign."""
    import cv2
    H, W = img.shape[:2]
    x0 = int(wd["x"] - 2); x1 = int(wd["x"] + wd["w"] + 3)
    y0 = int(wd["y"] - 0.5 * wd["h"]); y1 = int(wd["y"] + wd["h"] + 2)
    crop = img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    comps = [(st[i, 0], st[i, 1], st[i, 2], st[i, 3], i) for i in range(1, n) if st[i, 4] >= 2]
    if not comps:
        return None
    hmax = max(c[3] for c in comps)
    base = [c for c in comps if c[3] >= 0.75 * hmax]
    if len(base) < 2:
        return None
    base.sort(key=lambda c: c[0])
    base_bottom = max(c[1] + c[3] for c in base[:2]); base_mid = np.mean([c[1] + c[3] / 2 for c in base[:2]])
    xr = base[1][0] + base[1][2] - 1
    sup = [c for c in comps if c[0] >= xr - 1 and c[1] + c[3] / 2 < base_mid and c not in base[:2]]
    if not sup:
        return None
    sup.sort(key=lambda c: c[0])
    sign, digits = 1, ""
    hs = max(c[3] for c in sup)
    for (x, y, w, h, i) in sup:
        if w >= 1.8 * h and h <= 0.45 * hs:          # minus sign
            sign = -1
            continue
        d = _ocr_digit((lab[y:y + h, x:x + w] == i).astype(np.uint8))
        if d is None:
            return None
        digits += d
    return sign * int(digits) if digits else None


def _decades(img, words, axis):
    """Calibrate a logarithmic axis labelled 10^n: consecutive decades + OCR'd exponents."""
    cand = [w for w in words if re.match(r"^1[0O9oQ]", w["text"].replace(" ", ""))]
    if len(cand) < 2:
        return None
    key = (lambda w: w["x"] + w["w"] / 2) if axis == "x" else (lambda w: w["y"] + w["h"] / 2)
    cand.sort(key=key)
    pix = np.array([key(w) for w in cand])
    guesses = []
    for i, w in enumerate(cand):
        e = _sup_exponent(img, w) if img is not None else None
        if e is None:
            rest = re.sub(r"[^\d\-]", "", w["text"].replace("−", "-"))[2:]
            try:
                e = int(rest)
            except ValueError:
                e = None
        guesses.append(e)
    step = 1 if axis == "x" else -1
    offs = [g - step * i for i, g in enumerate(guesses) if g is not None]
    if not offs:
        return None
    vals, cnt = np.unique(offs, return_counts=True)
    off = int(vals[np.argmax(cnt)])
    exps = np.array([off + step * i for i in range(len(cand))], float)
    A = np.vstack([pix, np.ones_like(pix)]).T
    coef, *_ = np.linalg.lstsq(A, exps, rcond=None)
    res = float(np.max(np.abs(A @ coef - exps)) / max(np.ptp(exps), 1e-9))
    if res > 0.05:
        return None
    return Axis("log", float(coef[0]), float(coef[1]), ticks=list(zip(pix.tolist(), (10 ** exps).tolist())),
                residual=res)


def _calibrate_ocr(words, axis, frame, img=None):
    left, top, right, bottom = frame
    dec = _decades(img, words, axis)
    if dec is not None:
        return dec
    pts = []
    for wd in words:
        xc = wd["x"] + wd["w"] / 2; yc = wd["y"] + wd["h"] / 2
        pix = xc if axis == "x" else yc
        for kind, v in _parse_label(wd["text"]):
            pts.append((kind, pix, v))
    best = None
    for mode in ("plain", "pow"):
        pr = [(p, v) for k, p, v in pts if k == mode or (mode == "pow" and k == "plain" and v in (1.0, 10.0))]
        if len(pr) < 2:
            continue
        a = _calibrate(pr, None, tol=0.03)
        if a is not None and (best is None or a.residual < best.residual):
            best = a
    return best


def digitise_image(img_rgb, dpi_scale=1.0, page=0, source="raster"):
    """img_rgb: HxWx3 uint8 array.  Returns a list with at most one Plot."""
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fr = _frame(gray)
    if fr is None:
        return []
    left, top, right, bottom = fr
    H, W = gray.shape
    pad = 3
    up = 3
    tick = max(int(0.022 * H), 6)
    xs_reg = img_rgb[min(bottom + tick, H - 1): min(bottom + int(0.12 * H) + 6, H), max(left - 30, 0): min(right + 30, W)]
    tickx = max(int(0.022 * W), 6)
    ys_reg = img_rgb[max(top - 10, 0): min(bottom + 10, H), max(left - int(0.16 * W), 0): max(left - tickx, 1)]
    xw = _ocr(cv2.resize(cv2.cvtColor(xs_reg, cv2.COLOR_RGB2BGR), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)) \
        if xs_reg.size else []
    def _ocr_retry(reg):
        best = []
        for sub in (reg, reg[:, reg.shape[1] // 2:]):
            if sub.size == 0:
                continue
            big = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_RGB2BGR), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            for psm in (11, 6, 12):
                w_ = _ocr(big, psm=psm)
                num = [q for q in w_ if _parse_label(q["text"])]
                if len(num) > len(best):
                    off = reg.shape[1] - sub.shape[1]
                    for q in w_:
                        q["x"] += off * up
                    best = w_
                if len(num) >= 3:
                    return best
        return best

    yw = _ocr_retry(ys_reg) if ys_reg.size else []
    ox, oy = max(left - 30, 0), min(bottom + tick, H - 1)
    for wd in xw:
        wd["x"] = wd["x"] / up + ox; wd["y"] = wd["y"] / up + oy; wd["w"] /= up; wd["h"] /= up
    ox, oy = max(left - int(0.16 * W), 0), max(top - 10, 0)
    for wd in yw:
        wd["x"] = wd["x"] / up + ox; wd["y"] = wd["y"] / up + oy; wd["w"] /= up; wd["h"] /= up
    xa = _calibrate_ocr(xw, "x", fr, img_rgb)
    ya = _calibrate_ocr(yw, "y", fr, img_rgb)
    if xa is None or ya is None:
        return []
    inner = img_rgb[top + pad: bottom - pad, left + pad: right - pad]
    lab = cv2.cvtColor(inner, cv2.COLOR_RGB2LAB).astype(float)
    hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1] / 255.0; val = hsv[..., 2] / 255.0
    colored = (sat > 0.28) & (val > 0.15)
    series = []
    ys_, xs_ = np.where(colored)
    if len(xs_) > 30:
        feats = lab[ys_, xs_][:, 1:]
        best = None
        for k in range(1, 7):
            if len(feats) < k * 10:
                break
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            comp, lbl, cen = cv2.kmeans(feats.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            if best is not None and comp > 0.55 * best[0]:
                break
            best = (comp, lbl.ravel(), cen, k)
            if comp < 1e-3:
                break
        _, lbl, cen, k = best
        for j in range(k):
            m = np.zeros(colored.shape, np.uint8)
            sel = lbl == j
            m[ys_[sel], xs_[sel]] = 1
            rgb = tuple(np.median(inner[ys_[sel], xs_[sel]], axis=0) / 255.0)
            series.extend(_extract(m, rgb, left + pad, top + pad))
    # black data: remove text-like and straight lines
    dark = ((gray[top + pad: bottom - pad, left + pad: right - pad] < 90) & ~colored).astype(np.uint8)
    if dark.sum() > 30:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(dark.shape[1] // 6, 15), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(dark.shape[0] // 6, 15)))
        dark = dark & ~cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk) & ~cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
        series.extend(_extract(dark, (0.0, 0.0, 0.0), left + pad, top + pad, black=True))
    # legend: OCR inside the frame, attach 'T K' / 'd nm' words to the series sample on their left
    try:
        inner_bgr = cv2.cvtColor(img_rgb[top:bottom, left:right], cv2.COLOR_RGB2BGR)
        wl = _ocr(cv2.resize(inner_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        for w_ in wl:
            w_["x"] = w_["x"] / 2 + left; w_["y"] = w_["y"] / 2 + top; w_["w"] /= 2; w_["h"] /= 2
        # merge '250' + 'K' words on the same line
        wl.sort(key=lambda q: (round(q["y"] / 5), q["x"]))
        merged = []
        for q in wl:
            if merged and abs(merged[-1]["y"] - q["y"]) < 0.6 * q["h"] and 0 <= q["x"] - (merged[-1]["x"] + merged[-1]["w"]) < 1.2 * q["h"]:
                m_ = merged[-1]; m_["text"] += " " + q["text"]; m_["w"] = q["x"] + q["w"] - m_["x"]
            else:
                merged.append(dict(q))
        from ..io import tokens_from_text
        leg = [q for q in merged if tokens_from_text(q["text"].replace("°", "")) or re.search(r"\d+\s*K\b", q["text"])]
        for q in leg:
            yc = q["y"] + q["h"] / 2
            best = None
            pool = [s_ for s_ in series if s_.kind == "markers" and not s_.label and s_.color != (0.0, 0.0, 0.0)] or \
                [s_ for s_ in series if not s_.label]
            for s_ in pool:
                near = (np.abs(s_.y - yc) < 0.8 * q["h"]) & (s_.x < q["x"]) & (s_.x > q["x"] - 8 * q["h"])
                if near.any():
                    dist = q["x"] - s_.x[near].max()
                    if best is None or dist < best[0]:
                        best = (dist, s_, near)
            if best is not None and not best[1].label:
                best[1].label = q["text"]
                keep = ~best[2]
                best[1].x, best[1].y = best[1].x[keep], best[1].y[keep]
    except Exception:
        pass
    out = []
    for s in series:
        if len(s.x) < 3:
            continue
        s.x, s.y = xa.value(s.x), ya.value(s.y)
        out.append(s)
    xa.label = ""; ya.label = ""
    return [Plot(page=page, bbox=(left, top, right, bottom), xaxis=xa, yaxis=ya, series=out, source=source)]


def _extract(mask, rgb, ox, oy, black=False):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    w = stats[1:, cv2.CC_STAT_WIDTH]; h = stats[1:, cv2.CC_STAT_HEIGHT]
    medA = np.median(areas)
    blobs = (areas > 4) & (w < 4 * np.sqrt(medA) + 6) & (h < 4 * np.sqrt(medA) + 6) & (np.abs(w - h) <= 0.6 * np.maximum(w, h) + 2)
    out = []
    if blobs.sum() >= 4 and blobs.mean() > 0.5:
        typical = np.median(areas[blobs])
        ok = blobs & (areas > 0.4 * typical) & (areas < 2.5 * typical)
        c = cent[1:][ok]
        if len(c) >= 3:
            out.append(Series("markers", rgb, c[:, 0] + ox, c[:, 1] + oy, marker_size=float(np.sqrt(typical))))
        return out
    if black:
        return out        # black lines are too ambiguous with text; markers only
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return out
    cols = np.unique(xs)
    yy = np.array([np.median(ys[xs == c]) for c in cols])
    step = max(1, len(cols) // 200)
    out.append(Series("line", rgb, cols[::step] + ox, yy[::step] + oy))
    return out


def digitise_raster_pdf(path, pages=None, dpi=400):
    """Render pages that contain images and digitise each embedded figure."""
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        rng = range(len(pdf.pages)) if pages is None else pages
        for i in rng:
            pg = pdf.pages[i]
            for im in pg.images:
                bbox = (max(im["x0"], 0), max(im["top"], 0), min(im["x1"], pg.width), min(im["bottom"], pg.height))
                if (bbox[2] - bbox[0]) < 80 or (bbox[3] - bbox[1]) < 60:
                    continue
                try:
                    pil = pg.crop(bbox).to_image(resolution=dpi).original.convert("RGB")
                except Exception:
                    continue
                arr = np.array(pil)
                for p in digitise_image(arr, page=i, source="raster-in-pdf"):
                    out.append(p)
    return out


def digitise_image_file(path):
    from PIL import Image
    arr = np.array(Image.open(path).convert("RGB"))
    return digitise_image(arr, source=f"image:{os.path.basename(path)}")
