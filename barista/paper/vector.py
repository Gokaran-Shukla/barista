"""Exact data recovery from vector figures in PDF papers.

Most journal PDFs (and every LaTeX/matplotlib/Origin vector export) keep the plot as
drawing operators.  This module finds every axes frame on a page, calibrates both axes
from the tick labels (linear or logarithmic, including 10^n labels typeset with
superscripts), snaps labels to the drawn tick marks, and returns each data series —
polylines and marker clouds — in data coordinates, grouped by colour and labelled from
the legend.  The recovery is exact to the precision of the PDF coordinates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MINUS = "−–-"


@dataclass
class Axis:
    scale: str                    # 'linear' | 'log'
    a: float                      # value = a*pix + b   (log: log10 value)
    b: float
    label: str = ""
    ticks: list = field(default_factory=list)
    residual: float = 0.0

    def value(self, pix):
        v = self.a * np.asarray(pix, float) + self.b
        return 10 ** v if self.scale == "log" else v


@dataclass
class Series:
    kind: str                     # 'line' | 'markers'
    color: tuple
    x: np.ndarray
    y: np.ndarray
    label: str = ""
    marker_size: float = 0.0


@dataclass
class Plot:
    page: int
    bbox: tuple                   # x0, top, x1, bottom (PDF points)
    xaxis: Optional[Axis]
    yaxis: Optional[Axis]
    series: list
    panel: str = ""
    caption: str = ""
    source: str = "vector"

    def describe(self):
        xs = self.xaxis.label if self.xaxis else "?"
        ys = self.yaxis.label if self.yaxis else "?"
        return f"p{self.page + 1} {self.panel} [{ys} vs {xs}] {len(self.series)} series"


# ---------------------------------------------------------------------------
def _col(c):
    if c is None:
        return (0.0, 0.0, 0.0)
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    c = tuple(float(x) for x in c)
    if len(c) == 1:
        return c * 3
    if len(c) == 4:     # CMYK -> RGB
        C, M, Y, K = c
        return ((1 - C) * (1 - K), (1 - M) * (1 - K), (1 - Y) * (1 - K))
    return c[:3]


def _is_gray(c, tol=0.06):
    return max(c) - min(c) < tol


def _tokens(chars):
    """Group characters into tick-label tokens and evaluate them (handles 10^n superscripts)."""
    chars = sorted([c for c in chars if c["text"].strip()], key=lambda c: c["x0"])
    toks = []
    for c in chars:
        yc = (c["top"] + c["bottom"]) / 2
        home = None
        for t in toks:
            last = t[-1]
            ly = (last["top"] + last["bottom"]) / 2
            if -1.0 <= c["x0"] - last["x1"] <= 2.5 and abs(yc - ly) < 6:
                home = t
                break
        if home is None:
            toks.append([c])
        else:
            home.append(c)
    out = []
    for t in toks:
        sizes = [c["size"] for c in t]
        big = max(sizes)
        base = [c for c in t if c["size"] >= big - 0.3]
        sup = [c for c in t if c["size"] < big - 0.3 and c["top"] < base[0]["top"] + 1]
        btxt = "".join(c["text"] for c in base)
        stxt = "".join(c["text"] for c in sup)
        for m in MINUS:
            btxt = btxt.replace(m, "-"); stxt = stxt.replace(m, "-")
        val = None
        btxt = btxt.replace("×", "x").replace("·", "x")
        try:
            if sup and re.fullmatch(r"(?:(\d+(?:\.\d+)?)x)?10", btxt) and re.fullmatch(r"-?\d+", stxt):
                mant = re.fullmatch(r"(?:(\d+(?:\.\d+)?)x)?10", btxt).group(1)
                val = (float(mant) if mant else 1.0) * 10.0 ** int(stxt)
            elif re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", btxt):
                val = float(btxt)
            elif re.fullmatch(r"10\^?\{?-?\d+\}?", btxt):
                val = 10.0 ** int(re.sub(r"[^\d-]", "", btxt[2:]))
        except ValueError:
            val = None
        if val is None:
            continue
        x0 = min(c["x0"] for c in base); x1 = max(c["x1"] for c in base)
        top = min(c["top"] for c in base); bot = max(c["bottom"] for c in base)
        out.append({"value": val, "x0": x0, "x1": x1, "top": top, "bottom": bot,
                    "xc": (x0 + max(c["x1"] for c in t)) / 2, "yc": (top + bot) / 2,
                    "pow10": bool(sup) or btxt.startswith("10^"),
                    "x1_all": max(c["x1"] for c in t)})
    return out


def _calibrate(pairs, ticks=None, tol=0.01):
    """Robust version: drop up to two inconsistent labels (e.g. stray numbers) if needed."""
    import itertools
    best = _calibrate1(pairs, ticks)
    if best is not None and best.residual <= tol:
        return best
    n = len(pairs)
    for k in (1, 2):
        if n - k < 2:
            break
        cands = []
        for drop in itertools.combinations(range(n), k):
            sub = [p for i, p in enumerate(pairs) if i not in drop]
            a = _calibrate1(sub, ticks)
            if a is not None and a.residual <= tol:
                cands.append(a)
        if cands:
            return min(cands, key=lambda a: a.residual)
    return best if best is not None and best.residual <= 0.05 else None


def _calibrate1(pairs, ticks=None):
    """pairs: [(pixel, value)] -> Axis.  Chooses linear or log by residual."""
    if len(pairs) < 2:
        return None
    pix = np.array([p for p, _ in pairs], float)
    val = np.array([v for _, v in pairs], float)
    if ticks is not None and len(ticks):
        tk = np.asarray(ticks, float)
        tol = max(4.0, 0.35 * float(np.median(np.diff(tk)))) if len(tk) > 1 else 4.0
        pix = np.array([tk[np.argmin(np.abs(tk - p))] if np.min(np.abs(tk - p)) < tol else p for p in pix])
    best = None
    for scale in ("linear", "log"):
        if scale == "log" and np.any(val <= 0):
            continue
        y = val if scale == "linear" else np.log10(val)
        if np.ptp(pix) < 1e-6 or np.ptp(y) == 0:
            continue
        A = np.vstack([pix, np.ones_like(pix)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = np.max(np.abs(A @ coef - y)) / max(np.ptp(y), 1e-12)
        if best is None or res < best[2] - 1e-9:
            best = (scale, coef, res)
    if best is None:
        return None
    scale, coef, res = best
    # equally spaced powers of ten are ambiguous only if values are a geometric series
    return Axis(scale=scale, a=float(coef[0]), b=float(coef[1]), ticks=list(zip(pix.tolist(), val.tolist())),
                residual=float(res))


class _GPage:
    """pdfplumber page whose vector graphics come from PyMuPDF when available (keeps marker colours
    defined through XObjects, e.g. matplotlib PDFs)."""

    def __init__(self, page, fitz_page=None):
        self._p = page
        self.chars = page.chars
        self.width, self.height = page.width, page.height
        self.extract_words = page.extract_words
        if fitz_page is None:
            self.curves, self.lines, self.rects = page.curves, page.lines, page.rects
            return
        curves, lines, rects = [], [], []
        for d in fitz_page.get_drawings():
            r = d["rect"]
            base = {"x0": r.x0, "x1": r.x1, "top": r.y0, "bottom": r.y1, "width": r.width, "height": r.height,
                    "stroke": d.get("color") is not None, "fill": d.get("fill") is not None,
                    "stroking_color": d.get("color"), "non_stroking_color": d.get("fill"),
                    "linewidth": d.get("width") or 0}
            items = d["items"]
            if len(items) == 1 and items[0][0] == "re":
                rects.append(base); continue
            if len(items) == 1 and items[0][0] == "l":
                a, b = items[0][1], items[0][2]
                lines.append({**base, "x0": min(a.x, b.x), "x1": max(a.x, b.x), "top": min(a.y, b.y),
                              "bottom": max(a.y, b.y)}); continue
            pts = []
            for it in items:
                if it[0] == "l":
                    pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
                elif it[0] == "c":
                    pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
                elif it[0] == "re":
                    q = it[1]
                    pts += [(q.x0, q.y0), (q.x1, q.y0), (q.x1, q.y1), (q.x0, q.y1)]
                elif it[0] == "qu":
                    q = it[1]
                    pts += [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
            # drop consecutive duplicates
            dedup = [pts[0]] if pts else []
            for q in pts[1:]:
                if abs(q[0] - dedup[-1][0]) > 1e-6 or abs(q[1] - dedup[-1][1]) > 1e-6:
                    dedup.append(q)
            curves.append({**base, "pts": dedup})
        self.curves, self.lines, self.rects = curves, lines, rects


def _frames(page):
    """Axes rectangles: stroked rects, or boxes/L-shapes built from long lines."""
    fr = []
    for r in page.rects:
        if r.get("stroke") and r["width"] > 50 and r["height"] > 35:
            fr.append((r["x0"], r["top"], r["x1"], r["bottom"]))
    segs = list(page.lines)
    for c in page.curves:
        pts = c["pts"]
        if len(pts) == 2:
            segs.append({"x0": min(pts[0][0], pts[1][0]), "x1": max(pts[0][0], pts[1][0]),
                         "top": min(pts[0][1], pts[1][1]), "bottom": max(pts[0][1], pts[1][1])})
        elif len(pts) in (4, 5) and c["x1"] - c["x0"] > 50 and c["bottom"] - c["top"] > 35:
            xs = sorted({round(p[0], 1) for p in pts}); ys = sorted({round(p[1], 1) for p in pts})
            if len(xs) == 2 and len(ys) == 2 and not c.get("fill"):
                fr.append((xs[0], ys[0], xs[1], ys[1]))
    H = [s for s in segs if abs(s["bottom"] - s["top"]) < 0.8 and s["x1"] - s["x0"] > 50]
    V = [s for s in segs if abs(s["x1"] - s["x0"]) < 0.8 and s["bottom"] - s["top"] > 35]
    for h in H:
        for v in V:
            # bottom-left corner
            if abs(v["x0"] - h["x0"]) < 1.5 and abs(v["bottom"] - h["top"]) < 1.5:
                fr.append((h["x0"], v["top"], h["x1"], h["top"]))
    # unique, discard frames that contain others with nearly the same size (duplicates)
    out = []
    for f in sorted(set(tuple(round(x, 1) for x in f) for f in fr), key=lambda f: (f[1], f[0])):
        if all(max(abs(f[i] - g[i]) for i in range(4)) > 2 for g in out):
            out.append(f)
    # drop legend boxes / insets that are fully inside a larger frame and small
    final = []
    for f in out:
        inside = [g for g in out if g != f and g[0] <= f[0] + 1 and g[1] <= f[1] + 1 and g[2] >= f[2] - 1
                  and g[3] >= f[3] - 1]
        area = (f[2] - f[0]) * (f[3] - f[1])
        if inside and area < 0.35 * min((g[2] - g[0]) * (g[3] - g[1]) for g in inside):
            continue
        final.append(f)
    return final


def _ticks(page, f, axis):
    """Major tick positions (minor ticks, which are shorter, are discarded)."""
    x0, top, x1, bot = f
    out = []
    lens = []
    segs = list(page.lines)
    for s in segs:
        w, h = s["x1"] - s["x0"], s["bottom"] - s["top"]
        if axis == "x" and w < 0.6 and 1 < h < 9 and x0 - 1 <= s["x0"] <= x1 + 1 and \
                (abs(s["bottom"] - bot) < 1.5 or abs(s["top"] - bot) < 1.5):
            out.append(s["x0"]); lens.append(h)
        if axis == "y" and h < 0.6 and 1 < w < 9 and top - 1 <= s["top"] <= bot + 1 and \
                (abs(s["x0"] - x0) < 1.5 or abs(s["x1"] - x0) < 1.5):
            out.append(s["top"]); lens.append(w)
    if not out:
        return []
    L = max(lens)
    return sorted(set(round(v, 2) for v, l in zip(out, lens) if l >= 0.8 * L))


def _axis_title(page, f, axis):
    x0, top, x1, bot = f
    if axis == "x":
        ws = [w for w in page.extract_words(keep_blank_chars=True, use_text_flow=True)
              if w["top"] > bot + 8 and w["top"] < bot + 30 and w["x0"] > x0 - 10 and w["x1"] < x1 + 10]
        ws = [w for w in ws if not re.fullmatch(r"[\d\.\s−\-]+", w["text"])]
        ws.sort(key=lambda w: (round(w["top"]), w["x0"]))
        return " ".join(w["text"] for w in ws[:6]).strip()
    chars = [c for c in page.chars if not c["upright"] and c["x1"] < x0 - 2 and c["x0"] > x0 - 45 and
             top - 5 < c["top"] < bot + 5]
    if chars:
        chars.sort(key=lambda c: -c["bottom"])
        return "".join(c["text"] for c in chars).strip()
    ws = [w for w in page.extract_words() if w["x1"] < x0 - 14 and w["x0"] > x0 - 70 and top < w["top"] < bot]
    return " ".join(w["text"] for w in ws).strip()


def _series(page, f, legend_words):
    x0, top, x1, bot = f
    pad = 0.6

    def inside(o):
        return o["x0"] >= x0 - pad and o["x1"] <= x1 + pad and o["top"] >= top - pad and o["bottom"] <= bot + pad

    def overlaps(o):
        return o["x1"] > x0 and o["x0"] < x1 and o["bottom"] > top and o["top"] < bot

    lines, marks = {}, {}
    samples = []      # legend samples: (colour, x_right, y)
    for c in page.curves:
        if not overlaps(c):
            continue
        pts = np.array(c["pts"], float)
        w, h = c["x1"] - c["x0"], c["bottom"] - c["top"]
        if len(pts) == 2 and h < 0.6 and 5 < w < 35 and inside(c):
            samples.append((_col(c.get("stroking_color")), c["x1"], (c["top"] + c["bottom"]) / 2))
            continue
        if not inside(c) and not (len(pts) >= 4 and max(w, h) > 8):
            continue
        sc, fc = _col(c.get("stroking_color")), _col(c.get("non_stroking_color"))
        if max(w, h) <= 7 and len(pts) >= 3:                     # marker drawn as a small path
            key = ("m", sc if c.get("stroke") else fc, round(max(w, h), 1), len(pts), bool(c.get("fill")))
            marks.setdefault(key, []).append(((c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2))
        elif len(pts) >= 4 and max(w, h) > 8:
            if abs(h) < 0.5 or abs(w) < 0.5:                       # straight grid/reference line
                continue
            m = (pts[:, 0] >= x0 - pad) & (pts[:, 0] <= x1 + pad) & (pts[:, 1] >= top - pad) & (pts[:, 1] <= bot + pad)
            if m.sum() >= 4:
                key = ("l", sc)
                lines.setdefault(key, []).append(pts[m])
    for l in page.lines:
        if inside(l) and (l["bottom"] - l["top"]) < 0.6 and 5 < (l["x1"] - l["x0"]) < 35:
            samples.append((_col(l.get("stroking_color")), l["x1"], (l["top"] + l["bottom"]) / 2))
    for r in page.rects:
        if inside(r) and r["width"] <= 7 and r["height"] <= 7 and r["width"] > 0.5:
            col = _col(r.get("stroking_color") if r.get("stroke") else r.get("non_stroking_color"))
            key = ("m", col, round(r["width"], 1), 4, bool(r.get("fill")))
            marks.setdefault(key, []).append(((r["x0"] + r["x1"]) / 2, (r["top"] + r["bottom"]) / 2))
    out = []
    for (k, col), segs in lines.items():
        pts = np.vstack(segs)
        if len(pts) < 4:
            continue
        out.append(Series("line", col, pts[:, 0], pts[:, 1]))
    for (k, col, size, npts, fill), P in marks.items():
        P = np.array(P)
        if len(P) < 3:
            continue
        out.append(Series("markers", col, P[:, 0], P[:, 1], marker_size=size))
    # legend from line samples: colour -> text to the right of the sample
    col_label = {}
    for col, xr, yc in samples:
        cands = [w for w in legend_words if abs((w["top"] + w["bottom"]) / 2 - yc) < 4 and 0 <= w["x0"] - xr < 20]
        if cands:
            col_label.setdefault(col, min(cands, key=lambda w: w["x0"] - xr)["text"])
    for s in out:
        if s.kind == "line" and s.color in col_label:
            s.label = col_label[s.color]
    # legend: words inside the frame with a marker sample of the same colour just to their left
    for s in out:
        if s.label:
            continue
        best = None
        for w in legend_words:
            yc = (w["top"] + w["bottom"]) / 2
            near = (np.abs(s.y - yc) < 3.5) & (s.x < w["x0"]) & (s.x > w["x0"] - 30)
            if near.any():
                d = w["x0"] - s.x[near].max()
                if best is None or d < best[0]:
                    best = (d, w, near)
        if best is not None:
            s.label = best[1]["text"]
            keep = ~best[2]
            s.x, s.y = s.x[keep], s.y[keep]
    return [s for s in out if len(s.x) >= 3]


def _force_log(tokens):
    v = [t["value"] for t in tokens]
    if len(v) >= 2 and all(x > 0 for x in v):
        lg = np.log10(v)
        if np.all(np.abs(lg - np.round(lg)) < 1e-6) and len(set(np.round(lg))) == len(lg):
            return True
    return any(t["pow10"] for t in tokens)


def digitise_page(page, page_no=0, fitz_page=None):
    page = _GPage(page, fitz_page)
    plots = []
    words_all = page.extract_words(keep_blank_chars=True)
    for f in _frames(page):
        x0, top, x1, bot = f
        xchars = [c for c in page.chars if c["upright"] and c["top"] > bot - 1 and c["top"] < bot + 16 and
                  x0 - 15 < c["x0"] < x1 + 15]
        ychars = [c for c in page.chars if c["upright"] and c["x1"] < x0 + 1 and c["x0"] > x0 - 45 and
                  top - 8 < c["top"] < bot + 4]
        xt = _tokens(xchars)
        yt = _tokens(ychars)
        xa = _calibrate([(t["xc"], t["value"]) for t in xt], _ticks(page, f, "x"))
        ya = _calibrate([(t["yc"], t["value"]) for t in yt], _ticks(page, f, "y"))
        if xa is None or ya is None:
            continue
        for ax_name, toks in (("x", xt), ("y", yt)):
            axo = xa if ax_name == "x" else ya
            if axo.scale != "log" and _force_log(toks) and all(t["value"] > 0 for t in toks):
                key = "xc" if ax_name == "x" else "yc"
                pairs = [(t[key], t["value"]) for t in toks]
                pix = np.array([p for p, _ in pairs]); lv = np.log10([v for _, v in pairs])
                tk = np.asarray(_ticks(page, f, ax_name), float)
                if len(tk):
                    tol = max(4.0, 0.35 * float(np.median(np.diff(tk)))) if len(tk) > 1 else 4.0
                    pix = np.array([tk[np.argmin(np.abs(tk - q))] if np.min(np.abs(tk - q)) < tol else q for q in pix])
                A = np.vstack([pix, np.ones_like(pix)]).T
                coef, *_ = np.linalg.lstsq(A, lv, rcond=None)
                new = Axis("log", float(coef[0]), float(coef[1]), ticks=pairs, residual=0.0)
                if ax_name == "x":
                    xa = new
                else:
                    ya = new
        xa.label = _axis_title(page, f, "x"); ya.label = _axis_title(page, f, "y")
        legend_words = [w for w in words_all if x0 < w["x0"] < x1 and top < w["top"] < bot]
        ser = _series(page, f, legend_words)
        for s in ser:
            s.x, s.y = xa.value(s.x), ya.value(s.y)
        panel = ""
        for w in words_all:
            m = re.match(r"^\(([a-z])\)", w["text"])
            if m and abs(w["bottom"] - top) < 22 and x0 - 30 < w["x0"] < x1:
                panel = m.group(1); break
        plots.append(Plot(page=page_no, bbox=f, xaxis=xa, yaxis=ya, series=ser, panel=panel))
    return plots


def digitise_pdf(path, pages=None):
    import pdfplumber
    try:
        import pymupdf as fitz
        fdoc = fitz.open(path)
    except Exception:
        fdoc = None
    out = []
    with pdfplumber.open(path) as pdf:
        rng = range(len(pdf.pages)) if pages is None else pages
        for i in rng:
            try:
                out.extend(digitise_page(pdf.pages[i], i, fdoc[i] if fdoc is not None else None))
            except Exception:
                continue
    return out
