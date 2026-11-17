"""Read a paper's text: metadata, sample details, the mechanism the authors claim and the
parameters they report."""
from __future__ import annotations

import re
from collections import Counter

MECH_KEYWORDS = {
    "poole_frenkel": [r"poole[\s\-–]*frenkel", r"\bPF\b emission", r"frenkel[\s\-]*poole"],
    "schottky": [r"schottky emission", r"thermionic emission", r"richardson[\s\-–]*schottky", r"schottky barrier"],
    "coherent": [r"direct tunnel+ing", r"fowler[\s\-–]*nordheim", r"\bF-?N tunnel", r"simmons", r"coherent tunnel+ing",
                 r"field emission", r"tunnel+ing through the barrier"],
    "tat": [r"trap[\s\-]*assisted tunnel+ing", r"\bTAT\b", r"multiphonon"],
    "sclc_trap": [r"space[\s\-]*charge[\s\-]*limited", r"\bSCLC\b", r"mott[\s\-–]*gurney", r"trap[\s\-]*filled[\s\-]*limit",
                  r"\bTFL\b", r"child'?s law"],
    "sclc_exp": [r"exponential(ly)? distributed traps", r"mark[\s\-–]*helfrich"],
    "nn_hopping": [r"nearest[\s\-]*neighbou?r hopping", r"hopping conduction", r"ionic conduction", r"ionic hopping"],
    "vrh": [r"variable[\s\-]*range hopping", r"\bVRH\b", r"efros[\s\-–]*shklovskii", r"mott hopping"],
    "polaron": [r"polaron", r"marcus"],
    "gdm": [r"gaussian disorder", r"b[äa]ssler"],
    "ohmic_metallic": [r"filament", r"pinhole", r"metallic conduction", r"ohmic conduction"],
    "glazman_matveev": [r"glazman", r"matveev", r"inelastic tunnel+ing through .* localized states"],
    "band_to_band": [r"band[\s\-]*to[\s\-]*band tunnel+ing", r"zener tunnel+ing", r"kane"],
}

NUM = r"([0-9]+(?:\.[0-9]+)?(?:\s*[×x]\s*10\s*[\^]?\s*[−\-]?\s*[0-9]+|[eE][−\-]?[0-9]+)?)"


def _num(s):
    s = s.replace("−", "-").replace(" ", "")
    m = re.match(r"([0-9.]+)[×x]10\^?(-?[0-9]+)", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return float(s)


def full_text(path):
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        meta = pdf.metadata or {}
        first = pdf.pages[0] if pdf.pages else None
        title = ""
        if first is not None:
            chars = [c for c in first.chars if c["top"] < first.height * 0.4]
            if chars:
                big = max(c["size"] for c in chars)
                tchars = [c for c in chars if c["size"] >= big - 0.5]
                tchars.sort(key=lambda c: (round(c["top"]), c["x0"]))
                title = "".join(c["text"] for c in tchars)
                title = re.sub(r"\s+", " ", title).strip()
        for p in pdf.pages:
            try:
                pages.append(p.extract_text(x_tolerance=1.5) or "")
            except Exception:
                pages.append("")
    text = "\n".join(pages)
    return text, pages, (meta.get("Title") or title or "").strip()


def doi(text):
    m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", text)
    return m.group(0).rstrip(".,;)") if m else None


def abstract(text):
    m = re.search(r"(?is)abstract[:.\s]*(.{200,2500}?)(?:\n\s*(?:1\.?|I\.)?\s*introduction|\n\s*keywords)", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"\s+", " ", text[:1800]).strip()


def claimed_mechanism(text):
    """Mechanism(s) the authors conclude, scored by keyword frequency; abstract and conclusions weigh 3×."""
    low = text.lower()
    abs_ = abstract(text).lower()
    concl = ""
    m = re.search(r"(?is)\n\s*(?:[IVX0-9]+\.?\s*)?(?:conclusions?|summary)\s*\n(.{100,4000})", text)
    if m:
        concl = m.group(1).lower()
    score = Counter()
    hits = {}
    for key, pats in MECH_KEYWORDS.items():
        for p in pats:
            n_all = len(re.findall(p, low)); n_imp = len(re.findall(p, abs_)) + len(re.findall(p, concl))
            if n_all or n_imp:
                score[key] += n_all + 3 * n_imp
                hits.setdefault(key, []).append(p)
    ranked = score.most_common()
    return ranked, hits


def _grab(text, patterns, unit_re=None, lo=None, hi=None):
    vals = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.I):
            try:
                v = _num(m.group(1))
            except Exception:
                continue
            if (lo is None or v >= lo) and (hi is None or v <= hi):
                vals.append((v, text[max(0, m.start() - 60): m.end() + 20].replace("\n", " ")))
    return vals


def reported_parameters(text):
    t = text.replace("\n", " ")
    out = {
        "thickness_nm": _grab(t, [rf"{NUM}\s*nm[\s\-]*thick", rf"thickness(?:es)?\s*(?:of|=|:|was|is)?\s*(?:about|~|≈)?\s*{NUM}\s*nm",
                                  rf"\(?d\s*=\s*{NUM}\s*nm"], lo=0.2, hi=1e5),
        "area_um2": _grab(t, [rf"area\s*(?:of|=|:)?\s*(?:about|~|≈)?\s*{NUM}\s*(?:µm|μm|um)\s*(?:2|²)",
                              rf"{NUM}\s*(?:×|x)\s*{NUM}\s*(?:µm|μm|um)\s*(?:2|²)?"], lo=1e-6, hi=1e10),
        "barrier_eV": _grab(t, [rf"barrier height[^.;]{{0,30}}?(?:of|=|:|is|be|was|were|are)\s*(?:about|~|≈)?\s*{NUM}\s*eV",
                                rf"(?:Φ|ϕ|φ)\s*(?:B|b)?\s*=\s*{NUM}\s*eV"], lo=0.01, hi=6),
        "trap_depth_eV": _grab(t, [rf"trap (?:depth|energy|level)[^.;]{{0,30}}?(?:of|=|:|is|at|be|was)\s*(?:about|~|≈)?\s*{NUM}\s*eV",
                                   rf"(?:Φ|ϕ|φ)\s*t\s*=\s*{NUM}\s*eV", rf"activation energy\s*(?:of|=|:)?\s*(?:about|~|≈)?\s*{NUM}\s*(?:m?eV)"],
                               lo=0.001, hi=5),
        "eps_r": _grab(t, [rf"(?:dielectric constant|permittivity|ε\s*r?|κ)[^.;]{{0,25}}?(?:of|=|:|is|be|was|were|are)\s*(?:about|~|≈)?\s*{NUM}(?!\s*(?:nm|eV|K|V))"],
                       lo=1, hi=100),
        "n_opt": _grab(t, [rf"refractive index[^.;]{{0,30}}?(?:=|of|is|:)\s*(?:about|~|≈)?\s*{NUM}"], lo=1, hi=5),
        "trap_density_cm3": _grab(t, [rf"trap density[^.;]{{0,30}}?(?:of|=|:|is|be|was|were|are)\s*(?:about|~|≈)?\s*{NUM}\s*cm"],
                                  lo=1e8, hi=1e23),
        "mass_ratio": _grab(t, [rf"effective mass\s*(?:of|=|:|m\*\s*=)?\s*(?:about|~|≈)?\s*{NUM}\s*m", rf"m\*\s*=\s*{NUM}\s*m"],
                            lo=0.01, hi=5),
        "mobility_cm2Vs": _grab(t, [rf"mobility\s*(?:of|=|:|μ\s*=)?\s*(?:about|~|≈)?\s*{NUM}\s*cm"], lo=1e-8, hi=1e5),
    }
    temps = [float(m) for m in re.findall(r"([0-9]{1,3}(?:\.[0-9])?)\s*K\b", t) if 1 <= float(m) <= 800]
    out["temperatures_K"] = sorted(set(temps))
    return out


def best_guess(vals):
    if not vals:
        return None
    c = Counter(round(v, 6) for v, _ in vals)
    return c.most_common(1)[0][0]
