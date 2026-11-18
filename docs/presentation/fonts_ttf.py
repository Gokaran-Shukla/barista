"""Convert the Inter OpenType (CFF) fonts to TrueType so matplotlib embeds them cleanly (Type 42).

The converted family is called "Inter TT" to keep it apart from the system OTF files.
Used by make_figures.py; the TTF files are written to docs/presentation/fonts/.
"""
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

FAMILY = "Inter TT"
STYLES = {"Regular": "Inter-Regular", "Italic": "Inter-Italic", "Bold": "Inter-Bold",
          "Bold Italic": "Inter-BoldItalic", "Medium": "Inter-Medium", "SemiBold": "Inter-SemiBold"}


def find_otf(name):
    for d in ("/usr/share/fonts/opentype/inter", "/usr/share/fonts", str(Path.home() / ".fonts")):
        hits = list(Path(d).rglob(f"{name}.otf")) if Path(d).exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def otf_to_ttf(src, dst, style):
    f = TTFont(src)
    gs = f.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder = f.getGlyphOrder()
    glyf.glyphs = {}
    for g in glyf.glyphOrder:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[g] = pen.glyph()
    f["glyf"] = glyf
    f["loca"] = newTable("loca")
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    f["maxp"] = newTable("maxp"); m = f["maxp"]
    m.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(m, k, 0)
    m.maxZones = 1
    f["head"].glyphDataFormat = 0
    f["post"].formatType = 2.0
    f["post"].extraNames = []; f["post"].mapping = {}
    f.sfntVersion = "\x00\x01\x00\x00"
    name = f["name"]
    full = f"{FAMILY} {style}".replace(" Regular", "")
    for rec in list(name.names):
        if rec.nameID in (1, 16):
            name.setName(FAMILY, rec.nameID, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 4:
            name.setName(full, 4, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 6:
            name.setName(full.replace(" ", "-"), 6, rec.platformID, rec.platEncID, rec.langID)
    f.save(dst)


def build(outdir):
    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    files = []
    for style, fn in STYLES.items():
        dst = outdir / f"InterTT-{style.replace(' ', '')}.ttf"
        if not dst.exists():
            otf_to_ttf(find_otf(fn), dst, style)
        files.append(dst)
    return files


if __name__ == "__main__":
    print(build(Path(__file__).parent / "fonts"))
