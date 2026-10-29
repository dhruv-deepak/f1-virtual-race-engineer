"""
Shared design kit for the project's slide decks.

Every token here was extracted from the Review-1 deck rather than invented: the
same left margin and content grid, the badge/title/intro rhythm, the rounded-rect
chips and stat cards, the dark-header table, and the Raleway/Roboto/Consolas
pairing. Both the Review 2 and Review 3 generators build on it, so the decks
cannot drift apart.

A deck starts from the Review-1 file itself, which carries the theme, fonts and
masters over unchanged.
"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


# ---------------------------------------------------------------- tokens ----
LEFT      = 661475            # 0.723"
CONTENT_W = 10868745          # 11.885"
RIGHT     = LEFT + CONTENT_W
BADGE_Y   = 542230
TITLE_Y   = 785614
INTRO_Y   = 1330000
EMU_IN    = 914400

INK       = RGBColor(0x1B, 0x1B, 0x27)   # headings
BODY      = RGBColor(0x3C, 0x39, 0x39)   # body copy
CHIP      = RGBColor(0xE1, 0xE1, 0xEA)   # badges / stat cards / callouts
CHIP_MID  = RGBColor(0xC7, 0xC7, 0xD0)
CARD      = RGBColor(0xF8, 0xFA, 0xFC)
CARD2     = RGBColor(0xF1, 0xF5, 0xF9)
TBL_HDR   = RGBColor(0x0F, 0x17, 0x2A)
TBL_ALT   = RGBColor(0x1E, 0x29, 0x3B)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT    = RGBColor(0xC4, 0x1E, 0x3A)   # F1 crimson, used sparingly

H_FONT, B_FONT, M_FONT = "Raleway", "Roboto", "Consolas"


# ------------------------------------------------------------- utilities ----
DEFAULT_SRC = Path(os.environ.get(
    "REVIEW1_PPTX",
    r"C:\E\VIT\VIT 7th Sem\Deep Learning Project\Review-1.pptx"))
FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"


def blank_deck(src=None):
    """Start from Review-1 so the theme, fonts and masters carry over."""
    prs = Presentation(str(src or DEFAULT_SRC))
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        rId = sld.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rId)
        xml_slides.remove(sld)
    return prs


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[0])


def _style(tf, size, color, font, bold=False, align=PP_ALIGN.LEFT,
           line_pct=1.15, space_after=0):
    for p in tf.paragraphs:
        p.alignment = align
        p.line_spacing = line_pct
        p.space_after = Pt(space_after)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.name = font
            r.font.color.rgb = color


def text(slide, x, y, w, h, s, size=11, color=BODY, font=B_FONT, bold=False,
         align=PP_ALIGN.LEFT, line_pct=1.15, anchor=MSO_ANCHOR.TOP, space_after=0):
    box = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = s.split("\n")
    tf.text = lines[0]
    for extra in lines[1:]:
        tf.add_paragraph().text = extra
    _style(tf, size, color, font, bold, align, line_pct, space_after)
    return box


def rrect(slide, x, y, w, h, fill=CHIP, adj=0.09, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    sh.adjustments[0] = adj
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def badge(slide, label):
    w = Emu(int(len(label) * 62000 + 190000))
    sh = rrect(slide, LEFT, BADGE_Y, int(w), 201116, CHIP, adj=0.22)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = label
    _style(tf, 8, BODY, B_FONT, align=PP_ALIGN.CENTER, line_pct=0.96)
    return sh


def title(slide, s, size=26):
    return text(slide, LEFT, TITLE_Y, CONTENT_W, 413445, s, size, INK, H_FONT, line_pct=1.04)


def intro(slide, s, h=420000, y=INTRO_Y):
    return text(slide, LEFT, y, CONTENT_W, h, s, 11, BODY, B_FONT, line_pct=1.28)


def head(slide, x, y, w, s, size=13):
    return text(slide, x, y, w, 254135, s, size, INK, H_FONT, line_pct=1.04)


def statcard(slide, x, y, w, h, value, label, fill=CHIP, vsize=18, lsize=7.5,
             vcolor=INK):
    sh = rrect(slide, x, y, w, h, fill, adj=0.09)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(45000)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = value
    p2 = tf.add_paragraph()
    p2.text = label
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 1.06
    for r in tf.paragraphs[0].runs:
        r.font.size, r.font.bold, r.font.name = Pt(vsize), True, H_FONT
        r.font.color.rgb = vcolor
    for r in tf.paragraphs[1].runs:
        r.font.size, r.font.name = Pt(lsize), B_FONT
        r.font.color.rgb = BODY
    return sh


def callout(slide, s, y=5705078, h=648097, size=10.5):
    rrect(slide, LEFT, y, CONTENT_W, h, CHIP, adj=0.086)
    return text(slide, LEFT + 200000, y + 90000, CONTENT_W - 400000, h - 180000,
                s, size, BODY, B_FONT, line_pct=1.22, anchor=MSO_ANCHOR.MIDDLE)


def bullet_rows(slide, x, y, w, items, gap=560000, hsize=11, bsize=9.5,
                head_h=210000, body_h=300000):
    """Repeating 'bold header + description' block, the deck's main motif."""
    for i, (h_, b_) in enumerate(items):
        yy = y + i * gap
        text(slide, x, yy, w, head_h, h_, hsize, INK, H_FONT)
        text(slide, x, yy + 250000, w, body_h, b_, bsize, BODY, B_FONT, line_pct=1.22)


def table(slide, x, y, w, rows, col_w, row_h=290000, hdr_h=300000,
          fsize=9.75, hsize=9.75, bold_col0=False, highlight_rows=()):
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Emu(x), Emu(y), Emu(w), Emu(hdr_h + (n_r - 1) * row_h))
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(cw)
    tbl.rows[0].height = Emu(hdr_h)
    for i in range(1, n_r):
        tbl.rows[i].height = Emu(row_h)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Emu(72000)
            c.margin_top = c.margin_bottom = Emu(18000)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = TBL_HDR
            elif i in highlight_rows:
                c.fill.fore_color.rgb = CHIP
            else:
                c.fill.fore_color.rgb = CARD if i % 2 else WHITE
            tf = c.text_frame
            tf.word_wrap = True
            tf.text = str(val)
            for p in tf.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(hsize if i == 0 else fsize)
                    r.font.name = B_FONT
                    r.font.bold = (i == 0) or (j == 0 and bold_col0) or (i in highlight_rows)
                    r.font.color.rgb = WHITE if i == 0 else (INK if i in highlight_rows else BODY)
    return gf


def picture(slide, path, x, y, w=None, h=None):
    if w is not None:
        return slide.shapes.add_picture(str(path), Emu(x), Emu(y), width=Emu(w))
    return slide.shapes.add_picture(str(path), Emu(x), Emu(y), height=Emu(h))


def caption(slide, x, y, w, s, size=8.5):
    return text(slide, x, y, w, 230000, s, size, BODY, B_FONT, align=PP_ALIGN.CENTER,
                line_pct=1.15)
