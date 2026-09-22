"""
Build Review-2.pptx in the exact visual language of Review-1.pptx.

Run:  python slides/build_review2_deck.py
The Review-1 deck supplies the theme, fonts and masters; set REVIEW1_PPTX to
point at it. Figures are read from reports/figures/, so re-running after a
retrain picks up the new curves automatically.

Design tokens were extracted from Review-1 rather than invented: the same left
margin, badge/title/intro rhythm, rounded-rect chips, stat cards, table styling,
and the Raleway/Roboto/Consolas type pairing.
"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SRC = Path(os.environ.get("REVIEW1_PPTX", r"C:\E\VIT\VIT 7th Sem\Deep Learning Project\Review-1.pptx"))
OUT = Path(r"C:\E\VIT\VIT 7th Sem\Deep Learning Project\Review-2.pptx")
FIG = Path(r"C:\E\Project\Project\reports\figures")

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
def blank_deck():
    """Start from Review-1 so the theme, fonts and masters carry over."""
    prs = Presentation(str(SRC))
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


# ================================================================== deck ====
prs = blank_deck()

# ---------------------------------------------------------------- 1 title ---
s = add_slide(prs)
text(s, LEFT, 1_150_000, CONTENT_W, 250000, "REVIEW 2  ·  PROGRESS PRESENTATION",
     10, BODY, B_FONT)
text(s, LEFT, 1_450_000, CONTENT_W, 1_130_000,
     "Deep Learning-Based Virtual Race Engineer for\nReal-Time Formula 1 Strategy Optimization",
     30, INK, H_FONT, line_pct=1.10)
rrect(s, LEFT, 2_900_000, CONTENT_W, 18000, CHIP_MID, adj=0.0)
text(s, LEFT, 3_170_000, CONTENT_W, 300000,
     "Model 1 Implementation, Training and Evaluation on Real F1 Telemetry",
     12, BODY, B_FONT)

chips = [("Model", "Bi-LSTM  ·  104,769 params"),
         ("Dataset", "90 races  ·  98,357 driver-laps"),
         ("Test ROC-AUC", "0.806  ·  2024 season held out")]
cw, cgap = 3_630_000, 3_705_000
for i, (k, v) in enumerate(chips):
    x = LEFT + i * cgap
    rrect(s, x, 3_860_000, cw, 900_000, CHIP, adj=0.09)
    text(s, x + 230000, 3_990_000, cw - 460000, 200000, k, 8, BODY, B_FONT)
    text(s, x + 230000, 4_230_000, cw - 460000, 300000, v, 11, INK, H_FONT)

text(s, LEFT + 300000, 5_400_000, 7_200_000, 820_000,
     "Submitted By:\n"
     "Dhruv Deepak (23BCE9187)          Akhil V (23BCE8051)\n"
     "Mohan Simha Varma Dantuluri (23BCE9229)          Surya Prathap Mamillapalli (23BCE7783)",
     10.5, BODY, B_FONT, line_pct=1.45)

# ------------------------------------------------------- 2 review 2 scope ---
s = add_slide(prs)
badge(s, "01 — REVIEW 2 SCOPE")
title(s, "What Was Asked, and What Is Delivered")
intro(s, "Review 1 proposed the system. Review 2 implements and evaluates the first of its "
         "three models end to end on real Formula 1 telemetry, with the code, results, "
         "training curves and confusion matrix presented below.")
head(s, LEFT, 1_950_000, 5_130_000, "Requirement")
rrect(s, LEFT, 2_300_000, 5_130_000, 2_900_000, CARD, adj=0.03, line=CHIP_MID)
bullet_rows(s, LEFT + 260000, 2_500_000, 4_630_000, [
    ("Execute at least one of three models",
     "Model 1, a stacked Bi-LSTM, trained on all 90 races."),
    ("Explain the code",
     "Seven documented modules forming a reproducible pipeline."),
    ("Results, accuracy and loss graphs",
     "Reported on the held-out 2024 season, never seen in training."),
    ("Confusion matrix",
     "With row-normalised percentages and interpretation."),
], gap=680000, hsize=10.5, bsize=9)

head(s, 6_400_000, 1_950_000, 5_130_000, "Delivered")
for i, (v, l) in enumerate([("90", "Races processed"), ("98,357", "Driver-laps"),
                            ("55", "Engineered features"), ("81,270", "Training windows"),
                            ("0.806", "Test ROC-AUC"), ("5", "Figures produced")]):
    x = 6_400_000 + (i % 3) * 1_760_000
    y = 2_300_000 + (i // 3) * 1_140_000
    statcard(s, x, y, 1_600_000, 980_000, v, l)
callout(s, "Scope note:  Review 3 requires all three models compared. Every component below — the dataset, "
           "the splits, the loss, the threshold procedure and the metrics — is fixed and shared, so Models 2 "
           "and 3 plug into the same pipeline and are scored identically.",
        y=5_400_000, h=600_000)

# ------------------------------------------------------------ 3 pipeline ----
s = add_slide(prs)
badge(s, "02 — PIPELINE")
title(s, "From Proposal to Working System")
intro(s, "Five stages turn the official F1 timing feed into a trained model and a set of "
         "evaluation artefacts. Each stage is a separate module, and each was committed and "
         "pushed independently.")
stages = [("01", "Acquire", "FastF1 API.\n90 race sessions\ndownloaded and cached."),
          ("02", "Reduce", "Telemetry aggregated\nper lap at download:\n3.7 Hz → 8 channels."),
          ("03", "Engineer", "55 features across six\ngroups, plus the\nleak-free pit label."),
          ("04", "Window", "Sliding 10-lap windows\nper driver → 81,270\nlabelled sequences."),
          ("05", "Learn", "Bi-LSTM trained,\nthreshold tuned,\nevaluated on 2024.")]
bw, bgap = 2_000_000, 2_210_000
for i, (num, name, desc) in enumerate(stages):
    x = LEFT + i * bgap
    rrect(s, x, 2_030_000, bw, 2_250_000, CARD, adj=0.05, line=CHIP_MID)
    rrect(s, x + 200000, 2_240_000, 420_000, 320_000, CHIP, adj=0.22)
    text(s, x + 200000, 2_300_000, 420_000, 220000, num, 9, INK, H_FONT, align=PP_ALIGN.CENTER)
    text(s, x + 200000, 2_680_000, bw - 400000, 250000, name, 13, INK, H_FONT)
    text(s, x + 200000, 3_010_000, bw - 400000, 1_100_000, desc, 9, BODY, B_FONT, line_pct=1.30)

head(s, LEFT, 4_560_000, CONTENT_W, "Engineering Decisions Behind the Pipeline")
bullet_rows(s, LEFT, 4_900_000, 5_200_000, [
    ("Telemetry reduced, not stored",
     "Raw 3.7 Hz car data across 90 races is tens of gigabytes. Binning samples into laps with a single "
     "vectorised pass costs 0.25 s per race."),
], gap=600000, hsize=11, bsize=9.5, body_h=520000)
bullet_rows(s, 6_400_000, 4_900_000, 5_130_000, [
    ("Split by season, never by row",
     "Laps from one race are highly correlated. A random split would place near-duplicate laps in both "
     "training and test and inflate the score."),
], gap=600000, hsize=11, bsize=9.5, body_h=520000)

# ------------------------------------------------- 4 problem formulation ----
s = add_slide(prs)
badge(s, "03 — PROBLEM FORMULATION")
title(s, "The Prediction Task and Its Central Difficulty")
intro(s, "Given the last 10 laps of telemetry for one driver, predict whether that driver will "
         "pit within the next 3 laps. A binary sequence-classification problem — and a severely "
         "imbalanced one.")
head(s, LEFT, 1_950_000, 5_130_000, "Why a Three-Lap Horizon")
rrect(s, LEFT, 2_300_000, 5_130_000, 1_560_000, CARD, adj=0.05, line=CHIP_MID)
text(s, LEFT + 250000, 2_470_000, 4_650_000, 1_250_000,
     "A driver pits only 2–3 times in a ~55-lap race, so \"pits on this exact lap\" is roughly 5% "
     "positive. Widening the target to a three-lap window raises the positive rate to a trainable "
     "8.9% — and matches how far ahead a real strategy call is actually made.",
     10, BODY, B_FONT, line_pct=1.32)
head(s, LEFT, 4_060_000, 5_130_000, "Consequences for Evaluation")
bullet_rows(s, LEFT, 4_400_000, 5_130_000, [
    ("Accuracy is not the headline metric",
     "A model that never predicts a pit stop scores 91.5% accuracy on the test set."),
    ("Positives are up-weighted in the loss",
     "One upcoming stop counts as much as roughly ten quiet laps."),
], gap=620000, hsize=10.5, bsize=9.5)

head(s, 6_400_000, 1_950_000, 5_130_000, "The Imbalance, in Numbers")
for i, (v, l, c) in enumerate([("4.9%", "Pits on this exact lap", CHIP),
                               ("8.9%", "Pits within 3 laps (our label)", CHIP),
                               ("91.5%", "Accuracy of a useless model", CHIP),
                               ("9.93", "Positive class weight applied", CHIP)]):
    x = 6_400_000 + (i % 2) * 2_650_000
    y = 2_300_000 + (i // 2) * 1_180_000
    statcard(s, x, y, 2_480_000, 1_020_000, v, l, fill=c, vsize=21)
rrect(s, 6_400_000, 4_720_000, 5_130_000, 1_020_000, CARD, adj=0.05, line=CHIP_MID)
text(s, 6_400_000 + 250000, 4_880_000, 4_650_000, 760_000,
     "Reported metrics therefore include precision, recall, F1 and PR-AUC alongside accuracy, "
     "and every result is shown against a \"never pit\" baseline.",
     10, BODY, B_FONT, line_pct=1.30)

# -------------------------------------------------------------- 5 dataset ---
s = add_slide(prs)
badge(s, "04 — DATASET")
title(s, "Multi-Season Dataset Built from FastF1")
intro(s, "Every race session from 2021 to 2024 was downloaded, cleaned, temporally aligned and "
         "reduced to a driver-lap table. The FastF1 API permits 500 calls per hour and one race "
         "costs about nine, so the sweep was paced and resumed across several windows.")
for i, (v, l) in enumerate([("90", "Race events"), ("98,357", "Driver-laps"),
                            ("55", "Columns per lap"), ("81,270", "Sequence windows"),
                            ("4", "Seasons (2021–24)"), ("~12 GB", "Cached telemetry")]):
    statcard(s, LEFT + i * 1_830_000, 1_950_000, 1_700_000, 980_000, v, l,
             vsize=16 if len(v) > 5 else 18)

head(s, LEFT, 3_180_000, 5_130_000, "Split by Season — No Race Appears Twice")
table(s, LEFT, 3_520_000, 5_130_000,
      [["Split", "Seasons", "Races", "Windows", "Positive"],
       ["Train", "2021–22", "43", "38,890", "9.15%"],
       ["Validation", "2023", "22", "20,152", "9.55%"],
       ["Test", "2024", "24", "22,228", "8.52%"]],
      col_w=[1_330_000, 1_010_000, 790_000, 1_070_000, 950_000],
      row_h=300000, hdr_h=310000, bold_col0=True)

head(s, 6_400_000, 3_180_000, 5_130_000, "Data Quality Handled by the Pipeline")
bullet_rows(s, 6_400_000, 3_520_000, 5_130_000, [
    ("2021 Belgian GP removed automatically",
     "Abandoned after three laps behind the safety car in the rain, so no driver reached the 13 laps "
     "a labelled window requires."),
    ("Timing glitches corrected",
     "Duplicate lap-start timestamps are dropped together with their lap label, keeping bins and labels "
     "aligned rather than silently shifting every later lap."),
], gap=900000, hsize=10.5, bsize=9.5, body_h=620000)
callout(s, "Cross-driver context:  each lap also carries gap to leader, intervals to the cars ahead and behind, "
           "how much of the field has already stopped, and this driver's tyre age relative to the field — the "
           "hand-built precursor to the cross-driver attention planned for Model 3.",
        y=5_640_000, h=600_000)

# ----------------------------------------------------- 6 feature groups -----
s = add_slide(prs)
badge(s, "05 — FEATURE ENGINEERING")
title(s, "Fifty-Five Features Across Six Groups")
intro(s, "Every feature is computed per driver per lap, inside the driver's own race, so nothing "
         "crosses a race or driver boundary.")
groups = [("Timing & pace", "9",
           "Lap time, delta to stint median, lap-to-lap difference, rolling mean and spread, "
           "sector splits, and a rolling degradation slope."),
          ("Tyre & stint", "12",
           "Compound one-hot, tyre life, stint number, fresh-tyre flag, pit-in and pit-out flags, "
           "and stops completed so far."),
          ("Race position", "6",
           "Track position, grid position, positions gained, lap number, race fraction and laps "
           "remaining."),
          ("Track status", "4",
           "Yellow flag, safety car, virtual safety car and red flag, parsed from FastF1's "
           "concatenated status codes."),
          ("Weather & telemetry", "13",
           "Air and track temperature, humidity, pressure, wind, rainfall; plus speed, throttle, "
           "brake, RPM and gear aggregated per lap."),
          ("Competitor context", "7",
           "Gap to leader, intervals ahead and behind, rivals pitting this lap, field mean tyre "
           "age and this driver's tyre age relative to it.")]
for i, (name, n, desc) in enumerate(groups):
    x = LEFT + (i % 3) * 3_622_915
    y = 1_900_000 + (i // 3) * 1_700_000
    rrect(s, x, y, 3_382_915, 1_500_000, CARD, adj=0.05, line=CHIP_MID)
    rrect(s, x + 220000, y + 210000, 450_000, 300_000, CHIP, adj=0.22)
    text(s, x + 220000, y + 262000, 450_000, 210000, n, 9.5, INK, H_FONT, align=PP_ALIGN.CENTER)
    text(s, x + 760000, y + 250000, 2_400_000, 250000, name, 11, INK, H_FONT)
    text(s, x + 220000, y + 620000, 2_950_000, 780_000, desc, 8.5, BODY, B_FONT, line_pct=1.26)

callout(s, "Degradation slope:  a rolling least-squares fit of lap time against tyre age within the current "
           "stint, computed as cov(x,y)/var(x) so it stays a few vector operations. A steepening slope is "
           "precisely the signal a race engineer watches before calling a driver in.",
        y=5_320_000, h=600_000)

# ------------------------------------------------------- 7 architecture -----
s = add_slide(prs)
badge(s, "06 — MODEL ARCHITECTURE")
title(s, "Model 1 — Stacked Bidirectional LSTM")
intro(s, "The input window is a closed 10-lap block already in the past at prediction time, not a "
         "live stream, so the network is free to read it in both directions — a stint's degradation "
         "trend reads more clearly backwards than forwards.")
head(s, LEFT, 1_950_000, 6_000_000, "Layer Stack")
table(s, LEFT, 2_300_000, 6_000_000,
      [["Layer", "Output shape", "Params"],
       ["Input — lap window", "(10, 55)", "0"],
       ["Bidirectional LSTM (64)", "(10, 128)", "61,440"],
       ["Dropout 0.4", "(10, 128)", "0"],
       ["Bidirectional LSTM (32)", "(64)", "41,216"],
       ["Dropout 0.4", "(64)", "0"],
       ["Dense 32 — ReLU", "(32)", "2,080"],
       ["Dense 1 — Sigmoid", "(1)", "33"],
       ["Total trainable", "", "104,769"]],
      col_w=[2_800_000, 1_800_000, 1_400_000],
      row_h=272000, hdr_h=290000, fsize=9.5, bold_col0=False, highlight_rows=(8,))

head(s, 7_000_000, 1_950_000, 4_530_000, "Design Choices")
bullet_rows(s, 7_000_000, 2_300_000, 4_530_000, [
    ("Bidirectional, not causal",
     "The window is historical, so both directions are legitimate."),
    ("Narrowing 64 → 32 → 32",
     "The first layer returns the sequence; the second compresses it to one vector per window."),
    ("Dropout 0.4 with light L2",
     "Chosen from a six-configuration sweep, not by default."),
    ("Weighted cross-entropy",
     "Applied inside the loss rather than via class_weight, so training and validation curves stay on "
     "one comparable scale."),
], gap=790000, hsize=10.5, bsize=9, body_h=520000)

# ---------------------------------------------------------- 8 training ------
s = add_slide(prs)
badge(s, "07 — TRAINING PROTOCOL")
title(s, "How the Model Was Trained and Calibrated")
intro(s, "The protocol is fixed and shared: whatever varies between Models 1, 2 and 3, none of "
         "this does. That is what makes the Review 3 comparison a fair one.")
items = [("Optimiser", "Adam, learning rate 5×10⁻⁴, batch size 256."),
         ("Loss", "Weighted binary cross-entropy, positives scaled ×9.93."),
         ("Early stopping", "On validation PR-AUC, not validation loss — under imbalance the loss can "
                            "improve while the model gets worse at finding stops."),
         ("Threshold", "Tuned on validation to maximise F1, then applied unchanged to test."),
         ("Reproducibility", "Python, NumPy and TensorFlow all seeded; 11 epochs in 53 seconds on an RTX 3050.")]
for i, (k, v) in enumerate(items):
    y = 1_950_000 + i * 700_000
    rrect(s, LEFT, y, 5_600_000, 600_000, CARD, adj=0.06, line=CHIP_MID)
    text(s, LEFT + 230000, y + 105000, 1_500_000, 250000, k, 10.5, INK, H_FONT)
    text(s, LEFT + 1_850_000, y + 105000, 3_600_000, 420_000, v, 9, BODY, B_FONT, line_pct=1.25)

head(s, 6_620_000, 1_950_000, 4_930_000, "Guarding Against Leakage")
rrect(s, 6_620_000, 2_300_000, 4_930_000, 2_180_000, CARD, adj=0.04, line=CHIP_MID)
bullet_rows(s, 6_620_000 + 250000, 2_490_000, 4_430_000, [
    ("Label shifted forward",
     "Lap t's own pit stop can never enter lap t's own label."),
    ("Season-based splits",
     "Verified: zero race overlap between train, validation and test."),
    ("Scaler fit on train only",
     "No test statistics reach training."),
], gap=640000, hsize=10, bsize=9, body_h=330000)
callout(s, "Verification:  the label was checked against a brute-force re-derivation over every driver-race. "
           "Verstappen pits on lap 14 at Bahrain — laps 11, 12 and 13 are labelled 1, lap 14 itself is 0, and "
           "tyre age resets on lap 15.",
        y=5_560_000, h=600_000)

# ------------------------------------------- 9 accuracy and loss graphs -----
s = add_slide(prs)
badge(s, "08 — TRAINING CURVES")
title(s, "Accuracy and Loss per Epoch")
intro(s, "Training and validation losses sit on the same scale because the class weighting lives "
         "inside the loss function. The dashed line marks the epoch from which early stopping "
         "restored the weights.", h=360000)
picture(s, FIG / "bilstm_accuracy.png", LEFT, 1_800_000, h=3_100_000)
picture(s, FIG / "bilstm_loss.png", 6_261_475, 1_800_000, h=3_100_000)
caption(s, LEFT, 5_020_000, 5_270_000, "Accuracy per epoch — training vs. validation")
caption(s, 6_261_475, 5_020_000, 4_668_000, "Weighted cross-entropy per epoch")
callout(s, "Reading the curves:  validation loss rises from epoch 1 while training loss keeps falling — textbook "
           "overfitting. The cause is effective sample size: 38,890 training windows come from only 815 "
           "driver-races, and consecutive windows share 9 of their 10 laps.",
        y=5_400_000, h=640_000)

# ------------------------------------------------- 10 confusion matrix ------
s = add_slide(prs)
badge(s, "09 — CONFUSION MATRIX")
title(s, "Confusion Matrix on the Held-Out 2024 Season")
intro(s, "Cells show the raw count and that count as a share of its true class. The row "
         "percentages are what matter: in absolute terms the \"No pit\" row dwarfs everything else.",
     h=360000)
picture(s, FIG / "bilstm_confusion_matrix.png", LEFT, 1_800_000, h=4_150_000)
head(s, 6_500_000, 1_900_000, 5_030_000, "What Each Quadrant Means")
quads = [("831 stops correctly anticipated",
          "43.9% of the 1,894 real pit stops in the 2024 season were flagged in advance."),
         ("1,063 stops missed",
          "In a real race these are the expensive errors — the strategist gets no warning."),
         ("1,956 false alarms",
          "Comparatively cheap: a strategist who is warned and does not stop loses nothing."),
         ("18,378 quiet laps left alone",
          "90.4% of non-pit laps correctly identified, so the system is not crying wolf.")]
bullet_rows(s, 6_500_000, 2_260_000, 5_030_000, quads,
            gap=800000, hsize=10.5, bsize=9, body_h=430000)
rrect(s, 6_500_000, 5_540_000, 5_030_000, 700_000, CHIP, adj=0.08)
text(s, 6_500_000 + 230000, 5_680_000, 4_590_000, 440_000,
     "Shown at the tuned threshold 0.779. At 0.50 the model catches 72.5% of stops; the never-pit "
     "baseline catches none.",
     9.5, BODY, B_FONT, line_pct=1.25)

# ------------------------------------------------------- 11 roc and pr ------
s = add_slide(prs)
badge(s, "10 — THRESHOLD-INDEPENDENT CURVES")
title(s, "ROC and Precision-Recall Curves")
intro(s, "These score the model's ranking of laps independently of where the decision cut-off is "
         "drawn. On an imbalanced problem the precision-recall curve is the honest one — its "
         "baseline is the positive rate, not a diagonal.", h=360000)
picture(s, FIG / "bilstm_roc.png", 900_000, 1_880_000, h=3_700_000)
picture(s, FIG / "bilstm_precision_recall.png", 5_100_000, 1_880_000, h=3_700_000)
caption(s, 900_000, 5_660_000, 3_930_000, "ROC curve — ROC-AUC 0.806 against a 0.500 chance line")
caption(s, 5_100_000, 5_660_000, 3_930_000, "Precision-recall — PR-AUC 0.323 against a 0.085 base rate")
for i, (v, l) in enumerate([("0.806", "ROC-AUC"), ("0.323", "PR-AUC"), ("3.8×", "Above base rate")]):
    statcard(s, 9_500_000, 1_950_000 + i * 1_180_000, 2_030_000, 1_020_000, v, l, vsize=20)

# --------------------------------------------------------- 12 results -------
s = add_slide(prs)
badge(s, "11 — RESULTS")
title(s, "Model 1 Performance on the 2024 Season")
intro(s, "22,228 windows from 24 races, 8.52% positive. The test set was never used for training, "
         "validation or threshold selection.")
table(s, LEFT, 1_900_000, CONTENT_W,
      [["Model", "Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
       ["Never-pit baseline", "0.9148", "0.5000", "0.000", "0.000", "0.000", "0.500", "0.085"],
       ["Bi-LSTM @ threshold 0.50", "0.7285", "0.7271", "0.199", "0.725", "0.313", "0.806", "0.323"],
       ["Bi-LSTM @ tuned 0.779", "0.8642", "0.6713", "0.298", "0.439", "0.355", "0.806", "0.323"]],
      col_w=[3_150_000, 1_180_000, 1_380_000, 1_180_000, 1_050_000, 950_000, 1_180_000, 800_000],
      row_h=340000, hdr_h=360000, fsize=9.75, bold_col0=True, highlight_rows=(2,))

head(s, LEFT, 3_450_000, 5_130_000, "The Baseline Is the Argument")
rrect(s, LEFT, 3_790_000, 5_130_000, 1_620_000, CARD, adj=0.05, line=CHIP_MID)
text(s, LEFT + 250000, 3_960_000, 4_650_000, 1_300_000,
     "The never-pit baseline wins on raw accuracy — 91.5% — and scores exactly zero on precision, "
     "recall and F1. It never once calls a stop. This single row is why accuracy cannot be the "
     "headline metric for this problem.",
     10, BODY, B_FONT, line_pct=1.32)

head(s, 6_400_000, 3_450_000, 5_130_000, "Which Operating Point to Use")
rrect(s, 6_400_000, 3_790_000, 5_130_000, 1_620_000, CARD, adj=0.05, line=CHIP_MID)
text(s, 6_400_000 + 250000, 3_960_000, 4_650_000, 1_300_000,
     "F1-optimal tuning pushes the threshold to 0.779 and trades recall (0.725 → 0.439) for "
     "precision. For a race engineer that is likely the wrong trade — a missed stop is expensive, "
     "a false alarm is not — so the 0.50 point, catching 72.5% of stops, is arguably more useful. "
     "Both are reported rather than silently choosing.",
     10, BODY, B_FONT, line_pct=1.32)

# ------------------------------------------------- 13 honest comparison -----
s = add_slide(prs)
badge(s, "12 — CRITICAL ANALYSIS")
title(s, "Is a Deep Sequence Model Justified Yet?")
intro(s, "A deep model should be justified, not assumed. Two classical models were trained on "
         "exactly the same tensors and scored the same way.")
table(s, LEFT, 1_900_000, 6_100_000,
      [["Model", "ROC-AUC", "PR-AUC"],
       ["Random (base rate)", "0.500", "0.085"],
       ["Logistic regression — final lap only", "0.802", "0.308"],
       ["Gradient boosting — flattened window", "0.853", "0.391"],
       ["Model 1 — Bi-LSTM", "0.806", "0.323"]],
      col_w=[3_500_000, 1_300_000, 1_300_000],
      row_h=310000, hdr_h=330000, bold_col0=True, highlight_rows=(3,))
text(s, LEFT, 3_620_000, 6_100_000, 300000,
     "Answer, stated honestly: not yet.", 13, ACCENT, H_FONT)
text(s, LEFT, 3_960_000, 6_100_000, 1_100_000,
     "Gradient boosting currently outperforms the Bi-LSTM, and logistic regression on the final "
     "lap alone roughly matches it. Model 1 is not yet earning its complexity — which sets a real "
     "bar for Models 2 and 3 to clear, rather than a trivial one.",
     10, BODY, B_FONT, line_pct=1.32)

head(s, 7_000_000, 1_900_000, 4_530_000, "Why — and It Is Not Capacity")
bullet_rows(s, 7_000_000, 2_250_000, 4_530_000, [
    ("Effective sample size is small",
     "38,890 windows come from just 815 driver-races across 43 races, and consecutive windows share "
     "9 of their 10 laps."),
    ("Six architectures, one ceiling",
     "From 104k down to 10k parameters, varying dropout, L2 and recurrent dropout, validation PR-AUC "
     "moved only between 0.288 and 0.320."),
    ("More data still helps",
     "Training on 2021+2022 (0.314) beats 2022 alone (0.285) and 2021 alone (0.237), so the 2022 "
     "regulation change is not the dominant effect."),
], gap=1_020_000, hsize=10.5, bsize=9, body_h=700000)

# ---------------------------------------------------- 14 challenges ---------
s = add_slide(prs)
badge(s, "13 — ENGINEERING CHALLENGES")
title(s, "Three Problems Found and Solved")
intro(s, "Each of these cost real time and each changed the code. They are reported because the "
         "diagnosis, not just the result, is part of the work.")
probs = [("API rate limiting",
          "FastF1 permits 500 calls per hour and one race costs about nine, so a full 2021–24 sweep "
          "cannot finish inside a single window. Early sweeps lost whole blocks of races.",
          "Rate-limit errors are now caught separately from transient ones and wait for the rolling "
          "window to free up; downloads are paced and resume from cache."),
         ("A silent loss-function bug",
          "The first full run produced ROC-AUC 0.4954 — pure chance — with the loss frozen at 1.2595 "
          "every epoch. Keras passes y_true as (batch,) but y_pred as (batch, 1); multiplying them "
          "broadcast to a (batch, batch) outer product. Nothing raised an error.",
          "Found by proving the data clean, then showing gradient boosting reached 0.849 on identical "
          "tensors. Fixed by flattening both tensors: 0.531 → 0.760 on the same seed."),
         ("Corrupt lap timestamps",
          "Timing glitches occasionally give two laps the same start time, which breaks the telemetry "
          "binning. The obvious fix would misalign every later lap.",
          "The duplicate edge and its lap label are dropped together, keeping bins and labels in step.")]
for i, (name, problem, fix) in enumerate(probs):
    y = 1_990_000 + i * 1_230_000
    rrect(s, LEFT, y, CONTENT_W, 1_130_000, CARD, adj=0.04, line=CHIP_MID)
    text(s, LEFT + 240000, y + 175000, 2_500_000, 500_000, name, 11.5, INK, H_FONT, line_pct=1.15)
    text(s, LEFT + 3_020_000, y + 175000, 3_870_000, 1_000_000, problem, 8.5, BODY, B_FONT, line_pct=1.24)
    text(s, LEFT + 7_120_000, y + 175000, 3_500_000, 1_000_000, fix, 8.5, BODY, B_FONT, line_pct=1.24)
text(s, LEFT + 3_020_000, 1_770_000, 2_000_000, 200000, "PROBLEM", 8, BODY, B_FONT)
text(s, LEFT + 7_120_000, 1_770_000, 2_000_000, 200000, "RESOLUTION", 8, BODY, B_FONT)
callout(s, "A regression test suite now covers the loss: its shape contract, its value against the hand-computed "
           "definition, the weighting ratio and serialisation. The bug was completely silent, so the shape "
           "assertion is the important one.",
        y=5_705_078, h=648_097)

# --------------------------------------------------------- 15 conclusion ----
s = add_slide(prs)
badge(s, "14 — CONCLUSION")
title(s, "What Review 2 Establishes")
intro(s, "A complete, reproducible pipeline from the official F1 timing feed to a trained, "
         "evaluated sequence model — and an honest account of where it currently stands.")
concl = [("A working end-to-end system",
          "90 races acquired, 55 features engineered, 81,270 leak-free sequences built, and a "
          "Bi-LSTM trained and evaluated on a season it never saw."),
         ("A model that genuinely learns",
          "ROC-AUC 0.806 and PR-AUC 0.323 against a 0.085 base rate — 3.8× better than chance, "
          "catching 72.5% of pit stops at the 0.50 operating point."),
         ("Evaluation that cannot flatter itself",
          "Season-based splits with verified zero race overlap, a scaler fit on training data only, "
          "a forward-shifted label, and a never-pit baseline reported beside every result."),
         ("A measured, defensible limitation",
          "Gradient boosting still leads on PR-AUC. The constraint is effective sample size rather "
          "than model capacity, established by a six-configuration sweep."),
         ("Infrastructure the remaining models inherit",
          "Data, splits, loss, threshold procedure and metrics are fixed and shared, so Models 2 "
          "and 3 are scored identically with no further data collection."),
         ("Work version-controlled throughout",
          "Seven commits pushed to GitHub, one per phase, with a regression test suite guarding "
          "the loss function.")]
for i, (h_, b_) in enumerate(concl):
    x = LEFT + (i % 2) * 5_600_000
    y = 1_900_000 + (i // 2) * 1_420_000
    rrect(s, x, y, 5_268_745, 1_240_000, CARD, adj=0.05, line=CHIP_MID)
    text(s, x + 240000, y + 185000, 4_790_000, 280000, h_, 11.5, INK, H_FONT)
    text(s, x + 240000, y + 530000, 4_790_000, 640_000, b_, 9, BODY, B_FONT, line_pct=1.28)

# -------------------------------------------------------- 16 next steps -----
s = add_slide(prs)
badge(s, "15 — NEXT STEPS")
title(s, "Road to Review 3 — Three Models Compared")
intro(s, "Review 3 requires all three models compared on accuracy, loss curves and confusion "
         "matrices. Model 1 is complete; the remaining two reuse this pipeline unchanged.")
models = [("Model 1", "Bi-LSTM", "Sequential baseline, as benchmarked by Sasikumar et al.",
           "Complete", CHIP),
          ("Model 2", "CNN–BiLSTM", "Convolutional feature extraction ahead of the recurrent stack, "
           "to catch short-term degradation patterns the LSTM smooths over.", "Phase 4", CARD),
          ("Model 3", "Multi-task Transformer\nwith cross-driver attention",
           "The project's contribution: four strategic heads sharing one representation, with "
           "attention across all drivers on track.", "Phase 5", CARD)]
for i, (num, name, desc, status, fill) in enumerate(models):
    x = LEFT + i * 3_622_915
    rrect(s, x, 1_950_000, 3_382_915, 2_500_000, fill, adj=0.04,
          line=None if fill is CHIP else CHIP_MID)
    text(s, x + 240000, 2_140_000, 2_900_000, 220000, num, 8.5, BODY, B_FONT)
    text(s, x + 240000, 2_420_000, 2_900_000, 620_000, name, 12.5, INK, H_FONT, line_pct=1.12)
    text(s, x + 240000, 3_180_000, 2_900_000, 900_000, desc, 8.5, BODY, B_FONT, line_pct=1.26)
    rrect(s, x + 240000, 4_060_000, 1_150_000, 270_000, WHITE if fill is CHIP else CHIP, adj=0.22)
    text(s, x + 240000, 4_118_000, 1_150_000, 200000, status, 8, INK, B_FONT, align=PP_ALIGN.CENTER)

head(s, LEFT, 4_700_000, CONTENT_W, "The Bar Model 3 Has to Clear")
bullet_rows(s, LEFT, 5_030_000, 5_500_000, [
    ("Beat PR-AUC 0.391, not 0.085",
     "Gradient boosting, not the trivial baseline, is the number to exceed — a far stronger claim if "
     "the proposed architecture achieves it."),
], gap=600000, hsize=11, bsize=9.5, body_h=520000)
bullet_rows(s, 6_400_000, 5_030_000, 5_130_000, [
    ("Report efficiency, not just accuracy",
     "Parameter count, training time and per-lap inference latency, since the system's premise is "
     "real-time operation."),
], gap=600000, hsize=11, bsize=9.5, body_h=520000)

# -------------------------------------------------------- 17 references -----
s = add_slide(prs)
badge(s, "REFERENCES")
title(s, "References and Project Repository")
refs = [("FastF1 Documentation", "Official library for programmatic access to F1 timing, telemetry, "
         "position and weather data.  docs.fastf1.dev"),
        ("Sasikumar, A., Leema, A. A., & Balakrishnan, P. (2025)",
         "Data-driven pit stop decision support for Formula 1 using deep learning models. "
         "Frontiers in Artificial Intelligence, 8, 1673148."),
        ("Thomas, D., et al. (2025)",
         "Explainable reinforcement learning for Formula One race strategy. "
         "Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing."),
        ("Veerasagar, S. S., Coelho, R., & Chandrashekar, L. K. (2025)",
         "Optimum racing: A F1 strategy predictor using reinforcement learning. IJRASET."),
        ("Vaswani, A., et al. (2017)",
         "Attention is all you need. Advances in Neural Information Processing Systems, 30."),
        ("Chollet, F., et al. (2015)",
         "Keras. Deep learning framework used for all model implementation in this review.")]
for i, (who, what) in enumerate(refs):
    y = 1_700_000 + i * 630_000
    text(s, LEFT, y, 4_200_000, 520_000, who, 10, INK, H_FONT, line_pct=1.20)
    text(s, LEFT + 4_400_000, y, 6_468_745, 520_000, what, 9, BODY, B_FONT, line_pct=1.25)

rrect(s, LEFT, 5_620_000, CONTENT_W, 740_000, CHIP, adj=0.06)
text(s, LEFT + 300000, 5_760_000, 5_000_000, 260000, "Project Repository", 11, INK, H_FONT)
text(s, LEFT + 300000, 6_040_000, 6_000_000, 260000,
     "github.com/dhruv-deepak/f1-virtual-race-engineer", 10.5, BODY, M_FONT)
text(s, LEFT + 7_900_000, 5_790_000, 3_200_000, 480_000,
     "All code, figures, metrics and the\nexecuted Review 2 notebook.", 9, BODY, B_FONT, line_pct=1.28)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"saved {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
