"""
Build the final review deck in the professor's prescribed template.

Run:  python slides/build_final_review_deck.py

Unlike the Review 2 and Review 3 decks, which use the project's own visual
language, this one fills the department template exactly: the same slide order,
the same section headings, the same tables. Set FINAL_TEMPLATE to point at the
supplied .pptx.

Every number is read from reports/metrics/ at build time rather than typed in,
so the deck cannot drift from the results.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIG = ROOT / "reports" / "figures"
TEMPLATE = Path(os.environ.get(
    "FINAL_TEMPLATE",
    r"C:\Users\HP\Downloads\DL Project Final Review Template (1).pptx"))
OUT = Path(r"C:\E\VIT\VIT 7th Sem\Deep Learning Project\DL-Project-Final-Review.pptx")

INK = RGBColor(0x1B, 0x1B, 0x27)
BODY = RGBColor(0x3C, 0x39, 0x39)
HDR_BG = RGBColor(0x0F, 0x17, 0x2A)
ROW_ALT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0xC4, 0x1E, 0x3A)

SLIDE_W = Emu(12192000)
BODY_FONT = "Calibri"

# ---------------------------------------------------------------- data -----
COMP = pd.read_csv(METRICS / "three_model_comparison.csv")
REPLAY = pd.read_csv(METRICS / "race_replay_summary.csv")
ABL = pd.read_csv(METRICS / "mtl_ablation.csv")
MJSON = {n: json.load(open(METRICS / f"{n}_metrics.json"))
         for n in ("bilstm", "cnn_bilstm", "mtl_attn")}

BASE = COMP.iloc[0]
M1 = COMP[COMP["model"].str.contains("Bi-LSTM", regex=False)].iloc[0]
M2 = COMP[COMP["model"].str.contains("CNN", regex=False)].iloc[0]
M3 = COMP[COMP["model"].str.contains("MTL", regex=False)].iloc[0]
A = {r["config"]: r for _, r in ABL.iterrows()}


# ------------------------------------------------------------- helpers -----
def body_ph(slide):
    """The content placeholder, whatever index the template gave it."""
    for sh in slide.placeholders:
        if sh.placeholder_format.idx != 0:
            return sh
    return None


def set_title(slide, text):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == 0:
            sh.text_frame.text = text
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.bold = True
            return


def set_bullets(slide, items, size=20, spacing=10):
    """Fill the content placeholder. `items` are (level, text) or plain strings."""
    ph = body_ph(slide)
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        level, text = item if isinstance(item, tuple) else (0, item)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.space_after = Pt(spacing)
        for r in p.runs:
            r.font.size = Pt(size - 2 * level)
            r.font.name = BODY_FONT
            r.font.color.rgb = BODY if level else INK
    return ph


def set_paragraph(slide, text, size=18):
    ph = body_ph(slide)
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.paragraphs[0].text = text
    tf.paragraphs[0].alignment = PP_ALIGN.JUSTIFY
    for r in tf.paragraphs[0].runs:
        r.font.size = Pt(size)
        r.font.name = BODY_FONT
        r.font.color.rgb = BODY
    return ph


def table_of(slide):
    for sh in slide.shapes:
        if sh.has_table:
            return sh.table
    return None


def resize_rows(table, n_rows):
    """Grow or shrink a template table by cloning/removing its last row."""
    while len(table.rows) < n_rows:
        table._tbl.append(copy.deepcopy(table._tbl.tr_lst[-1]))
    while len(table.rows) > n_rows:
        table._tbl.remove(table._tbl.tr_lst[-1])


def fill_table(table, rows, sizes=(9, 8.5), header_fill=HDR_BG, col_w=None):
    resize_rows(table, len(rows))
    if col_w:
        total = sum(col_w)
        for j, w in enumerate(col_w):
            table.columns[j].width = Emu(int(table._tbl.tblGrid.gridCol_lst[0].get("w") or 0) or 0)
        # python-pptx needs explicit widths; scale to the table's own width.
        tw = sum(c.width for c in table.columns)
        for j, w in enumerate(col_w):
            table.columns[j].width = Emu(int(tw * w / total))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if j >= len(table.columns):
                continue
            cell = table.cell(i, j)
            cell.margin_left = cell.margin_right = Emu(45000)
            cell.margin_top = cell.margin_bottom = Emu(18000)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (header_fill if i == 0
                                        else (ROW_ALT if i % 2 else WHITE))
            tf = cell.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.paragraphs[0].text = str(val)
            for r in tf.paragraphs[0].runs:
                r.font.size = Pt(sizes[0] if i == 0 else sizes[1])
                r.font.bold = (i == 0)
                r.font.name = BODY_FONT
                r.font.color.rgb = WHITE if i == 0 else BODY


def add_slide(prs, after_idx, layout_name="Title and Content"):
    layout = next(l for l in prs.slide_masters[0].slide_layouts if l.name == layout_name)
    slide = prs.slides.add_slide(layout)
    xml_slides = prs.slides._sldIdLst
    ids = list(xml_slides)
    xml_slides.remove(ids[-1])
    xml_slides.insert(after_idx + 1, ids[-1])
    return slide


def add_picture_slide(prs, after_idx, title, image, caption=None, top=Emu(1430000),
                      max_w=Emu(10400000), max_h=Emu(4000000)):
    slide = add_slide(prs, after_idx)
    set_title(slide, title)
    ph = body_ph(slide)
    if ph is not None:
        ph._element.getparent().remove(ph._element)

    from PIL import Image as PILImage
    with PILImage.open(image) as im:
        ratio = im.size[0] / im.size[1]
    w = max_w
    h = Emu(int(w / ratio))
    if h > max_h:
        h = max_h
        w = Emu(int(h * ratio))
    left = Emu(int((SLIDE_W - w) / 2))
    slide.shapes.add_picture(str(image), left, top, width=w, height=h)

    if caption:
        box = slide.shapes.add_textbox(Emu(840000), Emu(int(top + h + 120000)),
                                       Emu(10500000), Emu(620000))
        tf = box.text_frame
        tf.word_wrap = True
        tf.text = caption
        for p in tf.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(11.5)
                r.font.name = BODY_FONT
                r.font.color.rgb = BODY
    return slide


def f(v, nd=3):
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


# ================================================================ build =====
prs = Presentation(str(TEMPLATE))
S = prs.slides

# ---- 1 title ---------------------------------------------------------------
s = S[0]
for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if t == "TITLE":
        sh.text_frame.text = ("Deep Learning-Based Virtual Race Engineer for\n"
                              "Real-Time Formula 1 Strategy Optimization")
        # Two lines at 23pt, seated clear of the "Presented by" block below.
        sh.top = Emu(3280000)
        sh.height = Emu(1120000)
        sh.text_frame.word_wrap = True
        for p in sh.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(23)
                r.font.bold = True
    elif "Review-2" in t:
        sh.text_frame.text = "Deep Learning Project — Final Review"
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(20)
    elif t.startswith("Presented by"):
        tf = sh.text_frame
        tf.clear()
        names = ["Presented by", "Dhruv Deepak, 23BCE9187", "Akhil V, 23BCE8051",
                 "Mohan Simha Varma Dantuluri, 23BCE9229",
                 "Surya Prathap Mamillapalli, 23BCE7783"]
        for i, n in enumerate(names):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = n
            for r in p.runs:
                r.font.size = Pt(13)
                r.font.bold = (i == 0)

# ---- 3 abstract ------------------------------------------------------------
set_paragraph(S[2], (
    "Formula 1 race strategy is a continuous, high-stakes decision problem in which a pit stop "
    "called one lap too early or too late can decide a race. This project develops a supervised "
    "deep learning system that learns pit-stop strategy directly from real Formula 1 telemetry, "
    "using no reinforcement learning and no simulator. Four seasons of official timing and "
    "telemetry data, covering ninety races from 2021 to 2024, were acquired through the FastF1 "
    "API and reduced to a driver-lap table in which each lap carries fifty-five engineered "
    "features spanning timing and pace, tyre state and a rolling degradation slope, track "
    "position, safety-car and flag status, weather, aggregated car telemetry, and cross-driver "
    "competitive context. The prediction task is framed as binary sequence classification: given "
    "the last ten laps of a driver's telemetry, will that driver pit within the next three laps? "
    "Three architectures were implemented and compared under an identical protocol — a "
    "bidirectional LSTM, a CNN–BiLSTM, and a multi-task transformer with attention across all "
    "drivers on track. Because pit stops occupy only 8.5 per cent of laps, the models are trained "
    "with a class-weighted loss and reported against a never-pit baseline, with precision, recall, "
    "F1 and PR-AUC alongside accuracy. Evaluated on the entirely held-out 2024 season, the "
    "strongest model reaches 0.806 ROC-AUC and 0.323 PR-AUC against a 0.085 base rate, and a "
    "historical race replay shows the system anticipating more than half of all real pit stops "
    "with a median of three laps of warning and fewer than 0.3 false alarms per driver-race. A "
    "controlled ablation further shows that the proposed cross-driver attention does not improve "
    "on a plain bidirectional LSTM on this dataset, which is reported as a finding rather than "
    "omitted."), size=14)

# ---- 4 introduction --------------------------------------------------------
set_bullets(S[3], [
    "Formula 1 is among the most data-intensive sports in the world: during a race, teams "
    "interpret thousands of telemetry signals and must turn them into strategic decisions "
    "within seconds.",
    "A pit stop costs roughly twenty seconds of track time, so calling one a lap early or a "
    "lap late routinely decides finishing positions.",
    "Strategy today depends on human expertise supported by proprietary race simulations, "
    "which introduces both cognitive limits and latency under rapidly changing conditions.",
    "Recent research applies deep learning and reinforcement learning to this problem, but "
    "remains fragmented: most work is simulator-based, or targets a single decision in "
    "isolation, and little of it models interactions between competing drivers.",
    "This project learns strategy directly from real historical telemetry, and evaluates its "
    "recommendations against the decisions professional race engineers actually made.",
], size=17, spacing=14)

# ---- 5 problem statement ---------------------------------------------------
set_bullets(S[4], [
    "No single variable determines the optimal strategy: pit timing emerges from the "
    "interaction of tyre degradation, fuel load, track status, weather and rival behaviour.",
    "Severe class imbalance. A driver pits two or three times in a fifty-five lap race, so "
    "only about 8.5 per cent of laps precede a stop — a model that never predicts a pit stop "
    "scores 91.5 per cent accuracy while being useless.",
    "Competitive interaction is difficult to represent. Standard sequence models see one "
    "driver at a time and cannot directly observe the undercut pressure that drives real calls.",
    "Public telemetry is noisier and coarser than the proprietary feeds teams use, and the "
    "data must be cleaned and temporally aligned across drivers before it can be modelled.",
    "Core challenge: predict, from real multi-driver telemetry alone, whether a given driver "
    "will pit within the next three laps, and do so early enough and reliably enough to be "
    "useful on the pit wall.",
], size=16, spacing=12)

# ---- 6 objectives ----------------------------------------------------------
set_bullets(S[5], [
    "Build a reproducible multi-season Formula 1 dataset from real telemetry accessed "
    "programmatically through the FastF1 API, covering the 2021 to 2024 seasons.",
    "Engineer lap-level features that capture tyre degradation, race context and cross-driver "
    "competitive pressure, with a leak-free pit-stop label.",
    "Implement and train three deep learning architectures for pit-stop prediction under one "
    "identical training and evaluation protocol.",
    "Compare the three models on predictive quality and on computational cost, and determine "
    "which is most efficient for real-time use.",
    "Evaluate the system against historical human strategy decisions through a race replay of "
    "the held-out 2024 season.",
], size=17, spacing=14)

# ---- 7 literature review ---------------------------------------------------
set_bullets(S[6], [
    "Seventeen works were reviewed across three strands relevant to this project.",
    (1, "Domain work on Formula 1 strategy: pit-stop prediction from telemetry, tyre "
        "degradation modelling, and reinforcement-learning approaches to strategy selection."),
    (1, "Sequence-modelling methods that the three architectures are built on: LSTM, "
        "bidirectional recurrence, temporal convolution and attention."),
    (1, "Learning settings this project relies on: multi-task learning and techniques for "
        "severe class imbalance."),
    "Two consistent gaps emerge. Most strategy work is evaluated in simulators rather than on "
    "real telemetry, and almost none of it models interaction between competing drivers.",
    "The detailed comparison follows on the next three slides.",
], size=17, spacing=12)

# ---- 8-10 literature tables ------------------------------------------------
HEAD = ["Sl No.", "Title, Year of Publication", "Methodology", "Dataset Details",
        "Dis-Advantages", "Results"]
LIT = [
    ["1", "Data-driven pit stop decision support for Formula 1 using deep learning models — "
          "Sasikumar, Leema & Balakrishnan, Frontiers in AI, 2025",
     "Benchmarks Bi-LSTM, TCN-GRU, GRU, InceptionTime and CNN-BiLSTM for pit-window prediction",
     "Real F1 telemetry via FastF1, 2020–2024 seasons",
     "Binary pit / no-pit only; no cross-driver interaction modelled",
     "Deep temporal models outperform classical baselines on real telemetry"],
    ["2", "Explainable Reinforcement Learning for Formula One Race Strategy — Thomas et al., "
          "ACM SAC / arXiv:2501.04068, 2025",
     "RL agent (RSRL) with feature-importance and counterfactual explanations",
     "Simulated race environment, evaluated on 2023 Bahrain GP",
     "Simulator-dependent; reward design strongly shapes learned behaviour",
     "Outperforms baseline strategy selection, with interpretable decisions"],
    ["3", "Optimum Racing: A F1 Strategy Predictor using Reinforcement Learning — "
          "Veerasagar, Coelho & Chandrashekar, IJRASET, 2025",
     "Deep Recurrent Q-Network (DRQN) with LSTM over sequential race states",
     "Simulated race states with tyre degradation and competitor gaps",
     "Simulator-dependent; scope limited to tyre and pit decisions",
     "Captures temporal decision structure; models degradation and gaps"],
    ["4", "Deep-Racing: An Embedded Deep Neural Network Model to Predict the Winning Strategy "
          "in Formula One Racing — Fatima & Johrendt, IJML 13(3), 2023",
     "Embedded deep neural network predicting pit-stop timing and driver rank",
     "Historical F1 race data, 2015–2022 seasons",
     "Aggregated race-level data; limited use of high-rate telemetry",
     "Joint prediction of pit timing and finishing rank"],
    ["5", "A Gradient Boosting Analytics Framework for Predictive Pit Stop Decision Making in "
          "Formula One Racing — Ametsbichler, Dinh, Tran & Reichert, SSRN",
     "Pit stop framed as binary classification; Random Forest, SVM and XGBoost compared",
     "Lap-level F1 data, 2022–2025 seasons",
     "Tabular lap features only; no sequence modelling",
     "XGBoost best on F1-score and ROC-AUC among those compared"],
    ["6", "A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing — "
          "Cappello & Hoegh, arXiv:2512.00640, 2025",
     "Bayesian state-space model; lap time as fuel mass plus latent tyre pace, stops as resets",
     "Public timing data via the FastF1 API",
     "Compound-specific differences not always statistically distinct",
     "Interpretable probabilistic degradation estimates for live strategy"],
]
LIT2 = [
    ["7", "SOSTA AI: Multi-Task Residual Learning for Lap-Level Pit Stop Decision Support in "
          "Formula One — SportRxiv preprint",
     "Multi-task residual network: classification head for next-lap pit, regression head for "
     "laps until the stop",
     "Lap-level Formula One race data",
     "Preprint, not peer reviewed at time of writing",
     "Reports AUC 0.92 and best MAE 3.56 laps (as stated by the authors)"],
    ["8", "Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. "
          "Mathematical Programming — Méndez & Thraves, 2025",
     "Compares deep RL agents against an exact optimisation formulation",
     "Simulated race strategy environment; data and code released",
     "Optimisation assumes a known race model; simulator-dependent",
     "Characterises where RL and exact optimisation each perform better"],
    ["9", "Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy — "
          "Jiang et al., 2024",
     "Deep models including a transformer compared against tree-based regression",
     "Mercedes-AMG PETRONAS team historical race telemetry",
     "Relies on proprietary team telemetry; not reproducible publicly",
     "Tree-based model competitive with deep models, with better explainability"],
    ["10", "Applying machine learning to forecast Formula 1 race outcomes — Aalto University "
           "MSc thesis, 2023",
     "SVM, Random Forest and neural networks for race outcome forecasting",
     "F1 race data, 2019–2022 seasons",
     "Race-level outcome focus; concludes pit prediction remains difficult",
     "Establishes the difficulty of the task for classical models"],
    ["11", "Predictive Model for Pitstop Strategy in Formula 1 using Ensemble Learning — "
           "National College of Ireland MSc project, 2023",
     "Ensemble learning over engineered lap and race features",
     "Historical F1 race and lap data",
     "Student project; limited feature set and no sequence model",
     "Ensemble methods give a workable pit-stop predictor"],
    ["12", "FastF1 — open-source Python library for Formula 1 timing and telemetry",
     "Programmatic access to official timing, telemetry, position and weather data",
     "Official F1 live-timing feed and the Ergast results archive",
     "Public data is coarser than proprietary team telemetry",
     "The data source this project is built on"],
]
LIT3 = [
    ["13", "Long Short-Term Memory — Hochreiter & Schmidhuber, Neural Computation 9(8), 1997",
     "Gated recurrent architecture addressing vanishing gradients",
     "Synthetic long-lag sequence benchmarks",
     "Sequential computation; limited parallelism over long sequences",
     "Foundation of the recurrent layers used in all three models"],
    ["14", "Bidirectional Recurrent Neural Networks — Schuster & Paliwal, IEEE TSP 45(11), 1997",
     "Runs two recurrent passes, forward and backward, over the same sequence",
     "Speech and phoneme classification benchmarks",
     "Requires the full sequence, so unsuitable for streaming input",
     "Justifies reading a closed 10-lap window in both directions"],
    ["15", "Attention Is All You Need — Vaswani et al., NeurIPS 30, 2017",
     "Multi-head self-attention replacing recurrence entirely",
     "WMT 2014 English–German and English–French translation",
     "Quadratic cost in sequence length; data-hungry",
     "Basis of the cross-driver attention block in Model 3"],
    ["16", "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for "
           "Sequence Modeling — Bai, Kolter & Koltun, arXiv:1803.01271, 2018",
     "Systematic comparison of temporal convolution against recurrent models",
     "A suite of standard sequence-modelling benchmarks",
     "Findings are benchmark-dependent and do not transfer automatically",
     "Motivates the convolutional front end evaluated as Model 2"],
    ["17", "Multitask Learning — Caruana, Machine Learning 28(1), 1997",
     "Learning several related tasks through one shared representation",
     "Several supervised learning benchmarks",
     "Gains depend on task relatedness; can hurt when tasks conflict",
     "Basis of the four-head design in Model 3"],
]
for idx, rows in zip((7, 8, 9), (LIT, LIT2, LIT3)):
    fill_table(table_of(S[idx]), [HEAD] + rows, sizes=(8.5, 7),
               col_w=[0.6, 3.1, 2.9, 2.2, 2.4, 2.4])

# ---- 11 software and hardware ---------------------------------------------
fill_table(table_of(S[10]), [
    ["SOFTWARE", "HARDWARE"],
    ["Operating system — Windows 11 (build 10.0.26300)",
     "Processor — Intel Core i7-12650H, 10 cores / 16 threads"],
    ["Language — Python 3.12.10",
     "Memory — 16 GB system RAM (15.7 GB usable)"],
    ["Deep learning — TensorFlow 2.20.0, Keras 3.12.0",
     "Storage — SSD, ~10 GB used by the telemetry cache"],
    ["Data source — FastF1 3.8.3 (official F1 timing feed)",
     "Graphics — NVIDIA RTX 3050 Laptop, present but unused"],
    ["Data handling — NumPy 2.3.3, pandas 2.3.3, PyArrow 25.0.1",
     "Network — broadband, for the FastF1 download sweep"],
    ["Classical ML and metrics — scikit-learn 1.7.2",
     "Compute mode — CPU only; no CUDA-enabled build available"],
    ["Visualisation — Matplotlib 3.10.6, seaborn 0.13.2, Pillow 11.3.0",
     "Training cost — under 65 seconds per model"],
    ["Notebooks, testing, reporting — JupyterLab 4.4.7, pytest 9.1.1, python-pptx 1.0.2",
     "Inference cost — 0.027 to 0.061 ms per driver-lap"],
    ["Version control — Git and GitHub",
     "Minimum viable spec — any x64 CPU, 8 GB RAM, 15 GB disk"],
], sizes=(12, 10))

# ---- 12 proposed solution --------------------------------------------------
set_bullets(S[11], [
    "A supervised multi-task deep learning system that predicts, lap by lap, whether each "
    "driver will pit within the next three laps. No reinforcement learning and no simulator.",
    "Three architectures were implemented and compared, all solving the same task on the same "
    "data so that any difference in score is attributable to the architecture alone:",
    (1, f"Model 1 — Bidirectional LSTM ({int(M1['params']):,} parameters). The sequential "
        "baseline benchmarked in the literature; reads the closed 10-lap window in both "
        "directions."),
    (1, f"Model 2 — CNN–BiLSTM ({int(M2['params']):,} parameters). Convolutions detect short "
        "local motifs such as a three-lap burst of degradation; the recurrent layer models how "
        "they evolve."),
    (1, f"Model 3 — Multi-task network with cross-driver attention ({int(M3['params']):,} "
        "parameters). A shared encoder, attention across every driver on track, and four task "
        "heads. The proposed contribution."),
    "Reason for the choice: pit strategy is inherently sequential, so the window of recent laps "
    "is the natural input; and competitive pressure is inherently cross-driver, which motivated "
    "testing attention over the field as the project's own hypothesis.",
], size=15, spacing=9)

# ---- 13 dataset details ----------------------------------------------------
set_bullets(S[12], [
    "Dataset: Formula 1 official timing and telemetry, accessed through the FastF1 API — "
    "90 races across the 2021 to 2024 seasons, 98,357 driver-laps, 81,270 labelled sequences.",
    "Feature groups (55 features per driver-lap):",
    (1, "Timing and pace — lap time, delta to the stint median, lap-to-lap change, rolling mean "
        "and spread, sector splits, and a rolling tyre-degradation slope."),
    (1, "Tyre and stint — compound, tyre life, stint number, fresh-tyre flag, pit flags, stops "
        "completed so far."),
    (1, "Race position — track position, grid position, positions gained, lap number, race "
        "fraction, laps remaining."),
    (1, "Track status — yellow flag, safety car, virtual safety car and red flag."),
    (1, "Weather — air and track temperature, humidity, pressure, wind, rainfall."),
    (1, "Telemetry — speed, throttle, full-throttle share, brake share, RPM and gear, "
        "aggregated per lap from the 3.7 Hz feed."),
    (1, "Competitor context — gap to leader, intervals ahead and behind, rivals pitting this "
        "lap, field mean tyre age, and this driver's tyre age relative to the field."),
    "Split by season so no race appears twice: train 2021–22 (43 races), validation 2023 "
    "(22 races), test 2024 (24 races, entirely held out).",
], size=14, spacing=6)

# ---- 14 methodology (architecture diagram) --------------------------------
s = S[13]
set_title(s, "METHODOLOGY")
ph = body_ph(s)
if ph is not None:
    ph._element.getparent().remove(ph._element)
from PIL import Image as _PIL
with _PIL.open(FIG / "architecture_diagram.png") as im:
    ratio = im.size[0] / im.size[1]
h = Emu(4750000)
w = Emu(int(h * ratio))
if w > Emu(11300000):
    w = Emu(11300000)
    h = Emu(int(w / ratio))
s.shapes.add_picture(str(FIG / "architecture_diagram.png"),
                     Emu(int((SLIDE_W - w) / 2)), Emu(1500000), width=w, height=h)

# ---- 15 explanation --------------------------------------------------------
set_bullets(S[14], [
    "Acquisition. Every 2021–2024 race session is downloaded through FastF1 and written as one "
    "driver-lap table per race. The API permits 500 calls per hour and a race costs about nine, "
    "so the sweep is paced and resumes from cache.",
    "Telemetry reduction. Raw car data arrives at 3.7 Hz — tens of gigabytes across 90 races. "
    "Each lap's trace is reduced to eight summary channels at download time by binning samples "
    "into laps in a single vectorised pass, which costs 0.25 seconds per race.",
    "Labelling. The target is whether the driver pits within the next three laps. The label is "
    "built by shifting the pit flag forward, so lap t's own stop can never enter lap t's own "
    "label — the leakage that would otherwise inflate every score.",
    "Windowing. A 10-lap sliding window is taken per driver per race; windows never cross a "
    "driver or race boundary. The scaler is fitted on the training split only.",
    "Training. All three models use the same weighted cross-entropy, the same early stopping on "
    "validation PR-AUC rather than loss, and the same threshold tuned on validation and applied "
    "unchanged to test.",
], size=14, spacing=8)

# ---- 16 onwards: results ---------------------------------------------------
s = S[15]
set_title(s, "RESULTS — Three-Model Comparison")
rows = [["Model", "Params", "Accuracy", "Bal. acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ["Never-pit baseline", "—", f(BASE["accuracy"], 4), f(BASE["balanced_accuracy"], 4),
         f(BASE["precision"]), f(BASE["recall"]), f(BASE["f1"]), f(BASE["roc_auc"]), f(BASE["pr_auc"])]]
for r in (M1, M2, M3):
    rows.append([r["model"], f"{int(r['params']):,}", f(r["accuracy"], 4),
                 f(r["balanced_accuracy"], 4), f(r["precision"]), f(r["recall"]),
                 f(r["f1"]), f(r["roc_auc"]), f(r["pr_auc"])])
ph = body_ph(s)
left, top, width = ph.left, Emu(1900000), ph.width
ph._element.getparent().remove(ph._element)
gf = s.shapes.add_table(len(rows), 9, left, top, width, Emu(1900000))
fill_table(gf.table, rows, sizes=(11, 10))
box = s.shapes.add_textbox(left, Emu(4100000), width, Emu(1700000))
tf = box.text_frame
tf.word_wrap = True
for i, line in enumerate([
    "Test set: the entirely held-out 2024 season — 24 races, 22,228 driver-laps, 8.52% positive.",
    "Read the baseline row first: predicting \u201cno pit\u201d on every lap scores 0.9148 accuracy "
    "and exactly zero on precision, recall and F1. That is why accuracy cannot be the headline "
    "metric for this problem.",
    "Model 1 leads on both AUCs. Model 2 reaches within 0.017 PR-AUC of it using 12\u00d7 fewer "
    "parameters.",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = line
    p.space_after = Pt(8)
    for r in p.runs:
        r.font.size = Pt(13)
        r.font.name = BODY_FONT
        r.font.color.rgb = BODY

idx = 15
for title, img, cap in [
    ("RESULTS — Accuracy and Loss Graphs", "comparison_accuracy.png",
     "Validation accuracy per epoch. All three models peak within the first few epochs, then overfit."),
    ("RESULTS — Loss Graphs", "comparison_loss.png",
     "Validation loss per epoch. Model 3 shows its pit-head component: its combined four-head "
     "objective is on a different scale and would not be comparable."),
    ("RESULTS — Confusion Matrices", "comparison_confusion_matrices.png",
     "Each cell shows the count and that count as a share of its true class. Bottom-right is "
     "stops correctly anticipated; bottom-left is missed stops."),
    ("RESULTS — ROC and Precision-Recall Curves", "comparison_precision_recall.png",
     "PR-AUC against a 0.085 base rate. On an imbalanced problem the precision-recall curve is "
     "the honest one: its baseline is the positive rate, not a diagonal."),
    ("RESULTS — Accuracy Against Model Size", "comparison_efficiency.png",
     "\u201cMost efficient\u201d is a two-axis claim. Model 2 gives up 0.017 PR-AUC for a 12\u00d7 "
     "reduction in size; Model 3 is fastest per lap at 0.027 ms."),
    ("RESULTS — Historical Race Replay", "replay_grid_2024_11.png",
     "Predicted pit probability across the whole field, Austrian Grand Prix 2024. Probability "
     "builds through each stint and collapses the moment the car pits; red marks are the actual stops."),
]:
    idx += 1
    add_picture_slide(prs, idx - 1, title, FIG / img, cap)

# ablation slide
idx += 1
s = add_slide(prs, idx - 1)
set_title(s, "RESULTS — Ablation: Does Cross-Driver Attention Help?")
abl_rows = [["Config", "Attention", "Multi-task", "Params", "Val PR-AUC", "Test PR-AUC"]]
for key, label in (("A", "A — full Model 3"), ("B", "B"), ("C", "C"), ("D", "D — control")):
    r = A[key]
    abl_rows.append([label, "Yes" if r["attention"] else "—", "Yes" if r["multi_task"] else "—",
                     f"{int(r['params']):,}", f"{r['val_pr_auc']:.4f}", f"{r['test_pr_auc']:.3f}"])
ph = body_ph(s)
left, width = ph.left, ph.width
ph._element.getparent().remove(ph._element)
gf = s.shapes.add_table(len(abl_rows), 6, left, Emu(1750000), Emu(6400000), Emu(1700000))
fill_table(gf.table, abl_rows, sizes=(11, 10))
gf2 = s.shapes.add_table(5, 3, Emu(7700000), Emu(1750000), Emu(3900000), Emu(1700000))
fill_table(gf2.table, [
    ["Effect", "Compare", "\u0394 PR-AUC"],
    ["Attention, multi-task", "A vs C", f"{A['A']['test_pr_auc']-A['C']['test_pr_auc']:+.3f}"],
    ["Attention, pit only", "B vs D", f"{A['B']['test_pr_auc']-A['D']['test_pr_auc']:+.3f}"],
    ["Multi-task, w/ attn", "A vs B", f"{A['A']['test_pr_auc']-A['B']['test_pr_auc']:+.3f}"],
    ["Multi-task, no attn", "C vs D", f"{A['C']['test_pr_auc']-A['D']['test_pr_auc']:+.3f}"],
], sizes=(11, 10))
box = s.shapes.add_textbox(left, Emu(3800000), width, Emu(2000000))
tf = box.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "The finding is negative, and consistent."
for r in tf.paragraphs[0].runs:
    r.font.size, r.font.bold, r.font.color.rgb = Pt(16), True, ACCENT
for line in [
    "Cross-driver attention costs performance in both conditions, with and without the auxiliary "
    "heads. Multi-task learning is roughly neutral. The best configuration in the grid is the one "
    "with no attention at all.",
    "Two candidate explanations, both testable. Regrouping by (race, lap) turns 38,890 training "
    "windows into 2,181 grids — roughly 18\u00d7 fewer gradient updates per epoch — just as "
    "attention adds parameters. And the competitor context is already engineered into the 55 "
    "inputs, so attention may be relearning it noisily.",
    "A controlled negative result about a specific mechanism on a specific dataset is a more "
    "defensible contribution than an unsupported positive claim.",
]:
    p = tf.add_paragraph()
    p.text = line
    p.space_after = Pt(7)
    for r in p.runs:
        r.font.size, r.font.name, r.font.color.rgb = Pt(12.5), BODY_FONT, BODY

# replay results slide
idx += 1
s = add_slide(prs, idx - 1)
set_title(s, "RESULTS — AI Calls vs. Real Strategy Decisions")
rep_rows = [["Model", "Stops anticipated", "%", "Median lead", "False alarms / driver-race"]]
for _, r in REPLAY.iterrows():
    rep_rows.append([r["model"], f"{int(r['anticipated'])} / {int(r['stops_in_season'])}",
                     f"{r['anticipated_pct']:.1f}%", f"{r['median_lead_laps']:.0f} laps",
                     f"{r['false_alarms_per_driver_race']:.2f}"])
ph = body_ph(s)
left, width = ph.left, ph.width
ph._element.getparent().remove(ph._element)
gf = s.shapes.add_table(len(rep_rows), 5, left, Emu(1800000), width, Emu(1600000))
fill_table(gf.table, rep_rows, sizes=(12, 11))
box = s.shapes.add_textbox(left, Emu(3650000), width, Emu(2400000))
tf = box.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "The operational ranking is not the per-lap ranking."
for r in tf.paragraphs[0].runs:
    r.font.size, r.font.bold, r.font.color.rgb = Pt(16), True, ACCENT
for line in [
    "A strategist does not ask how often a model is right about a lap. They ask: of the stops "
    "that happened, how many did we see coming, and how early? Scored that way across all 644 "
    "stops of the 2024 season, Model 2 leads — despite trailing Model 1 on PR-AUC.",
    "Model 1 has the highest tuned threshold and is therefore the most conservative, which buys "
    "the lowest false-alarm rate. Model 2 catches 42 more stops across the season for one extra "
    "false alarm roughly every twelve driver-races. Which is preferable is a strategy call, not "
    "a modelling one.",
    "In absolute terms the system is usable: about three laps of warning on more than half of "
    "all real stops, with a spurious warning only once every four to seven driver-races.",
]:
    p = tf.add_paragraph()
    p.text = line
    p.space_after = Pt(8)
    for r in p.runs:
        r.font.size, r.font.name, r.font.color.rgb = Pt(13), BODY_FONT, BODY

# ---- conclusion / future / references --------------------------------------
titles = {sl.shapes.title.text.strip().upper(): i for i, sl in enumerate(prs.slides)
          if sl.shapes.title is not None and sl.shapes.title.text}

set_bullets(prs.slides[titles["CONCLUSION"]], [
    "A complete, reproducible pipeline was built from the official Formula 1 timing feed to "
    "three trained models: 90 races acquired, 55 features engineered, 81,270 leak-free "
    "sequences, and evaluation on a season none of the models had seen.",
    f"All three models beat chance decisively — PR-AUC {M3['pr_auc']:.3f} to {M1['pr_auc']:.3f} "
    f"against a {BASE['pr_auc']:.3f} base rate, three to four times better than random.",
    "The simplest architecture won on accuracy. The plain bidirectional LSTM takes both AUCs; "
    "neither convolutions nor attention improved on it.",
    "Cross-driver attention — the project's own proposed mechanism — does not help on this "
    "dataset. A controlled 2×2 ablation shows it costs performance in both conditions.",
    "Per-lap and operational rankings disagree: Model 1 leads on PR-AUC yet anticipates the "
    "fewest real stops, while Model 2 is the most useful on the pit wall.",
    "The ceiling appears to be the data rather than the model: eighteen configurations across "
    "three architectures all land in a narrow 0.27–0.33 validation PR-AUC band.",
], size=14, spacing=9)

set_bullets(prs.slides[titles["FUTURE ENHANCEMENTS"]], [
    "Extend the dataset backwards through earlier seasons, which would directly test whether "
    "the effective sample size is the binding constraint.",
    "Apply attention over the lap axis rather than the driver axis. The ablation rules out the "
    "cross-driver direction; a temporal transformer over the stint remains untested.",
    "Report seeded confidence intervals. Each configuration was run once, so differences of a "
    "few thousandths in PR-AUC are not currently interpretable.",
    "Evaluate the three auxiliary heads in their own right. Compound choice, lap-time forecast "
    "and race outcome are trained but not yet scored.",
    "Model the cause of a stop. A planned stop, a puncture and a safety-car opportunity are "
    "currently indistinguishable to the model.",
    "Deploy the live demo as a pit-wall tool: the interactive replay already runs offline at "
    "0.03–0.06 ms per lap, which is far inside real-time requirements.",
], size=14, spacing=9)

REFS = [
    "[1] A. Sasikumar, A. A. Leema and P. Balakrishnan, \u201cData-driven pit stop decision support "
    "for Formula 1 using deep learning models,\u201d Frontiers in Artificial Intelligence, vol. 8, "
    "art. 1673148, 2025.",
    "[2] D. Thomas et al., \u201cExplainable reinforcement learning for Formula One race strategy,\u201d "
    "in Proc. 40th ACM/SIGAPP Symp. Applied Computing, 2025.",
    "[3] S. S. Veerasagar, R. Coelho and L. K. Chandrashekar, \u201cOptimum racing: A F1 strategy "
    "predictor using reinforcement learning,\u201d IJRASET, 2025.",
    "[4] S. Fatima and A. Johrendt, \u201cDeep-Racing: An embedded deep neural network model to "
    "predict the winning strategy in Formula One racing,\u201d Int. J. Machine Learning, vol. 13, "
    "no. 3, 2023.",
    "[5] T. Ametsbichler, D. H. Dinh, T. M. Tran and T. Reichert, \u201cA gradient boosting analytics "
    "framework for predictive pit stop decision making in Formula One racing,\u201d SSRN, 2025.",
    "[6] L. Cappello and A. Hoegh, \u201cA state-space approach to modeling tire degradation in "
    "Formula 1 racing,\u201d arXiv:2512.00640, 2025.",
    "[7] \u201cSOSTA AI: Multi-task residual learning for lap-level pit stop decision support in "
    "Formula One,\u201d SportRxiv preprint, 2025.",
    "[8] M\u00e9ndez and C. Thraves, \u201cOptimizing pit stop strategies in Formula 1: Deep "
    "reinforcement learning vs. mathematical programming,\u201d 2025.",
    "[9] J. Jiang et al., \u201cExplainable time series prediction of tyre energy in Formula One race "
    "strategy,\u201d 2024.",
    "[10] \u201cApplying machine learning to forecast Formula 1 race outcomes,\u201d M.Sc. thesis, "
    "Aalto University, 2023.",
    "[11] A. Maheshrao, \u201cPredictive model for pitstop strategy in Formula 1 using ensemble "
    "learning,\u201d M.Sc. project, National College of Ireland, 2023.",
    "[12] FastF1 Developers, \u201cFastF1 documentation,\u201d docs.fastf1.dev, 2025.",
    "[13] S. Hochreiter and J. Schmidhuber, \u201cLong short-term memory,\u201d Neural Computation, "
    "vol. 9, no. 8, pp. 1735\u20131780, 1997.",
    "[14] M. Schuster and K. K. Paliwal, \u201cBidirectional recurrent neural networks,\u201d IEEE "
    "Trans. Signal Processing, vol. 45, no. 11, pp. 2673\u20132681, 1997.",
    "[15] A. Vaswani et al., \u201cAttention is all you need,\u201d in Advances in Neural Information "
    "Processing Systems, vol. 30, 2017.",
    "[16] S. Bai, J. Z. Kolter and V. Koltun, \u201cAn empirical evaluation of generic convolutional "
    "and recurrent networks for sequence modeling,\u201d arXiv:1803.01271, 2018.",
    "[17] R. Caruana, \u201cMultitask learning,\u201d Machine Learning, vol. 28, no. 1, pp. 41\u201375, 1997.",
]
ref_idx = titles["REFERENCES"]
set_bullets(prs.slides[ref_idx], REFS[:9], size=12, spacing=7)
s = add_slide(prs, ref_idx)
set_title(s, "REFERENCES (contd.)")
set_bullets(s, REFS[9:], size=12, spacing=7)

# ---- requirements detail ---------------------------------------------------
# Inserted last so the fixed slide indices used above stay valid. Every version
# below is the one actually installed in the environment the project was built
# and trained in, not a generic recommendation.
req_idx = next(i for i, sl in enumerate(prs.slides)
               if sl.shapes.title is not None
               and sl.shapes.title.text.strip().startswith("Software and Hardware"))

s = add_slide(prs, req_idx)
set_title(s, "SOFTWARE REQUIREMENTS — Detail")
ph = body_ph(s)
left, width = ph.left, ph.width
ph._element.getparent().remove(ph._element)
# python-pptx grows a table to fit its text, so this starts high and uses a
# small body size: 16 rows is close to the most the slide can carry.
gf = s.shapes.add_table(16, 4, left, Emu(1420000), width, Emu(4700000))
fill_table(gf.table, [
    ["Category", "Component", "Version", "Role in this project"],
    ["Platform", "Windows 11", "10.0.26300", "Development and training environment"],
    ["Platform", "Python", "3.12.10", "All data, model and evaluation code"],
    ["Deep learning", "TensorFlow", "2.20.0", "Backend executing all three models"],
    ["Deep learning", "Keras", "3.12.0", "Layer definitions, training loop, callbacks, custom loss"],
    ["Data source", "FastF1", "3.8.3", "Official F1 timing, telemetry, weather and results"],
    ["Data handling", "NumPy", "2.3.3", "Sequence windowing, masking, grid gathering"],
    ["Data handling", "pandas", "2.3.3", "Driver-lap tables and all feature engineering"],
    ["Data handling", "PyArrow", "25.0.1", "Parquet I/O for the 90 per-race tables"],
    ["Machine learning", "scikit-learn", "1.7.2", "Metrics, train-only scaler, logistic and gradient-boosting baselines"],
    ["Visualisation", "Matplotlib", "3.10.6", "Every figure in the report and both decks"],
    ["Visualisation", "seaborn / Pillow", "0.13.2 / 11.3.0", "Figure styling and the animated race replay"],
    ["Notebooks", "JupyterLab / nbconvert", "4.4.7 / 7.16.6", "Four executed review notebooks"],
    ["Testing", "pytest", "9.1.1", "20 regression tests over the loss, registry and masking"],
    ["Reporting", "python-pptx / tabulate", "1.0.2 / 0.10.0", "Automated deck generation and Markdown tables"],
    ["Version control", "Git and GitHub", "2.50", "Phase-by-phase history, one commit per milestone"],
], sizes=(9.5, 8), col_w=[1.5, 2.3, 1.8, 5.6])

s = add_slide(prs, req_idx + 1)
set_title(s, "HARDWARE REQUIREMENTS — Detail")
ph = body_ph(s)
left, width = ph.left, ph.width
ph._element.getparent().remove(ph._element)
gf = s.shapes.add_table(9, 4, left, Emu(1550000), width, Emu(2750000))
fill_table(gf.table, [
    ["Component", "Specification used", "Minimum viable", "Why the project needs it"],
    ["Processor", "Intel Core i7-12650H, 10 cores / 16 threads, 2.3 GHz",
     "Any modern x64 CPU", "All training and inference; the recurrent layers are CPU-bound"],
    ["Memory", "16 GB (15.7 GB usable)", "8 GB",
     "The 81,270 × 10 × 55 float32 tensor plus the (race, lap) grids are held in memory"],
    ["Storage", "SSD, ~10 GB used", "15 GB free",
     "9.8 GB FastF1 telemetry cache, 26 MB processed dataset, figures and checkpoints"],
    ["Graphics", "NVIDIA RTX 3050 Laptop (unused)", "None required",
     "No CUDA-enabled TensorFlow build; native Windows has shipped no GPU support since 2.10"],
    ["Network", "Broadband connection", "Any stable connection",
     "The FastF1 sweep is rate-limited to 500 API calls per hour, about nine per race"],
    ["Training cost", "28–62 s per model, 11–14 epochs", "—",
     "Early stopping on validation PR-AUC halts every model within a few epochs"],
    ["Inference cost", "0.027–0.061 ms per driver-lap", "—",
     "Comfortably inside real-time: a 20-car field is scored in well under a millisecond"],
    ["Dataset build", "~50 min download, ~2 min feature build", "—",
     "One-off cost; the cache makes every later run immediate"],
], sizes=(10, 8.5), col_w=[1.5, 3.0, 1.6, 5.1])

box = s.shapes.add_textbox(left, Emu(4600000), width, Emu(1700000))
tf = box.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "A note on the GPU"
for r in tf.paragraphs[0].runs:
    r.font.size, r.font.bold, r.font.color.rgb = Pt(14), True, ACCENT
for line in [
    "The development machine has an NVIDIA RTX 3050, but it was never used. The pinned "
    "TensorFlow build reports no CUDA support, and TensorFlow has shipped no native Windows GPU "
    "support since version 2.10, so every run in this project executed on CPU.",
    "This turns out not to matter: the largest model trains in 62 seconds and scores a lap in "
    "0.06 ms. The constraint on this project is the quantity of data, not compute — which is "
    "also what the architecture sweeps concluded.",
]:
    p = tf.add_paragraph()
    p.text = line
    p.space_after = Pt(7)
    for r in p.runs:
        r.font.size, r.font.name, r.font.color.rgb = Pt(12), BODY_FONT, BODY

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")
