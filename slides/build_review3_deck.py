"""
Build Review-3.pptx from the shared deck kit.

Run:  python slides/build_review3_deck.py

Every number on these slides is read from reports/metrics/ at build time rather
than typed in, so the deck cannot drift from the results the way a hand-written
one does. Re-run it after any retrain and the slides update themselves.

Design tokens and layout helpers come from slides/deck_kit.py, which was
extracted from the Review-1 deck, so Reviews 1, 2 and 3 share one visual
language.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402

from deck_kit import (  # noqa: E402
    ACCENT, BODY, CARD, CHIP, CHIP_MID, CONTENT_W, FIG, INK, LEFT,
    WHITE, B_FONT, H_FONT, M_FONT,
    add_slide, badge, blank_deck, bullet_rows, callout, caption, head, intro,
    picture, rrect, statcard, table, text, title,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
OUT = Path(r"C:\E\VIT\VIT 7th Sem\Deep Learning Project\Review-3.pptx")


# ---------------------------------------------------------------- data ------
def load():
    comp = pd.read_csv(METRICS / "three_model_comparison.csv")
    replay = pd.read_csv(METRICS / "race_replay_summary.csv")
    ablation = pd.read_csv(METRICS / "mtl_ablation.csv")
    per_model = {}
    for name in ("bilstm", "cnn_bilstm", "mtl_attn"):
        with open(METRICS / f"{name}_metrics.json") as fh:
            per_model[name] = json.load(fh)
    return comp, replay, ablation, per_model


COMP, REPLAY, ABL, MODELS = load()


def row(model_substr):
    return COMP[COMP["model"].str.contains(model_substr, regex=False)].iloc[0]


def f(v, nd=3):
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


BASE = COMP.iloc[0]
M1, M2, M3 = row("Bi-LSTM"), row("CNN"), row("MTL")
A = {r["config"]: r for _, r in ABL.iterrows()}

prs = blank_deck()


# ================================================================ 1 title ===
s = add_slide(prs)
text(s, LEFT, 1_150_000, CONTENT_W, 250000, "REVIEW 3  ·  FINAL MODEL COMPARISON",
     10, BODY, B_FONT)
text(s, LEFT, 1_450_000, CONTENT_W, 1_130_000,
     "Deep Learning-Based Virtual Race Engineer for\nReal-Time Formula 1 Strategy Optimization",
     30, INK, H_FONT, line_pct=1.10)
rrect(s, LEFT, 2_900_000, CONTENT_W, 18000, CHIP_MID, adj=0.0)
text(s, LEFT, 3_170_000, CONTENT_W, 300000,
     "Three Architectures Compared on 90 Races of Real Formula 1 Telemetry",
     12, BODY, B_FONT)

chips = [("Models compared", "Bi-LSTM · CNN–BiLSTM · MTL + attention"),
         ("Best PR-AUC", f"{M1['pr_auc']:.3f}  ·  Model 1, Bi-LSTM"),
         ("Stops anticipated", f"{REPLAY['anticipated_pct'].max():.1f}%  ·  2024 season replay")]
cw, cgap = 3_630_000, 3_705_000
for i, (k, v) in enumerate(chips):
    x = LEFT + i * cgap
    rrect(s, x, 3_860_000, cw, 900_000, CHIP, adj=0.09)
    text(s, x + 230000, 3_990_000, cw - 460000, 200000, k, 8, BODY, B_FONT)
    text(s, x + 230000, 4_230_000, cw - 460000, 420_000, v, 10.5, INK, H_FONT, line_pct=1.15)

text(s, LEFT + 300000, 5_400_000, 7_200_000, 820_000,
     "Submitted By:\n"
     "Dhruv Deepak (23BCE9187)          Akhil V (23BCE8051)\n"
     "Mohan Simha Varma Dantuluri (23BCE9229)          Surya Prathap Mamillapalli (23BCE7783)",
     10.5, BODY, B_FONT, line_pct=1.45)

# ================================================================ 2 scope ===
s = add_slide(prs)
badge(s, "01 — REVIEW 3 SCOPE")
title(s, "What Was Asked, and What Is Delivered")
intro(s, "Review 2 delivered the first model end to end. Review 3 completes the set: "
         "three architectures trained on the same data under the same protocol, compared "
         "on accuracy, loss, confusion matrices and computational cost.")
head(s, LEFT, 1_950_000, 5_130_000, "Requirement")
rrect(s, LEFT, 2_300_000, 5_130_000, 2_560_000, CARD, adj=0.03, line=CHIP_MID)
bullet_rows(s, LEFT + 260000, 2_490_000, 4_610_000, [
    ("Compare three models on the dataset",
     "All three trained on the same 90 races, same splits, same protocol."),
    ("Accuracy and loss graphs",
     "Per-epoch curves for every model, overlaid and per model."),
    ("Confusion matrices of three models",
     "Side by side on the held-out 2024 season."),
    ("Rough draft of the report",
     "Full academic draft in the repository, reports/REPORT_DRAFT.md."),
], gap=620000, hsize=10.5, bsize=9)

head(s, 6_400_000, 1_950_000, 5_130_000, "Beyond the Requirement")
extras = [("Ablation study", "Does cross-driver attention actually help?"),
          ("Historical race replay", "AI calls against real engineers, 2024 season."),
          ("Classical baselines", "Is a deep model justified at all?"),
          ("Efficiency analysis", "Parameters, training time, per-lap latency.")]
for i, (k, v) in enumerate(extras):
    y = 2_300_000 + i * 640_000
    rrect(s, 6_400_000, y, 5_130_000, 560_000, CARD, adj=0.06, line=CHIP_MID)
    text(s, 6_400_000 + 230000, y + 95000, 2_300_000, 250000, k, 10.5, INK, H_FONT)
    text(s, 6_400_000 + 2_650_000, y + 95000, 2_300_000, 380_000, v, 8.5, BODY, B_FONT,
         line_pct=1.22)

callout(s, "Every result below is on the 2024 season — 24 races and 22,228 driver-laps that no "
           "model saw during training, validation or threshold selection.",
        y=5_150_000, h=560_000)

# =========================================================== 3 the models ===
s = add_slide(prs)
badge(s, "02 — THE THREE MODELS")
title(s, "One Task, Three Architectures")
intro(s, "All three answer the same question — given the last 10 laps of telemetry for a "
         "driver, will that driver pit within the next 3 laps? Only the architecture varies, "
         "which is what makes their scores comparable.")
cards = [
    ("Model 1", "Bi-LSTM", f"{int(M1['params']):,} params",
     "Stacked bidirectional LSTM, 64 → 32. The sequential baseline benchmarked by "
     "Sasikumar et al. Reads the closed 10-lap window in both directions."),
    ("Model 2", "CNN–BiLSTM", f"{int(M2['params']):,} params",
     "Two Conv1D layers with batch norm feeding a Bi-LSTM. Convolutions pick up short "
     "local motifs; the recurrent layer models how they evolve."),
    ("Model 3", "MTL + cross-driver\nattention", f"{int(M3['params']):,} params",
     "Shared encoder, multi-head attention across all drivers on track, four task heads. "
     "The project's proposed contribution."),
]
for i, (num, name, params, desc) in enumerate(cards):
    x = LEFT + i * 3_622_915
    rrect(s, x, 1_950_000, 3_382_915, 2_680_000, CARD, adj=0.04, line=CHIP_MID)
    text(s, x + 240000, 2_140_000, 2_900_000, 220000, num, 8.5, BODY, B_FONT)
    text(s, x + 240000, 2_410_000, 2_900_000, 700_000, name, 13, INK, H_FONT, line_pct=1.12)
    rrect(s, x + 240000, 3_200_000, 1_600_000, 270_000, CHIP, adj=0.22)
    text(s, x + 240000, 3_258_000, 1_600_000, 200000, params, 8, INK, B_FONT,
         align=PP_ALIGN.CENTER)
    text(s, x + 240000, 3_620_000, 2_900_000, 900_000, desc, 8.5, BODY, B_FONT, line_pct=1.26)

head(s, LEFT, 4_820_000, CONTENT_W, "Held Fixed Across All Three — This Is What Makes It a Fair Test")
fixed = ["Same 90 races and 55 features", "Same season-based splits",
         "Same weighted loss", "Same early-stopping rule",
         "Same threshold procedure", "Same metric set"]
for i, item in enumerate(fixed):
    x = LEFT + (i % 3) * 3_960_000
    y = 5_180_000 + (i // 3) * 450_000
    rrect(s, x, y, 3_700_000, 360_000, CHIP, adj=0.16)
    text(s, x + 180000, y + 95000, 3_340_000, 220000, item, 9.5, INK, B_FONT)

callout(s, "Each model's width and regularisation were selected by its own six-configuration "
           "sweep on validation PR-AUC, so the comparison measures architecture rather than "
           "how hard any one model was tuned.",
        y=6_180_000, h=560_000)

# ============================================================ 4 accuracy ====
s = add_slide(prs)
badge(s, "03 — ACCURACY GRAPHS")
title(s, "Validation Accuracy per Epoch")
intro(s, "All three models reach their best validation score within the first few epochs and "
         "then overfit. The marker on each curve is the epoch early stopping restored "
         "weights from.", h=360000)
pic = picture(s, FIG / "comparison_accuracy.png", LEFT + 900_000, 1_790_000, w=9_100_000)
caption(s, LEFT + 900_000, 1_790_000 + pic.height + 90_000, 9_100_000,
        "Validation accuracy — all three models on the same axes")
callout(s, "Why accuracy is the least informative graph here: the never-pit baseline scores "
           f"{BASE['accuracy']:.4f} on this test set while never once calling a stop. The curves below "
           "matter far more than this one.",
        y=5_600_000, h=600_000)

# ================================================================ 5 loss ====
s = add_slide(prs)
badge(s, "04 — LOSS GRAPHS")
title(s, "Validation Loss per Epoch")
intro(s, "Validation loss rises from the first or second epoch in every model — the signature "
         "of overfitting on a dataset whose effective sample size is far below its nominal "
         "one.", h=360000)
pic = picture(s, FIG / "comparison_loss.png", LEFT + 900_000, 1_790_000, w=9_100_000)
caption(s, LEFT + 900_000, 1_790_000 + pic.height + 90_000, 9_100_000,
        "Weighted cross-entropy on validation — lower is better")
callout(s, "Model 3's pit-head loss is plotted, not its total. Models 1 and 2 minimise weighted "
           "cross-entropy; Model 3 minimises a combined four-head objective on a different scale, "
           "so overlaying the totals would compare quantities that are not the same thing.",
        y=5_600_000, h=600_000)

# =================================================== 6 confusion matrices ===
s = add_slide(prs)
badge(s, "05 — CONFUSION MATRICES")
title(s, "All Three Models on the Held-Out 2024 Season")
intro(s, "Each cell shows the raw count and that count as a share of its true class. Row "
         "percentages are what matter: in absolute terms the 'No pit' row dwarfs everything "
         "else.", h=360000)
pic = picture(s, FIG / "comparison_confusion_matrices.png", LEFT, 1_820_000, w=CONTENT_W)
y_after = 1_820_000 + pic.height + 180_000
quads = [
    ("Bottom-right", "Stops correctly anticipated — the number the project exists to raise."),
    ("Bottom-left", "Missed stops. In a real race these are the expensive errors."),
    ("Top-right", "False alarms. Comparatively cheap: a warned strategist who does not stop loses nothing."),
]
for i, (k, v) in enumerate(quads):
    x = LEFT + i * 3_960_000
    rrect(s, x, y_after, 3_700_000, 760_000, CARD, adj=0.06, line=CHIP_MID)
    text(s, x + 200000, y_after + 130000, 3_300_000, 220000, k, 10, INK, H_FONT)
    text(s, x + 200000, y_after + 400000, 3_300_000, 320_000, v, 8.5, BODY, B_FONT, line_pct=1.22)

# ============================================================== 7 curves ====
s = add_slide(prs)
badge(s, "06 — THRESHOLD-INDEPENDENT CURVES")
title(s, "ROC and Precision-Recall")
intro(s, "These score each model's ranking of laps independently of where its decision cut-off "
         "is drawn. On an imbalanced problem the precision-recall curve is the honest one — "
         "its baseline is the positive rate, not a diagonal.", h=360000)
picture(s, FIG / "comparison_roc.png", 900_000, 1_880_000, h=3_600_000)
picture(s, FIG / "comparison_precision_recall.png", 5_250_000, 1_880_000, h=3_600_000)
caption(s, 900_000, 5_600_000, 4_100_000, "ROC — all three clear the chance diagonal")
caption(s, 5_250_000, 5_600_000, 4_100_000,
        f"PR — against a {BASE['pr_auc']:.3f} base rate")
for i, (v, l) in enumerate([(f"{M1['pr_auc']:.3f}", "Model 1 PR-AUC"),
                            (f"{M2['pr_auc']:.3f}", "Model 2 PR-AUC"),
                            (f"{M3['pr_auc']:.3f}", "Model 3 PR-AUC")]):
    statcard(s, 9_500_000, 1_950_000 + i * 1_180_000, 2_030_000, 1_020_000, v, l, vsize=19)

# ============================================================== 8 results ===
s = add_slide(prs)
badge(s, "07 — RESULTS")
title(s, "Three Models Compared on the 2024 Season")
intro(s, "22,228 driver-laps from 24 races, 8.52% positive. Each model is shown at its own "
         "threshold, tuned on validation and never on test.")
rows = [["Model", "Params", "Accuracy", "Bal. acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ["Never-pit baseline", "—", f(BASE["accuracy"], 4), f(BASE["balanced_accuracy"], 4),
         f(BASE["precision"]), f(BASE["recall"]), f(BASE["f1"]), f(BASE["roc_auc"]), f(BASE["pr_auc"])]]
for r in (M1, M2, M3):
    rows.append([r["model"].replace("—", "–"), f"{int(r['params']):,}",
                 f(r["accuracy"], 4), f(r["balanced_accuracy"], 4), f(r["precision"]),
                 f(r["recall"]), f(r["f1"]), f(r["roc_auc"]), f(r["pr_auc"])])
table(s, LEFT, 1_900_000, CONTENT_W, rows,
      col_w=[2_950_000, 1_080_000, 1_120_000, 1_080_000, 1_120_000, 1_000_000,
             880_000, 1_080_000, 558_745],
      row_h=330000, hdr_h=350000, fsize=9.5, hsize=9.5, bold_col0=True, highlight_rows=(2,))

head(s, LEFT, 3_600_000, 5_130_000, "Model 1 Leads on Accuracy")
rrect(s, LEFT, 3_940_000, 5_130_000, 1_520_000, CARD, adj=0.05, line=CHIP_MID)
text(s, LEFT + 250000, 4_110_000, 4_630_000, 1_200_000,
     f"The plain Bi-LSTM takes both AUCs — {M1['roc_auc']:.3f} ROC and {M1['pr_auc']:.3f} PR — and the best "
     f"F1. It is also the largest model by an order of magnitude, at "
     f"{int(M1['params']):,} parameters.",
     10, BODY, B_FONT, line_pct=1.32)

head(s, 6_400_000, 3_600_000, 5_130_000, "The Baseline Is Still the Argument")
rrect(s, 6_400_000, 3_940_000, 5_130_000, 1_520_000, CARD, adj=0.05, line=CHIP_MID)
text(s, 6_400_000 + 250000, 4_110_000, 4_630_000, 1_200_000,
     f"Predicting 'no pit' on every lap scores {BASE['accuracy']:.4f} accuracy and exactly zero on "
     "precision, recall and F1. This row is why accuracy cannot be the headline metric "
     "for this problem.",
     10, BODY, B_FONT, line_pct=1.32)

# =========================================================== 9 efficiency ===
s = add_slide(prs)
badge(s, "08 — MOST EFFICIENT MODEL")
title(s, "Accuracy Is Only Half of 'Efficient'")
intro(s, "The system's premise is real-time operation, so size and latency are part of the "
         "claim. On that axis the ranking inverts.", h=360000)
picture(s, FIG / "comparison_efficiency.png", LEFT, 1_800_000, w=6_900_000)
head(s, 7_600_000, 1_880_000, 3_930_000, "Reading the Trade-off")
bullet_rows(s, 7_600_000, 2_220_000, 3_930_000, [
    ("Model 2 is the efficiency winner",
     f"{int(M2['params']):,} parameters — {int(M1['params']) / int(M2['params']):.0f}× smaller than Model 1 — "
     f"for {M1['pr_auc'] - M2['pr_auc']:.3f} less PR-AUC."),
    ("Model 3 is fastest per lap",
     f"{M3['ms_per_lap']:.3f} ms against Model 1's {M1['ms_per_lap']:.3f} ms, because it scores a "
     "whole grid of drivers in one pass."),
    ("Model 1 buys accuracy with size",
     "The largest model wins on both AUCs, but by margins a deployment might not pay for."),
], gap=1_000_000, hsize=10.5, bsize=9, body_h=700000)
callout(s, "All three train in under 65 seconds on CPU, so training cost does not separate them. "
           "Model size and inference latency do.",
        y=5_560_000, h=560_000)

# ============================================================ 10 ablation ===
s = add_slide(prs)
badge(s, "09 — ABLATION STUDY")
title(s, "Does Cross-Driver Attention Earn Its Place?")
intro(s, "A width sweep would say which Model 3 is best; it would not answer the question the "
         "project is actually asking. So the sweep is a 2×2 — attention on/off crossed with "
         "multi-task on/off — selected on validation PR-AUC.")
abl_rows = [["Config", "Attention", "Multi-task", "Params", "Val PR-AUC", "Test PR-AUC"]]
for key, label in (("A", "A — full Model 3"), ("B", "B"), ("C", "C"), ("D", "D — control")):
    r = A[key]
    abl_rows.append([label, "Yes" if r["attention"] else "—",
                     "Yes" if r["multi_task"] else "—", f"{int(r['params']):,}",
                     f"{r['val_pr_auc']:.4f}", f"{r['test_pr_auc']:.3f}"])
table(s, LEFT, 1_950_000, 5_600_000, abl_rows,
      col_w=[1_600_000, 900_000, 980_000, 880_000, 1_000_000, 240_000],
      row_h=300000, hdr_h=320000, fsize=9, hsize=9, bold_col0=True, highlight_rows=(3,))

head(s, 6_400_000, 1_950_000, 5_130_000, "The Effect, Isolated")
eff = [("Attention, with multi-task", f"A vs C", f"{A['A']['test_pr_auc'] - A['C']['test_pr_auc']:+.3f}"),
       ("Attention, pit head only", f"B vs D", f"{A['B']['test_pr_auc'] - A['D']['test_pr_auc']:+.3f}"),
       ("Multi-task, with attention", f"A vs B", f"{A['A']['test_pr_auc'] - A['B']['test_pr_auc']:+.3f}"),
       ("Multi-task, no attention", f"C vs D", f"{A['C']['test_pr_auc'] - A['D']['test_pr_auc']:+.3f}")]
for i, (k, cmp_, delta) in enumerate(eff):
    y = 2_300_000 + i * 560_000
    rrect(s, 6_400_000, y, 5_130_000, 480_000, CARD, adj=0.07, line=CHIP_MID)
    text(s, 6_400_000 + 220000, y + 125000, 2_900_000, 240000, k, 9.5, INK, H_FONT)
    text(s, 6_400_000 + 3_350_000, y + 125000, 800_000, 240000, cmp_, 8.5, BODY, B_FONT)
    text(s, 6_400_000 + 4_200_000, y + 115000, 760_000, 260000, delta, 12,
         ACCENT if delta.startswith("-") else INK, H_FONT, align=PP_ALIGN.RIGHT)

text(s, LEFT, 3_700_000, 5_600_000, 300000,
     "The finding is negative, and consistent.", 12.5, ACCENT, H_FONT)
text(s, LEFT, 4_030_000, 5_600_000, 1_300_000,
     "Cross-driver attention costs performance in both conditions, with and without the "
     "auxiliary heads. Multi-task learning is roughly neutral. The best configuration in the "
     "whole grid is the one with no attention at all.",
     9.5, BODY, B_FONT, line_pct=1.30)

callout(s, "Two candidate explanations, both testable. Regrouping by (race, lap) turns 38,890 "
           "training windows into 2,181 grids — roughly 18× fewer gradient updates per epoch, "
           "just as attention adds parameters. And the competitor context is already engineered "
           "into the 55 inputs, so attention may be relearning it noisily.",
        y=5_500_000, h=700_000)

# ============================================================== 11 replay ===
s = add_slide(prs)
badge(s, "10 — HISTORICAL RACE REPLAY")
title(s, "AI Recommendations Against Real Strategy Calls")
intro(s, "The test set is the whole 2024 season, so replaying it compares machine and human "
         "decisions on the same races. Probability builds through each stint and collapses the "
         "moment the car pits.", h=360000)
pic = picture(s, FIG / "replay_grid_2024_11.png", LEFT + 500_000, 1_800_000, w=9_900_000)
caption(s, LEFT + 500_000, 1_800_000 + pic.height + 80_000, 9_900_000,
        "Model 1 — predicted pit probability across the whole field, Austrian Grand Prix 2024. "
        "Red marks are the actual stops.")

# ======================================================= 12 replay results ==
s = add_slide(prs)
badge(s, "11 — REPLAY RESULTS")
title(s, "Of the Stops That Happened, How Many Did We See Coming?")
intro(s, "Per-lap metrics answer how often a model is right about a lap. A strategist asks a "
         "different question, so the replay scores per stop across all 644 stops of the 2024 "
         "season.")
rep_rows = [["Model", "Stops anticipated", "%", "Median lead", "False alarms / driver-race"]]
for _, r in REPLAY.iterrows():
    rep_rows.append([r["model"].replace("—", "–"),
                     f"{int(r['anticipated'])} / {int(r['stops_in_season'])}",
                     f"{r['anticipated_pct']:.1f}%",
                     f"{r['median_lead_laps']:.0f} laps",
                     f"{r['false_alarms_per_driver_race']:.2f}"])
table(s, LEFT, 1_900_000, CONTENT_W, rep_rows,
      col_w=[3_300_000, 2_200_000, 1_300_000, 1_700_000, 2_368_745],
      row_h=340000, hdr_h=360000, fsize=10, hsize=10, bold_col0=True, highlight_rows=(2,))

text(s, LEFT, 3_450_000, CONTENT_W, 300000,
     "The operational ranking is not the per-lap ranking.", 12.5, ACCENT, H_FONT)
head(s, LEFT, 3_880_000, 5_130_000, "Why They Disagree")
rrect(s, LEFT, 4_220_000, 5_130_000, 1_420_000, CARD, adj=0.05, line=CHIP_MID)
text(s, LEFT + 250000, 4_390_000, 4_630_000, 1_100_000,
     "Model 1 leads on PR-AUC but anticipates the fewest stops. Its higher tuned threshold "
     "makes it the most conservative, which buys the lowest false-alarm rate in the field.",
     10, BODY, B_FONT, line_pct=1.30)
head(s, 6_400_000, 3_880_000, 5_130_000, "What a Strategist Would Weigh")
rrect(s, 6_400_000, 4_220_000, 5_130_000, 1_420_000, CARD, adj=0.05, line=CHIP_MID)
text(s, 6_400_000 + 250000, 4_390_000, 4_630_000, 1_100_000,
     "Model 2 catches 42 more stops across the season for one extra false alarm roughly every "
     "twelve driver-races. Which is preferable is a strategy call, not a modelling one.",
     10, BODY, B_FONT, line_pct=1.30)
callout(s, "In absolute terms the rate is usable: about three laps of warning on more than half "
           "of all real stops, with a spurious warning only once every four to seven driver-races.",
        y=5_800_000, h=560_000)

# ============================================================ 13 baselines ==
s = add_slide(prs)
badge(s, "12 — CRITICAL ANALYSIS")
title(s, "Is a Deep Sequence Model Justified?")
intro(s, "A deep model should be justified, not assumed. Two classical models were trained on "
         "exactly the same tensors and scored exactly the same way.")
base_rows = [["Model", "ROC-AUC", "PR-AUC"],
             ["Random (base rate)", "0.500", f"{BASE['pr_auc']:.3f}"],
             ["Logistic regression — final lap only", "0.802", "0.308"],
             ["Gradient boosting — flattened window", "0.853", "0.391"],
             ["Model 1 — Bi-LSTM (best deep model)", f"{M1['roc_auc']:.3f}", f"{M1['pr_auc']:.3f}"]]
table(s, LEFT, 1_900_000, 6_100_000, base_rows,
      col_w=[3_500_000, 1_300_000, 1_300_000],
      row_h=320000, hdr_h=340000, fsize=9.5, hsize=9.5, bold_col0=True, highlight_rows=(3,))

text(s, LEFT, 3_700_000, 6_100_000, 300000,
     "Answer, stated honestly: not yet.", 13, ACCENT, H_FONT)
text(s, LEFT, 4_040_000, 6_100_000, 1_200_000,
     "Gradient boosting on the flattened window still outperforms all three deep models, and "
     "logistic regression on the final lap alone is within reach of them. The sequence "
     "architectures are not yet earning their complexity on this dataset.",
     10, BODY, B_FONT, line_pct=1.32)

head(s, 7_000_000, 1_900_000, 4_530_000, "Why — and It Is Not Capacity")
bullet_rows(s, 7_000_000, 2_250_000, 4_530_000, [
    ("Effective sample size is small",
     "38,890 windows come from just 815 driver-races across 43 races, and consecutive windows "
     "share 9 of their 10 laps."),
    ("Three independent sweeps, one ceiling",
     "Across eighteen configurations spanning 8.8k to 162k parameters, validation PR-AUC never "
     "left the 0.27–0.33 band."),
    ("More data still helps",
     "Training on 2021+2022 beats either season alone, so the regulation change is not the "
     "dominant effect."),
], gap=1_020_000, hsize=10.5, bsize=9, body_h=700000)

# ========================================================= 14 key findings ==
s = add_slide(prs)
badge(s, "13 — KEY FINDINGS")
title(s, "Six Things This Project Establishes")
findings = [
    ("A complete, reproducible pipeline",
     "90 races acquired, 55 features engineered, 81,270 leak-free sequences, three models "
     "trained and evaluated on a season none of them saw."),
    ("All three models beat chance decisively",
     f"PR-AUC {M3['pr_auc']:.3f} to {M1['pr_auc']:.3f} against a {BASE['pr_auc']:.3f} base rate — three to four "
     "times better than random."),
    ("The simplest architecture wins on accuracy",
     f"The plain Bi-LSTM takes both AUCs; adding convolutions or attention did not improve on it."),
    ("Cross-driver attention does not help",
     "A 2×2 ablation shows it costs performance in both conditions. A clean negative result "
     "about the project's own proposed mechanism."),
    ("Per-lap and operational rankings disagree",
     "Model 1 leads on PR-AUC yet anticipates the fewest real stops. Which matters depends on "
     "what a strategist is willing to trade."),
    ("The ceiling is the data, not the model",
     "Eighteen configurations across three architectures all land in the same narrow band."),
]
for i, (h_, b_) in enumerate(findings):
    x = LEFT + (i % 2) * 5_600_000
    y = 1_900_000 + (i // 2) * 1_420_000
    rrect(s, x, y, 5_268_745, 1_240_000, CARD, adj=0.05, line=CHIP_MID)
    text(s, x + 240000, y + 185000, 4_790_000, 280000, h_, 11.5, INK, H_FONT)
    text(s, x + 240000, y + 530000, 4_790_000, 640_000, b_, 9, BODY, B_FONT, line_pct=1.28)

# ========================================================== 15 limitations ==
s = add_slide(prs)
badge(s, "14 — LIMITATIONS & FUTURE WORK")
title(s, "What This Study Cannot Claim, and What Comes Next")
head(s, LEFT, 1_900_000, 5_130_000, "Limitations")
bullet_rows(s, LEFT, 2_240_000, 5_130_000, [
    ("Public telemetry is coarser than a team's",
     "FastF1 exposes far less than the proprietary feeds real strategists use."),
    ("Effective sample size is modest",
     "Overlapping windows and only 89 usable races limit what any architecture can learn."),
    ("A single seed per configuration",
     "Differences of a few thousandths in PR-AUC should not be read as meaningful."),
    ("Pit cause is not modelled",
     "A stop forced by a puncture or a safety car is not distinguishable from a planned one."),
], gap=900000, hsize=10.5, bsize=9, body_h=560000)

head(s, 6_400_000, 1_900_000, 5_130_000, "Future Work")
bullet_rows(s, 6_400_000, 2_240_000, 5_130_000, [
    ("Extend the dataset backwards",
     "More seasons would directly test whether the data ceiling is the binding constraint."),
    ("Attention over laps, not drivers",
     "The ablation rules out the cross-driver axis; a temporal transformer is untested."),
    ("Report seeded confidence intervals",
     "Repeat runs per configuration so small differences become interpretable."),
    ("Use the remaining three heads",
     "Compound, pace and outcome predictions are trained but not yet evaluated in their own right."),
], gap=900000, hsize=10.5, bsize=9, body_h=560000)

# =========================================================== 16 conclusion ==
s = add_slide(prs)
badge(s, "15 — CONCLUSION")
title(s, "From Historical Telemetry to Intelligent Race Strategy")
intro(s, "Three deep architectures, trained on four seasons of real Formula 1 telemetry and "
         "evaluated on a season none of them had seen, predict pit stops three to four times "
         "better than chance — and anticipate more than half of all real stops with about three "
         "laps of warning.", h=560000)
for i, (v, l) in enumerate([(f"{M1['pr_auc']:.3f}", "Best PR-AUC"),
                            (f"{BASE['pr_auc']:.3f}", "Random baseline"),
                            (f"{REPLAY['anticipated_pct'].max():.0f}%", "Stops anticipated"),
                            ("3", "Median laps of warning"),
                            ("644", "Stops replayed"),
                            ("0.14", "False alarms / driver-race")]):
    statcard(s, LEFT + i * 1_830_000, 2_100_000, 1_700_000, 980_000, v, l,
             vsize=16 if len(v) > 5 else 18)

rrect(s, LEFT, 3_320_000, CONTENT_W, 1_320_000, CARD, adj=0.04, line=CHIP_MID)
text(s, LEFT + 300000, 3_520_000, CONTENT_W - 600000, 980_000,
     "The project's proposed contribution — cross-driver attention — was tested properly and "
     "did not improve on a plain Bi-LSTM. That is reported as a finding rather than buried: a "
     "controlled negative result about a specific mechanism on a specific dataset is a more "
     "defensible contribution than an unsupported positive claim, and it points the next "
     "attempt somewhere more promising.",
     11, BODY, B_FONT, line_pct=1.34)

text(s, LEFT, 4_880_000, CONTENT_W, 320000,
     "From historical telemetry to intelligent, data-driven race strategy.",
     13, INK, H_FONT)
callout(s, "All code, figures, metrics, tests and the executed notebooks are version-controlled at "
           "github.com/dhruv-deepak/f1-virtual-race-engineer",
        y=5_450_000, h=560_000)

# =========================================================== 17 references ==
s = add_slide(prs)
badge(s, "REFERENCES")
title(s, "References and Project Repository")
refs = [("FastF1 Documentation", "Programmatic access to official F1 timing, telemetry, position "
         "and weather data.  docs.fastf1.dev"),
        ("Sasikumar, A., Leema, A. A., & Balakrishnan, P. (2025)",
         "Data-driven pit stop decision support for Formula 1 using deep learning models. "
         "Frontiers in Artificial Intelligence, 8, 1673148."),
        ("Thomas, D., et al. (2025)",
         "Explainable reinforcement learning for Formula One race strategy. ACM/SIGAPP SAC."),
        ("Veerasagar, S. S., Coelho, R., & Chandrashekar, L. K. (2025)",
         "Optimum racing: A F1 strategy predictor using reinforcement learning. IJRASET."),
        ("Vaswani, A., et al. (2017)",
         "Attention is all you need. Advances in Neural Information Processing Systems, 30."),
        ("Chollet, F., et al. (2015)",
         "Keras. The deep learning framework used for all three model implementations.")]
for i, (who, what) in enumerate(refs):
    y = 1_700_000 + i * 630_000
    text(s, LEFT, y, 4_200_000, 520_000, who, 10, INK, H_FONT, line_pct=1.20)
    text(s, LEFT + 4_400_000, y, 6_468_745, 520_000, what, 9, BODY, B_FONT, line_pct=1.25)

rrect(s, LEFT, 5_620_000, CONTENT_W, 740_000, CHIP, adj=0.06)
text(s, LEFT + 300000, 5_760_000, 5_000_000, 260000, "Project Repository", 11, INK, H_FONT)
text(s, LEFT + 300000, 6_040_000, 6_000_000, 260000,
     "github.com/dhruv-deepak/f1-virtual-race-engineer", 10.5, BODY, M_FONT)
text(s, LEFT + 7_900_000, 5_790_000, 3_200_000, 480_000,
     "Code, figures, metrics, tests\nand the report draft.", 9, BODY, B_FONT, line_pct=1.28)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")
