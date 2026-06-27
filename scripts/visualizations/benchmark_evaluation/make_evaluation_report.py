"""
Build a single self-contained HTML report for the RareSim benchmark evaluation.

Run:
    python -m scripts.visualizations.benchmark_evaluation.make_evaluation_report \
    --plots outputs/evaluation_visual_questions \
    --output outputs/evaluation_visual_questions/evaluation_report.html
"""

import argparse
import base64
from html import escape
from pathlib import Path

import pandas as pd

from scripts.visualizations.benchmark_evaluation.config import (
    DATASETS,
    SUMMARY_METRIC_COLUMNS,
    SYSTEM_TYPE_COLORS,
)

# Metrics shown in the per-dataset tables
TABLE_METRICS = SUMMARY_METRIC_COLUMNS
HIGHER_IS_BETTER = set(TABLE_METRICS)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def data_uri(path: Path) -> str:
    """Return a base64 data URI for a PNG image file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fmt(value, nd: int = 3) -> str:
    """Format a numeric value for display in report tables."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{nd}f}"


def datasets_present(metrics: pd.DataFrame) -> list[str]:
    """Return available datasets in the configured PhenoBrain order."""
    present = set(metrics["dataset"].unique())
    return [d for d in DATASETS if d in present]


def badge(system_type: str) -> str:
    """Render an HTML badge for a system type."""
    color = SYSTEM_TYPE_COLORS.get(system_type, "#777")
    return (f'<span class="badge" style="background:{color}1a;color:{color};'
            f'border:1px solid {color}55">{escape(system_type)}</span>')


def img_block(path: Path, caption: str = "") -> str:
    """Render an embedded image figure block with an optional caption."""
    cap = f'<figcaption>{escape(caption)}</figcaption>' if caption else ""
    return (f'<figure><img src="{data_uri(path)}" alt="{escape(caption)}">'
            f'{cap}</figure>')


def first_existing(plots_dir: Path, names: list[str]) -> Path | None:
    """Return the first existing file from a list of candidate names."""
    for n in names:
        p = plots_dir / n
        if p.exists():
            return p
    return None


def read_csv(plots_dir: Path, name: str) -> pd.DataFrame:
    """Read a CSV from the plots directory, or return an empty frame."""
    p = plots_dir / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# metric tables
# ---------------------------------------------------------------------------

# pylint: disable=too-many-locals
def metric_table(metrics: pd.DataFrame, dataset: str) -> str:
    """Render the per-dataset metric table as native HTML."""
    sub = metrics[metrics["dataset"] == dataset].copy()
    if sub.empty:
        return "<p class='missing'>No results for this dataset.</p>"
    sub = sub.sort_values(["R@10", "MRR"], ascending=False)

    best = {m: sub[m].max() for m in TABLE_METRICS if m in sub.columns}
    r10_max = max(best.get("R@10", 0) or 0, 1e-9)

    head = "".join(f"<th>{escape(m)}</th>" for m in TABLE_METRICS)
    rows = []
    for _, r in sub.iterrows():
        cells = [f'<td class="method"><span class="dot" style="background:'
                 f'{SYSTEM_TYPE_COLORS.get(r["system_type"], "#777")}"></span>'
                 f'{escape(str(r["method_label"]))}</td>']
        for m in TABLE_METRICS:
            v = r.get(m)
            is_best = (m in best) and pd.notna(v) and abs(v - best[m]) < 1e-9
            cls = "num best" if is_best else "num"
            if m == "R@10" and pd.notna(v):
                w = max(2, min(100, 100 * v / r10_max))
                cells.append(f'<td class="{cls}"><span class="bar" '
                             f'style="width:{w:.0f}%"></span>'
                             f'<span class="barval">{fmt(v)}</span></td>')
            else:
                cells.append(f'<td class="{cls}">{fmt(v)}</td>')

        found = r.get("found_count")
        n = r.get("n_cases")
        found_str = (f"{int(found)}/{int(n)}"
                     if pd.notna(found) and pd.notna(n) else "—")
        avg = r.get("avg_query_time_sec")
        avg_str = fmt(avg, 3) if pd.notna(avg) else "—"
        cells.append(f'<td class="num">{found_str}</td>')
        cells.append(f'<td class="num">{avg_str}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (f'<table class="metrics"><thead><tr>'
            f'<th class="method">Method / tool</th>{head}'
            f'<th>Found</th><th>Avg&nbsp;(s)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ---------------------------------------------------------------------------
# auto findings
# ---------------------------------------------------------------------------

def li(text: str) -> str:
    """Wrap an HTML snippet in a list item."""
    return f"<li>{text}</li>"


def findings_q1(metrics: pd.DataFrame, plots_dir: Path) -> str:
    """Summarize the strongest methods and best Recall@10 per dataset."""
    out = []
    overall = (metrics.groupby(["method_label", "system_type"])["R@10"]
               .mean().sort_values(ascending=False))
    if not overall.empty:
        (label, stype), val = overall.index[0], overall.iloc[0]
        out.append(li(f"Strongest on average: <b>{escape(label)}</b> "
                      f"({escape(stype)}), mean Recall@10 = {val:.3f}."))
    best = read_csv(plots_dir, "q1_best_methods_by_dataset.csv")
    best = best[best["criterion"] == "Best Recall@10"] if not best.empty else best
    if not best.empty:
        per = ", ".join(f"{escape(r['dataset'])}: {escape(str(r['method_or_tool']))} "
                        f"({r['value']:.2f})" for _, r in best.iterrows())
        out.append(li(f"Best per dataset (Recall@10) — {per}."))
    return "".join(out)


def findings_q3(metrics: pd.DataFrame) -> str:
    """Summarize the weakest speed/performance trade-off."""
    timed = metrics[metrics["avg_query_time_sec"].notna()
                    & (metrics["avg_query_time_sec"] > 0)]
    if timed.empty:
        return ""
    per_method = (timed.groupby("method_label")
                  .agg(t=("avg_query_time_sec", "mean"), r=("R@10", "mean")))
    slow = per_method.sort_values("t", ascending=False).iloc[0]
    best_r = per_method["r"].max()
    name = per_method.sort_values("t", ascending=False).index[0]
    gap = best_r - slow["r"]
    return li(f"Slowest is <b>{escape(name)}</b> at ~{slow['t']:.2f}s/case, "
              f"yet its Recall@10 ({slow['r']:.2f}) trails the best by "
              f"{gap:.2f} — poor speed/accuracy trade-off.")


def findings_q4(plots_dir: Path) -> str:
    """Summarize dataset difficulty from the generated CSV table."""
    df = read_csv(plots_dir, "q4_dataset_difficulty_summary.csv")
    if df.empty or "best_recall10" not in df:
        return ""
    hardest = df.sort_values("best_recall10").iloc[0]
    easiest = df.sort_values("best_recall10", ascending=False).iloc[0]
    return (li(f"Hardest: <b>{escape(str(hardest['dataset']))}</b> "
               f"(best Recall@10 only {hardest['best_recall10']:.2f}).")
            + li(f"Easiest: <b>{escape(str(easiest['dataset']))}</b> "
                 f"(best Recall@10 {easiest['best_recall10']:.2f})."))


def findings_q5(plots_dir: Path) -> str:
    """Summarize whether validation tools beat RareSim systems."""
    df = read_csv(plots_dir, "q5_difference_table.csv")
    if df.empty or "delta" not in df:
        return ""
    wins = df[df["delta"] > 0]["dataset"].tolist()
    loss = df[df["delta"] <= 0]["dataset"].tolist()
    out = []
    if wins:
        out.append(li("Validation tools win on: "
                      + ", ".join(escape(str(d)) for d in wins) + "."))
    if loss:
        out.append(li("RareSim is competitive or ahead on: "
                      + ", ".join(escape(str(d)) for d in loss) + "."))
    return "".join(out)


def findings_q6(plots_dir: Path) -> str:
    """Summarize the Recall@10 gain from RRF fusion."""
    df = read_csv(plots_dir, "q6_ensemble_gain.csv")
    if df.empty or "gain" not in df:
        return ""
    helps = df[df["gain"] > 0]["dataset"].tolist()
    avg = df["gain"].mean()
    text = (f"RRF fusion changes Recall@10 by {avg:+.3f} on average; "
            f"it helps on {len(helps)}/{len(df)} datasets")
    text += (": " + ", ".join(escape(str(d)) for d in helps) + "." if helps else ".")
    return li(text)


def findings_q7(plots_dir: Path) -> str:
    """Summarize the strongest method family overall."""
    df = read_csv(plots_dir, "q7_family_overview.csv")
    if df.empty:
        return ""
    df = df.sort_values("mean_recall10", ascending=False)
    top = df.iloc[0]
    return li(f"Strongest family overall: <b>{escape(str(top['method_family']))}</b> "
              f"(mean Recall@10 {top['mean_recall10']:.2f}).")


def findings_box(items: str) -> str:
    """Render the key-findings box when findings are available."""
    if not items.strip():
        return ""
    return (f'<div class="findings"><h3>Key findings</h3>'
            f'<ul>{items}</ul></div>')


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def build_html(plots_dir: Path) -> str:
    """Build the complete self-contained HTML report."""
    metrics = read_csv(plots_dir, "combined_metrics.csv")
    if metrics.empty:
        raise FileNotFoundError(
            f"combined_metrics.csv not found in {plots_dir}. "
            "Run plot_evaluation_questions.py first.")
    present = datasets_present(metrics)

    sections = []

    # Q1
    q1_img = first_existing(plots_dir, ["q1_best_method_recall10_heatmap.png"])
    tables = "".join(
        f'<details class="ds" open><summary>{escape(d)} — full metrics</summary>'
        f'{metric_table(metrics, d)}</details>' for d in present)
    sections.append(section(
        "q1", "Q1 · Which method performs best?",
        "Recall@10 across every method and dataset. The heatmap gives the "
        "overview; the per-dataset tables below give the exact numbers "
        "(R@1, R@5, R@10, MRR, NDCG@10, coverage, runtime).",
        findings_box(findings_q1(metrics, plots_dir)),
        (img_block(q1_img, "Recall@10 by method and dataset") if q1_img else "")
        + f'<div class="tables">{tables}</div>'))

    # Q2
    q2_imgs = sorted(plots_dir.glob("q2_*recall_curve_top_methods.png"))
    sections.append(section(
        "q2", "Q2 · Which method ranks the correct disease highest?",
        "Recall@k curves. A method that rises steeply at @1–@3 puts the correct "
        "disease near the top; one that only catches up by @10 ranks it lower.",
        "", "".join(img_block(p, p.stem.replace("_", " ")) for p in q2_imgs)))

    # Q3
    q3_imgs = sorted(plots_dir.glob("q3_*speed_vs_recall10.png"))
    sections.append(section(
        "q3", "Q3 · Which method is too slow for its performance?",
        "Average runtime per case vs Recall@10 (log x-axis). Upper-left"
        "means accurate and fast.",
        findings_box(findings_q3(metrics)),
        "".join(img_block(p, p.stem.replace("_", " ")) for p in q3_imgs)))

    # Q4
    q4_img = first_existing(plots_dir, ["q4_dataset_difficulty.png"])
    sections.append(section(
        "q4", "Q4 · Are some datasets much harder?",
        "Best and mean Recall@10 per dataset. Low bars mark datasets where even "
        "the best system struggles.",
        findings_box(findings_q4(plots_dir)),
        img_block(q4_img, "Dataset difficulty") if q4_img else ""))

    # Q5
    q5_imgs = [p for p in [
        first_existing(plots_dir, ["q5_validation_vs_raresim_best_recall10.png"]),
        first_existing(plots_dir, ["q5_validation_minus_raresim_difference.png"]),
    ] if p]
    sections.append(section(
        "q5", "Q5 · Do validation tools beat RareSim methods?",
        "Best Recall@10 per system type, and the gap (validation best − RareSim "
        "best). Positive = validation tool ahead; negative = RareSim ahead.",
        findings_box(findings_q5(plots_dir)),
        "".join(img_block(p, p.stem.replace("_", " ")) for p in q5_imgs)))

    # Q6
    q6_img = first_existing(plots_dir, ["q6_ensemble_vs_single.png"])
    sections.append(section(
        "q6", "Q6 · Does combining methods (RRF) help?",
        "Best RRF ensemble vs the best single RareSim method per dataset.",
        findings_box(findings_q6(plots_dir)),
        img_block(q6_img, "Ensemble vs best single method") if q6_img else ""))

    # Q7
    q7_img = first_existing(plots_dir, ["q7_family_overview.png"])
    sections.append(section(
        "q7", "Q7 · How do method families compare?",
        "Mean Recall@10 by family (semantic, set-based, transformer, ensemble, "
        "validation tool, …) — the big-picture summary.",
        findings_box(findings_q7(plots_dir)),
        img_block(q7_img, "Method family overview") if q7_img else ""))

    nav = "".join(
        f'<a href="#{sid}">{escape(short)}</a>'
        for sid, short in [("q1", "Q1 Best"), ("q2", "Q2 Rank"), ("q3", "Q3 Speed"),
                           ("q4", "Q4 Difficulty"), ("q5", "Q5 vs tools"),
                           ("q6", "Q6 Ensemble"), ("q7", "Q7 Families")])

    legend = "".join(
        f'<span class="leg"><span class="dot" style="background:{c}"></span>{escape(t)}</span>'
        for t, c in SYSTEM_TYPE_COLORS.items())

    return PAGE.format(
        nav=nav, legend=legend,
        datasets=", ".join(escape(d) for d in present),
        n_datasets=len(present),
        sections="".join(sections),
    )


def section(sid: str, title: str, *args: str) -> str:
    """section(sid, title, [tag], description, findings, body)."""
    tag = ""
    rest = list(args)
    description, findings, body = (rest + ["", "", ""])[:3]
    return (f'<section id="{sid}"><h2>{escape(title)}{tag}</h2>'
            f'<p class="desc">{escape(description)}</p>{findings}{body}</section>')


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RareSim — Benchmark Evaluation</title>
<style>
  :root {{
    --ink:#1c2733; --muted:#5b6b7a; --line:#e3e8ee; --bg:#f6f8fa;
    --card:#ffffff; --accent:#2a7f9e;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;
    line-height:1.55;
  }}
  header.hero {{
    background:linear-gradient(120deg,#1c2733,#243b4a);
    color:#fff; padding:40px 28px 28px;
  }}
  header.hero .wrap {{ max-width:1080px; margin:0 auto; }}
  header.hero h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.4px; }}
  header.hero p {{ margin:0; color:#c7d4de; font-size:15px; }}
  header.hero .meta {{ margin-top:14px; font-size:13px; color:#9fb3c0; }}
  header.hero .legend {{ margin-top:14px; display:flex; gap:18px; flex-wrap:wrap; }}
  .leg {{ font-size:13px; color:#d6e0e8; display:flex; align-items:center; gap:6px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  nav {{
    position:sticky; top:0; z-index:5; background:rgba(255,255,255,.92);
    backdrop-filter:blur(6px); border-bottom:1px solid var(--line);
  }}
  nav .wrap {{ max-width:1080px; margin:0 auto; padding:10px 28px;
    display:flex; gap:6px; flex-wrap:wrap; }}
  nav a {{ font-size:13px; color:var(--muted); text-decoration:none;
    padding:5px 11px; border-radius:999px; }}
  nav a:hover {{ background:#eef3f7; color:var(--accent); }}
  main {{ max-width:1080px; margin:0 auto; padding:8px 28px 60px; }}
  section {{
    background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:24px 26px; margin:26px 0;
  }}
  h2 {{ font-size:21px; margin:0 0 4px; display:flex; align-items:center; gap:10px; }}
  .tag {{ font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:.5px; color:#8a6d3b; background:#fdf3df;
    border:1px solid #ecd9ab; padding:2px 8px; border-radius:999px; }}
  .desc {{ color:var(--muted); margin:4px 0 16px; font-size:14.5px; }}
  figure {{ margin:18px 0; }}
  img {{ max-width:100%; height:auto; border:1px solid var(--line);
    border-radius:10px; background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:6px;
    text-transform:capitalize; }}
  .findings {{ background:#eef6f4; border-left:4px solid var(--accent);
    border-radius:0 8px 8px 0; padding:12px 18px; margin:6px 0 8px; }}
  .findings h3 {{ margin:0 0 6px; font-size:14px; color:var(--accent);
    text-transform:uppercase; letter-spacing:.5px; }}
  .findings ul {{ margin:0; padding-left:18px; }}
  .findings li {{ font-size:14px; margin:3px 0; }}
  .tables {{ margin-top:18px; }}
  details.ds {{ border:1px solid var(--line); border-radius:10px;
    margin:10px 0; overflow:hidden; }}
  details.ds summary {{ cursor:pointer; padding:11px 16px; font-weight:600;
    background:#f3f6f9; font-size:14px; }}
  table.metrics {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.metrics th, table.metrics td {{ padding:8px 10px;
    border-bottom:1px solid var(--line); text-align:center; }}
  table.metrics th {{ background:#fafbfc; color:var(--muted); font-weight:600;
    position:sticky; top:0; }}
  th.method, td.method {{ text-align:left; white-space:nowrap; }}
  td.method .dot {{ margin-right:8px; }}
  td.num {{ font-variant-numeric:tabular-nums; position:relative; }}
  td.best {{ font-weight:700; color:var(--accent); }}
  .bar {{ position:absolute; left:0; top:0; bottom:0; background:#dceff0;
    z-index:0; }}
  .barval {{ position:relative; z-index:1; }}
  .badge {{ font-size:11px; padding:2px 8px; border-radius:999px; }}
  .missing {{ color:#9b1c1c; font-style:italic; }}
  @media print {{
    body {{ background:#fff; }}
    nav {{ display:none; }}
    section {{ box-shadow:none; page-break-inside:avoid; border:1px solid #ddd; }}
    header.hero {{ background:#243b4a !important; -webkit-print-color-adjust:exact; }}
  }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <h1>RareSim — Benchmark Evaluation on PhenoBrain Datasets</h1>
  <p>Method performance, ranking quality, runtime trade-offs, dataset difficulty, and comparison with validation tools.</p>
  <div class="meta">Datasets ({n_datasets}): {datasets} · </div>
  <div class="legend">{legend}</div>
</div></header>
<nav><div class="wrap">{nav}</div></nav>
<main>{sections}</main>
</body>
</html>
"""


def main() -> None:
    """Parse command-line arguments and write the HTML report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", type=Path,
                        default=Path("outputs/evaluation_visual_questions"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/evaluation_visual_questions/evaluation_report.html"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(args.plots), encoding="utf-8")
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
