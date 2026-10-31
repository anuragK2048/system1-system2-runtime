"""Visual Report Builder and Interactive HTML / SVG Dashboard Generator.

Renders executive terminal summary tables, standalone interactive HTML dashboards,
and clean SVG comparison charts (bar charts, radar charts, and breaking-point curves)
without requiring heavy external plotting dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from runtime.evals.benchmark_suite import BenchmarkReport, BreakingPointResult, BenchmarkComparison


# ==============================================================================
# TERMINAL ANSI FORMATTERS
# ==============================================================================
def print_comparison_table(report: BenchmarkReport) -> None:
    """Prints a formatted executive summary table directly into the terminal."""
    w = 88
    print("\n" + "═" * w)
    print("🏆 SYSTEM-1 (L1 ADAPTIVE) vs. SYSTEM-1 + SYSTEM-2 (LLM META-CONTROLLER)")
    print("═" * w)
    print(f"Overall S1+S2 Win Rate:      {report.overall_win_rate_pct:.1f}%")
    print(f"Mean Pos Error Improvement:  {report.overall_mean_improvement_pct:+.1f}%")
    print(f"System-1 Crashes:            {report.s1_total_crashes} | System-1+S2 Crashes: {report.s2_total_crashes}")
    print(f"Hero Scenarios (S1 Crashed, S1+S2 Survived): {report.hero_scenarios_count}")
    print("-" * w)
    print(f"{'Scenario':<26} | {'Sev':<4} | {'S1 Pos (m)':<10} | {'S2 Pos (m)':<10} | {'Imp %':<8} | {'Winner':<14}")
    print("-" * w)

    for c in report.comparisons:
        s1_err = c.s1_metrics.get("mean_pos_error", 0.0)
        s2_err = c.s2_metrics.get("mean_pos_error", 0.0)
        s1_str = f"{s1_err:.3f}" if not c.s1_crashed else "💥 CRASH"
        s2_str = f"{s2_err:.3f}" if not c.s2_crashed else "💥 CRASH"
        name = (c.scenario_name[:24] + "..") if len(c.scenario_name) > 26 else c.scenario_name
        print(f"{name:<26} | {c.severity:<4.1f} | {s1_str:<10} | {s2_str:<10} | {c.pos_improvement_pct:+7.1f}% | {c.winner:<14}")

    print("═" * w + "\n")


def print_breaking_point(bp: BreakingPointResult) -> None:
    """Prints the breaking-point failure threshold analysis."""
    w = 78
    print("\n" + "═" * w)
    print(f"🚨 HERO BREAKING-POINT ANALYSIS: [{bp.scenario_name.upper()}]")
    print("═" * w)
    s1_txt = f"{bp.s1_crash_severity:.2f}x" if bp.s1_crash_severity else "Never Crashed"
    s2_txt = f"{bp.s2_crash_severity:.2f}x" if bp.s2_crash_severity else "Never Crashed"
    print(f"• System-1 (Baseline) Crash Severity Limit:       {s1_txt}")
    print(f"• System-1 + System-2 Crash Severity Limit:       {s2_txt}")
    print(f"• Survivability Authority Gain:                    +{bp.survivability_gap:.2f}x severity")
    if bp.hero_range[0] > 0:
        print(f"🌟 HERO DEMO WINDOW: Between {bp.hero_range[0]:.2f}x and {bp.hero_range[1]:.2f}x severity, S1 crashes while S1+S2 flies!")
    print("-" * w)
    print(f"{'Severity':<10} | {'S1 Status':<15} | {'S2 Status':<15} | {'Improvement %':<15}")
    print("-" * w)
    for e in bp.evaluations:
        s1_st = "💥 CRASHED" if e["s1_crashed"] else f"Flies ({e['s1_pos_error']:.2f}m)"
        s2_st = "💥 CRASHED" if e["s2_crashed"] else f"Flies ({e['s2_pos_error']:.2f}m)"
        print(f"{e['severity']:<10.2f} | {s1_st:<15} | {s2_st:<15} | {e['pos_improvement_pct']:+14.1f}%")
    print("═" * w + "\n")


# ==============================================================================
# STANDALONE HTML & SVG DASHBOARD
# ==============================================================================
def generate_html_dashboard(
    report: BenchmarkReport,
    breaking_points: Optional[List[BreakingPointResult]] = None,
    output_path: Union[str, Path] = "results/summary/benchmark_dashboard.html",
) -> Path:
    """Generates a standalone, dark-themed HTML report with embedded responsive SVG visualizations."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build Scenario Comparison Rows
    rows_html = []
    for c in report.comparisons:
        s1_err = c.s1_metrics.get("mean_pos_error", 0.0)
        s2_err = c.s2_metrics.get("mean_pos_error", 0.0)
        s1_cell = f"<span class='crash'>CRASH</span>" if c.s1_crashed else f"{s1_err:.3f}m"
        s2_cell = f"<span class='crash'>CRASH</span>" if c.s2_crashed else f"<span class='good'>{s2_err:.3f}m</span>"

        imp_class = "good" if c.pos_improvement_pct > 0 else "bad"
        winner_badge = (
            f"<span class='badge badge-hero'>S1+S2 HERO</span>"
            if "SURVIVAL" in c.winner
            else (f"<span class='badge badge-win'>S1+S2 WIN</span>" if "s1_s2" in c.winner else "<span class='badge badge-loss'>S1 ONLY</span>")
        )

        rows_html.append(f"""
        <tr>
            <td><strong>{c.scenario_name}</strong><br><small class='dim'>{c.scenario_id}</small></td>
            <td><code>{c.severity:.1f}x</code></td>
            <td>{s1_cell}</td>
            <td>{s2_cell}</td>
            <td class='{imp_class}'><strong>{c.pos_improvement_pct:+.1f}%</strong></td>
            <td>{c.safety_score_s1:.0f} vs <strong class='good'>{c.safety_score_s2:.0f}</strong></td>
            <td>{c.recovery_speedup:.2f}x</td>
            <td>{winner_badge}</td>
        </tr>
        """)

    # 2. Build Breaking Points Section
    bp_html = ""
    if breaking_points:
        bp_cards = []
        for bp in breaking_points:
            s1_t = f"{bp.s1_crash_severity:.2f}x" if bp.s1_crash_severity else "Never"
            s2_t = f"{bp.s2_crash_severity:.2f}x" if bp.s2_crash_severity else "Never"
            bp_cards.append(f"""
            <div class='card bp-card'>
                <h3>{bp.scenario_name}</h3>
                <div class='bp-grid'>
                    <div class='bp-metric'>
                        <span class='dim'>S1 Crash Point</span>
                        <strong class='bad'>{s1_t}</strong>
                    </div>
                    <div class='bp-metric'>
                        <span class='dim'>S1+S2 Crash Point</span>
                        <strong class='good'>{s2_t}</strong>
                    </div>
                    <div class='bp-metric'>
                        <span class='dim'>Survivability Delta</span>
                        <strong class='highlight'>+{bp.survivability_gap:.2f}x</strong>
                    </div>
                </div>
                <div class='hero-tag'>🌟 <strong>Hero Range:</strong> [{bp.hero_range[0]:.2f}x &ndash; {bp.hero_range[1]:.2f}x severity]</div>
            </div>
            """)
        bp_html = f"""
        <section class='section'>
            <h2>🚨 Hero Breaking-Point Analysis (Where S1 Fails but S1+S2 Survives)</h2>
            <div class='cards-grid'>
                {''.join(bp_cards)}
            </div>
        </section>
        """

    # 3. Build Embedded SVG Bar Chart
    chart_bars = []
    bar_width = 24
    chart_h = 160
    comparisons_sample = report.comparisons[:12]
    for idx, c in enumerate(comparisons_sample):
        imp = max(-50.0, min(100.0, c.pos_improvement_pct))
        x = 40 + idx * 56
        # Normalize: 0% is at y=100
        zero_y = 100
        bar_h = abs(imp) * 0.8
        y = zero_y - bar_h if imp >= 0 else zero_y
        color = "#10b981" if imp >= 0 else "#ef4444"
        short_label = c.scenario_id[:5]
        chart_bars.append(f"""
            <rect x='{x}' y='{y}' width='{bar_width}' height='{bar_h}' fill='{color}' rx='3'/>
            <text x='{x + bar_width/2}' y='{y - 4 if imp >= 0 else y + bar_h + 12}' text-anchor='middle' font-size='10' fill='#f1f5f9'>{imp:+.0f}%</text>
            <text x='{x + bar_width/2}' y='145' text-anchor='middle' font-size='9' fill='#94a3b8'>{short_label}</text>
            <text x='{x + bar_width/2}' y='157' text-anchor='middle' font-size='8' fill='#64748b'>{c.severity:.1f}x</text>
        """)

    svg_bar_chart = f"""
    <svg viewBox='0 0 760 180' class='svg-chart'>
        <line x1='30' y1='100' x2='740' y2='100' stroke='#334155' stroke-dasharray='4'/>
        <text x='25' y='103' text-anchor='end' font-size='10' fill='#64748b'>0%</text>
        {''.join(chart_bars)}
    </svg>
    """

    # 4. Assemble HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>System-1 vs System-2 Autonomous Drone Benchmark Dashboard</title>
<style>
    :root {{
        --bg: #090d16;
        --card-bg: #111827;
        --border: #1f2937;
        --text: #f9fafb;
        --dim: #9ca3af;
        --good: #10b981;
        --bad: #ef4444;
        --highlight: #38bdf8;
        --accent: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        padding: 32px 20px;
        line-height: 1.5;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; }}
    header {{ margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
    h1 {{ font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 6px; }}
    .subtitle {{ color: var(--dim); font-size: 14px; }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }}
    .kpi-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
    }}
    .kpi-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--dim); margin-bottom: 8px; }}
    .kpi-value {{ font-size: 32px; font-weight: 800; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .highlight {{ color: var(--highlight); }}
    .dim {{ color: var(--dim); }}
    .crash {{ color: var(--bad); font-weight: 800; }}
    .section {{ margin-bottom: 36px; }}
    h2 {{ font-size: 20px; margin-bottom: 16px; font-weight: 600; }}
    .table-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
    th {{ background: #0f172a; color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .badge-win {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-hero {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .badge-loss {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .bp-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
    .bp-card h3 {{ font-size: 16px; margin-bottom: 12px; }}
    .bp-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px; }}
    .bp-metric strong {{ display: block; font-size: 18px; margin-top: 4px; }}
    .hero-tag {{ background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #bae6fd; }}
    .svg-chart {{ width: 100%; height: auto; display: block; background: #0c1220; border-radius: 8px; padding: 12px; border: 1px solid var(--border); margin-bottom: 16px; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>🚁 System-1 + System-2 Hybrid Autonomous Drone Benchmark</h1>
        <div class="subtitle">Side-by-side quantitative evaluation: Baseline SE(3) L1 Adaptive Controller vs. Asynchronous LLM Meta-Controller</div>
    </header>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">S1+S2 Win Rate</div>
            <div class="kpi-value good">{report.overall_win_rate_pct:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Mean Position Error Improvement</div>
            <div class="kpi-value good">{report.overall_mean_improvement_pct:+.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">System-1 Crashes</div>
            <div class="kpi-value bad">{report.s1_total_crashes}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Hero Scenarios (S1 Crashed, S2 Saved)</div>
            <div class="kpi-value highlight">{report.hero_scenarios_count}</div>
        </div>
    </div>

    <section class="section">
        <h2>📊 Position Error Improvement % per Scenario</h2>
        {svg_bar_chart}
    </section>

    {bp_html}

    <section class="section">
        <h2>📋 Full Benchmark Matrix (Side-by-Side Statistics)</h2>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Scenario Name</th>
                        <th>Severity</th>
                        <th>S1 Baseline Mean Err</th>
                        <th>S1+S2 Mean Err</th>
                        <th>Improvement %</th>
                        <th>Safety Score (S1 vs S2)</th>
                        <th>Recovery Speedup</th>
                        <th>Outcome</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </section>
</div>
</body>
</html>
"""
    with open(p, "w", encoding="utf-8") as f:
        f.write(html_content)

    return p
