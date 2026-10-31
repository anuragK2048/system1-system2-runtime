"""Configuration & Evaluation Run Comparator.

Compares two System-2 configurations or historical evaluation runs:
1. Calculates exact parameter diffs between configuration files.
2. Compares scenario-by-scenario physical performance metrics.
3. Generates executive terminal comparison tables and an interactive HTML report.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from runtime.sim.config import System2MasterConfig
from runtime.evals.benchmark_suite import BenchmarkReport, BenchmarkComparison, BenchmarkSuite
from runtime.evals.metrics import compute_improvement_pct


def diff_configs(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """Recursively computes differences between two configuration dictionaries."""
    diffs = []
    all_keys = set(cfg_a.keys()).union(set(cfg_b.keys()))
    for k in sorted(all_keys):
        if k.startswith("_"):
            continue
        val_a = cfg_a.get(k)
        val_b = cfg_b.get(k)
        path = f"{prefix}.{k}" if prefix else k

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            diffs.extend(diff_configs(val_a, val_b, prefix=path))
        elif val_a != val_b:
            diffs.append({
                "param": path,
                "val_a": val_a,
                "val_b": val_b,
            })
    return diffs


def compare_reports(
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
    name_a: str = "Config A",
    name_b: str = "Config B",
) -> Dict[str, Any]:
    """Compares two benchmark reports scenario by scenario."""
    # 1. Config differences
    cfg_diffs = []
    if report_a.config_dict and report_b.config_dict:
        cfg_diffs = diff_configs(report_a.config_dict, report_b.config_dict)

    # 2. Match scenarios
    map_b = {(c.scenario_id, round(c.severity, 2)): c for c in report_b.comparisons}

    scenario_comparisons = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for ca in report_a.comparisons:
        key = (ca.scenario_id, round(ca.severity, 2))
        cb = map_b.get(key)
        if not cb:
            continue

        err_a = ca.s2_metrics.get("mean_pos_error", 0.0)
        err_b = cb.s2_metrics.get("mean_pos_error", 0.0)

        # Delta %: positive means B is better (lower error) than A
        delta_pct = compute_improvement_pct(err_a, err_b)

        if ca.s2_crashed and not cb.s2_crashed:
            sub_winner = name_b + " (Saved Crash)"
            b_wins += 1
        elif cb.s2_crashed and not ca.s2_crashed:
            sub_winner = name_a + " (A survived)"
            a_wins += 1
        elif delta_pct > 3.0:
            sub_winner = name_b
            b_wins += 1
        elif delta_pct < -3.0:
            sub_winner = name_a
            a_wins += 1
        else:
            sub_winner = "Tie"
            ties += 1

        scenario_comparisons.append({
            "scenario_id": ca.scenario_id,
            "scenario_name": ca.scenario_name,
            "severity": ca.severity,
            "err_a": err_a,
            "err_b": err_b,
            "crashed_a": ca.s2_crashed,
            "crashed_b": cb.s2_crashed,
            "delta_pct": round(delta_pct, 1),
            "safety_a": ca.safety_score_s2,
            "safety_b": cb.safety_score_s2,
            "winner": sub_winner,
        })

    net_imp = (
        float(sum(sc["delta_pct"] for sc in scenario_comparisons) / len(scenario_comparisons))
        if scenario_comparisons
        else 0.0
    )

    return {
        "name_a": name_a,
        "name_b": name_b,
        "config_diffs": cfg_diffs,
        "scenario_comparisons": scenario_comparisons,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "net_improvement_b_over_a_pct": round(net_imp, 1),
    }


def print_comparison(comp_data: Dict[str, Any]) -> None:
    """Prints a clean CLI comparison of the two configurations."""
    na = comp_data["name_a"]
    nb = comp_data["name_b"]
    w = 88

    print("\n" + "═" * w)
    print(f"⚖️ CONFIGURATION COMPARISON: [{na}] vs [{nb}]")
    print("═" * w)

    diffs = comp_data["config_diffs"]
    print(f"\n🔧 PARAMETER DIFFERENCES ({len(diffs)} changed):")
    if not diffs:
        print("   (Configurations have identical parameters)")
    else:
        print(f"   {'Parameter':<40} | {na:<18} | {nb:<18}")
        print("   " + "-" * (w - 3))
        for d in diffs:
            print(f"   {d['param']:<40} | {str(d['val_a']):<18} | {str(d['val_b']):<18}")

    print("\n" + "-" * w)
    print(f"📊 SIDE-BY-SIDE FLIGHT DYNAMICS COMPARISON:")
    print("-" * w)
    print(f"{'Scenario':<26} | {'Sev':<4} | {na + ' (m)':<12} | {nb + ' (m)':<12} | {'Delta %':<9} | {'Winner':<14}")
    print("-" * w)

    for sc in comp_data["scenario_comparisons"]:
        a_str = f"{sc['err_a']:.3f}" if not sc["crashed_a"] else "💥 CRASH"
        b_str = f"{sc['err_b']:.3f}" if not sc["crashed_b"] else "💥 CRASH"
        name = (sc["scenario_name"][:24] + "..") if len(sc["scenario_name"]) > 26 else sc["scenario_name"]
        print(f"{name:<26} | {sc['severity']:<4.1f} | {a_str:<12} | {b_str:<12} | {sc['delta_pct']:+8.1f}% | {sc['winner']:<14}")

    print("-" * w)
    print(f"🏆 VERDICT: [{nb}] won {comp_data['b_wins']} | [{na}] won {comp_data['a_wins']} | Ties: {comp_data['ties']}")
    print(f"📈 Net Average Tracking Improvement ({nb} vs {na}): {comp_data['net_improvement_b_over_a_pct']:+.1f}%")
    print("═" * w + "\n")


def generate_comparison_html(
    comp_data: Dict[str, Any],
    output_path: Union[str, Path] = "results/summary/config_comparison.html",
) -> Path:
    """Generates an interactive HTML dashboard showing configuration diff and performance delta."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    na = comp_data["name_a"]
    nb = comp_data["name_b"]

    # Parameter diff rows
    diff_rows = []
    for d in comp_data["config_diffs"]:
        diff_rows.append(f"""
        <tr>
            <td><code>{d['param']}</code></td>
            <td><span class='dim'>{d['val_a']}</span></td>
            <td><strong class='highlight'>{d['val_b']}</strong></td>
        </tr>
        """)

    # Scenario table rows
    sc_rows = []
    for sc in comp_data["scenario_comparisons"]:
        a_cell = f"<span class='crash'>CRASH</span>" if sc["crashed_a"] else f"{sc['err_a']:.3f}m"
        b_cell = f"<span class='crash'>CRASH</span>" if sc["crashed_b"] else f"{sc['err_b']:.3f}m"
        d_cls = "good" if sc["delta_pct"] > 0 else ("bad" if sc["delta_pct"] < 0 else "dim")

        badge = (
            f"<span class='badge badge-win'>{nb}</span>"
            if nb in sc["winner"]
            else (f"<span class='badge badge-loss'>{na}</span>" if na in sc["winner"] else "<span class='badge'>TIE</span>")
        )

        sc_rows.append(f"""
        <tr>
            <td><strong>{sc['scenario_name']}</strong></td>
            <td><code>{sc['severity']:.1f}x</code></td>
            <td>{a_cell}</td>
            <td>{b_cell}</td>
            <td class='{d_cls}'><strong>{sc['delta_pct']:+.1f}%</strong></td>
            <td>{sc['safety_a']:.0f} vs {sc['safety_b']:.0f}</td>
            <td>{badge}</td>
        </tr>
        """)

    net_cls = "good" if comp_data["net_improvement_b_over_a_pct"] > 0 else "bad"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Configuration Comparison: {na} vs {nb}</title>
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
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 32px 20px; line-height: 1.5; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{ margin-bottom: 28px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    h1 {{ font-size: 26px; color: #fff; margin-bottom: 6px; }}
    .subtitle {{ color: var(--dim); font-size: 14px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
    .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
    .kpi-title {{ font-size: 12px; text-transform: uppercase; color: var(--dim); margin-bottom: 6px; }}
    .kpi-value {{ font-size: 32px; font-weight: 800; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .highlight {{ color: var(--highlight); }}
    .dim {{ color: var(--dim); }}
    .crash {{ color: var(--bad); font-weight: 800; }}
    .section {{ margin-bottom: 32px; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; font-weight: 600; }}
    .table-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
    th {{ background: #0f172a; color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr:last-child td {{ border-bottom: none; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: #1e293b; color: #94a3b8; }}
    .badge-win {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-loss {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>⚖️ Configuration A/B Comparison</h1>
        <div class="subtitle">Evaluating <strong>{na}</strong> vs. <strong>{nb}</strong> across all benchmark scenarios</div>
    </header>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Net Improvement ({nb} vs {na})</div>
            <div class="kpi-value {net_cls}">{comp_data['net_improvement_b_over_a_pct']:+.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">{nb} Win Count</div>
            <div class="kpi-value good">{comp_data['b_wins']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">{na} Win Count</div>
            <div class="kpi-value dim">{comp_data['a_wins']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Parameters Changed</div>
            <div class="kpi-value highlight">{len(comp_data['config_diffs'])}</div>
        </div>
    </div>

    <section class="section">
        <h2>🔧 Configuration Parameters Changed</h2>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Parameter Path</th>
                        <th>{na}</th>
                        <th>{nb}</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(diff_rows) if diff_rows else "<tr><td colspan='3' class='dim'>Identical configurations</td></tr>"}
                </tbody>
            </table>
        </div>
    </section>

    <section class="section">
        <h2>📊 Scenario-by-Scenario Performance Comparison</h2>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Severity</th>
                        <th>{na} Mean Err</th>
                        <th>{nb} Mean Err</th>
                        <th>Delta % (Gain)</th>
                        <th>Safety Score</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(sc_rows)}
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
