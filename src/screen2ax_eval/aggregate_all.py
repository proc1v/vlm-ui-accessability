import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/inference_results_all")
GT_KIND = sys.argv[2] if len(sys.argv) > 2 else "silver"

rows = {}
# Hierarchy
for p in sorted(ROOT.glob(f"*/val/metrics_{GT_KIND}.json")):
    run = p.parts[-3]
    d = json.loads(p.read_text())
    rows.setdefault(run, {"run": run})
    rows[run].update({
        "n":               d.get("n_total"),
        "parse_rate":      d.get("parse_rate"),
        "edge_f1":         d["edge_f1"]["mean"],
        "leaves_f1":       d["leaves_f1"]["mean"],
        "container_match": d["container_match"]["mean"],
        "edge_f1_micro":   d["edge_f1_micro"]["f1"],
        "leaves_f1_micro": d["leaves_f1_micro"]["f1"],
    })
# Detection
for p in sorted(ROOT.glob(f"*/val/metrics_detection_{GT_KIND}.json")):
    run = p.parts[-3]
    d = json.loads(p.read_text())
    rows.setdefault(run, {"run": run})
    rows[run].update({
        "det_macro_p":  d["macro"]["precision"],
        "det_macro_r":  d["macro"]["recall"],
        "det_macro_f1": d["macro"]["f1"],
        "det_macro_ap50": d["macro"]["ap50"],
        "det_micro_f1": d.get("micro", {}).get("f1"),
    })

ordered = sorted(rows.values(), key=lambda r: r["run"])

# --- Hierarchy table ---
hdr = f'{"run":<28} {"n":>3} {"parse":>6} {"edge":>7} {"leaves":>7} {"cont":>6} {"edge-µ":>7} {"leaves-µ":>9}'
print(hdr); print("-" * len(hdr))
for r in ordered:
    if "edge_f1" not in r: continue
    print(f'{r["run"]:<28} {r["n"]:>3} {r["parse_rate"]:>6.3f} '
          f'{r["edge_f1"]:>7.3f} {r["leaves_f1"]:>7.3f} {r["container_match"]:>6.3f} '
          f'{r["edge_f1_micro"]:>7.3f} {r["leaves_f1_micro"]:>9.3f}')

# --- Detection table ---
print()
hdr2 = f'{"run":<28} {"P":>7} {"R":>7} {"F1":>7} {"AP50":>7} {"F1-µ":>7}'
print(hdr2); print("-" * len(hdr2))
for r in ordered:
    if "det_macro_p" not in r: continue
    print(f'{r["run"]:<28} {r["det_macro_p"]:>7.3f} {r["det_macro_r"]:>7.3f} '
          f'{r["det_macro_f1"]:>7.3f} {r["det_macro_ap50"]:>7.3f} '
          f'{(r.get("det_micro_f1") or 0):>7.3f}')

out = ROOT / f"summary_{GT_KIND}.json"
out.write_text(json.dumps(ordered, indent=2))
print(f"\nWrote {out}")
