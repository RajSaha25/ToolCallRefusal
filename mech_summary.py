#!/usr/bin/env python3
"""Print a cross-model mech-interp summary from interp_artifacts/<model>/ JSONs."""
import json, glob, os
for d in sorted(glob.glob("interp_artifacts/*/")):
    m = os.path.basename(d.rstrip("/"))
    p1 = os.path.join(d, "interp_summary.json")
    p2 = os.path.join(d, "interp_scaled_summary.json")
    if not os.path.exists(p1):
        continue
    a = json.load(open(p1))
    if "MODEL_ID" not in a:
        continue  # skip old-format Qwen handoff files without MODEL_ID
    print(f"=== {m}  (layer {a['LAYER']}/{a['N_LAYERS']}, proj_gap {a['proj_gap']:.0f}) ===")
    pn = a["paired"]["Tool-enabled normal"]; ps = a["paired"]["Tool-enabled stress"]
    print(f"  suppression (No-tool->Tool proj drop): normal d={pn['mean_delta']:.1f} t={pn['t']:.1f} | stress d={ps['mean_delta']:.1f} t={ps['t']:.1f}")
    print(f"  ablation {a['ablation']['base']:.0%}->{a['ablation']['ablated']:.0%} | addition {a['addition']['base']:.0%}->{a['addition']['added']:.0%}  (part1, n=120)")
    if os.path.exists(p2):
        b = json.load(open(p2))
        au = b.get("auc", {}); pa = b.get("patching", {})
        print(f"  [part2] ablation {b['ablation']['base_refuse']:.0%}->{b['ablation']['ablated_refuse']:.0%} | AUC {au.get('auc')} {au.get('auc_ci')} (n_unsafe={au.get('n_unsafe')}) | patching {pa.get('flipped_to_safe')}/{pa.get('n_unsafe_baseline')}={pa.get('rate')}")
    else:
        print("  [part2] (none)")
