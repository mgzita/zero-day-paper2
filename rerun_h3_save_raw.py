"""
Re-run ONLY the H3 retraining ablation and SAVE raw AUROC (full / stable / sensitive / middle) per seed.
Needed for reviewer comments #3/#7: the original IEC-104, SMD and Flow H3 files stored only the |AUROC-0.5| deltas.

It imports your existing pipeline modules and calls their own train_and_score_subset(), so the models, features and
seeds are the ones used for the manuscript. AE/DAE/SVDD reuse their cached checkpoints (fast). Flow retrains each
subset (a few minutes per seed). Per-seed results are cached, so you can stop and restart.

It also CHECKS reproduction: recomputed delta_stable / delta_sensitive vs the stored ones. Any drift is printed.

Run from the repo root (where the pipeline .py files are):
  python -u rerun_h3_save_raw.py --which ae_iec104 ae_smd
  python -u rerun_h3_save_raw.py --which flow_unsw flow_iec104 flow_smd
Output: paper2\pilot\h3_raw_auroc_<which>.json
"""
import argparse, importlib, json, os, sys
import numpy as np

sys.path.insert(0, os.getcwd())

CFG = {
    # key: (module, kind, loader, stored delta file, target label)
    "ae_iec104":  ("ae_dae_svdd_pipeline_iec104", "ae",   "load_iec104_eval", "ae_dae_svdd_h3_iec104.json"),
    "ae_smd":     ("ae_dae_svdd_pipeline_smd",    "ae",   "load_iec104_eval", "ae_dae_svdd_h3_smd.json"),
    "flow_unsw":  ("flow_full_pipeline",          "flow", "load_unsw_eval",   "flow_h3_ablation.json"),
    "flow_iec104": ("flow_full_pipeline_iec104",  "flow", "load_iec104_eval", "flow_h3_ablation_iec104.json"),
    "flow_smd":   ("flow_full_pipeline_smd",      "flow", "load_iec104_eval", "flow_h3_ablation_smd.json"),
}


def run(key):
    modname, kind, loader, stored_name = CFG[key]
    m = importlib.import_module(modname)
    pilot = m.PILOT_DIR
    out_path = os.path.join(pilot, f"h3_raw_auroc_{key}.json")
    done = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path)):
            done[(r["family"], r["seed"])] = r
    stored = {}
    sp = os.path.join(pilot, stored_name)
    if os.path.exists(sp):
        for r in json.load(open(sp)):
            stored[(r.get("family", "flow"), r["seed"])] = r

    X_eval, y_eval = getattr(m, loader)()
    fams = m.MODEL_FAMILIES if kind == "ae" else ["flow"]
    subsets = [("full", m.DECORRELATED_FEATURES), ("stable", m.STABLE_V2),
               ("sensitive", m.SENSITIVE_V2), ("middle", m.MIDDLE_V2)]
    if kind == "flow":
        Xtr, Xva = m.load_lodo3_train_val()

    for fam in fams:
        for seed in m.SEEDS_TO_RUN:
            if (fam, seed) in done:
                print(f"[cache] {key} {fam} seed {seed}")
                continue
            res = {}
            for name, feats in subsets:
                if kind == "ae":
                    sep, au = m.train_and_score_subset(seed, fam, feats, X_eval, y_eval)
                else:
                    sep, au = m.train_and_score_subset(seed, feats, Xtr, Xva, X_eval, y_eval)
                res[name] = (float(sep), float(au))
            rec = {"seed": seed, "family": fam,
                   "auroc_full": res["full"][1], "auroc_stable": res["stable"][1],
                   "auroc_sensitive": res["sensitive"][1], "auroc_middle": res["middle"][1],
                   "delta_stable": res["full"][0] - res["stable"][0],
                   "delta_sensitive": res["full"][0] - res["sensitive"][0],
                   "delta_middle": res["full"][0] - res["middle"][0]}
            st = stored.get((fam, seed))
            if st and "delta_stable" in st:
                rec["drift_vs_stored"] = max(abs(rec["delta_stable"] - st["delta_stable"]),
                                             abs(rec["delta_sensitive"] - st["delta_sensitive"]))
            done[(fam, seed)] = rec
            json.dump(list(done.values()), open(out_path, "w"), indent=2)
            print(f"[{key}] {fam} seed {seed}: AUROC full={rec['auroc_full']:.3f} stable={rec['auroc_stable']:.3f} "
                  f"sens={rec['auroc_sensitive']:.3f}  drift vs stored={rec.get('drift_vs_stored', float('nan')):.2e}")
    drifts = [r["drift_vs_stored"] for r in done.values() if "drift_vs_stored" in r]
    if drifts:
        print(f"\n{key}: max drift vs stored deltas = {max(drifts):.2e} over {len(drifts)} seeds "
              f"({'reproduces' if max(drifts) < 1e-3 else 'DOES NOT reproduce exactly: report both'})")
    print("Saved", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", required=True, choices=list(CFG))
    for k in ap.parse_args().which:
        run(k)
