"""
Re-run ONLY the H3 retraining ablation and SAVE raw AUROC (full / stable / sensitive / middle) per seed.

*** For ae_iec104 / ae_smd run patch_subset_checkpoints.py FIRST. Before the patch, Sensitive and Middle reused the
    Stable checkpoint (same file name ..._dim5.pt), so those models were never trained on their own features. ***

It imports your existing pipeline modules and calls their own train_and_score_subset(), so models, features and seeds
are the manuscript's. Per-seed results are cached (delete h3_raw_auroc_<which>.json to force a clean rerun).
Drift vs the stored deltas is printed: it should be ~0 for flow_*; for ae_* it is EXPECTED to be non-zero after the fix.

Run from the repo root:
  python -u rerun_h3_save_raw.py --which ae_iec104 ae_smd --write-corrected-h3
  python -u rerun_h3_save_raw.py --which flow_unsw
Output: paper2/pilot/h3_raw_auroc_<which>.json
With --write-corrected-h3 (ae_* only) the old ae_dae_svdd_h3_<target>.json is kept as *.contaminated.bak and replaced
by the corrected deltas, so paper2_followup_tests.py and the stats scripts pick up the corrected numbers.
"""
import argparse, importlib, json, os, sys

sys.path.insert(0, os.getcwd())

CFG = {
    "ae_iec104":   ("ae_dae_svdd_pipeline_iec104", "ae",   "load_iec104_eval", "ae_dae_svdd_h3_iec104.json"),
    "ae_smd":      ("ae_dae_svdd_pipeline_smd",    "ae",   "load_iec104_eval", "ae_dae_svdd_h3_smd.json"),
    "flow_unsw":   ("flow_full_pipeline",          "flow", "load_unsw_eval",   "flow_h3_ablation.json"),
    "flow_iec104": ("flow_full_pipeline_iec104",   "flow", "load_iec104_eval", "flow_h3_ablation_iec104.json"),
    "flow_smd":    ("flow_full_pipeline_smd",      "flow", "load_iec104_eval", "flow_h3_ablation_smd.json"),
}
WRITE_CORRECTED = False


def run(key):
    modname, kind, loader, stored_name = CFG[key]
    m = importlib.import_module(modname)
    if kind == "ae" and "subset_tag" not in open(m.__file__, encoding="utf-8").read():
        raise SystemExit(f"{modname}.py is not patched. Run: python patch_subset_checkpoints.py")
    pilot = m.PILOT_DIR
    out_path = os.path.join(pilot, f"h3_raw_auroc_{key}.json")
    done = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path)):
            done[(r["family"], r["seed"])] = r
    stored = {}
    sp = os.path.join(pilot, stored_name)
    src = sp + ".contaminated.bak" if os.path.exists(sp + ".contaminated.bak") else sp
    if os.path.exists(src):
        for r in json.load(open(src)):
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
                  f"sens={rec['auroc_sensitive']:.3f} mid={rec['auroc_middle']:.3f}  "
                  f"drift vs stored={rec.get('drift_vs_stored', float('nan')):.2e}")
    drifts = [r["drift_vs_stored"] for r in done.values() if "drift_vs_stored" in r]
    if drifts:
        print(f"\n{key}: max drift vs stored deltas = {max(drifts):.2e} over {len(drifts)} seeds")
    print("Saved", out_path)
    if kind == "ae" and WRITE_CORRECTED:
        bak = sp + ".contaminated.bak"
        if os.path.exists(sp) and not os.path.exists(bak):
            os.replace(sp, bak)
        keep = ("seed", "family", "delta_stable", "delta_sensitive", "delta_middle")
        json.dump([{k: r[k] for k in keep} for r in done.values()], open(sp, "w"), indent=2)
        print(f"Wrote corrected H3 file {sp} (old file kept as {os.path.basename(bak)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", required=True, choices=list(CFG))
    ap.add_argument("--write-corrected-h3", action="store_true")
    a = ap.parse_args()
    WRITE_CORRECTED = a.write_corrected_h3
    for k in a.which:
        run(k)
