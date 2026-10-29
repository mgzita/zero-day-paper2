"""
Paper 3, RQ3: where do successfully disguised attacks land?
=========================================================================
Hypothesis (links to Paper 2's calibration analysis): multi-environment
training widens the region the detector treats as normal, so a padded or
delayed attack can hide in a region that looks like benign traffic from
ANOTHER network (CIC-IDS2018 or UNSW-NB15), which Single never learned.

For every Single and Multi detector (4 families x 10 seeds) this script:

  1. Rebuilds each successfully disguised CIC-IDS2017 attack from the grid's
     saved per-flow actions (pad bytes, delay), using the same recomputation
     as the attack itself.
  2. Finds the nearest benign flow (Euclidean, in the detector's scaled input
     space) in a pooled reference set of benign validation traffic from
     CIC-IDS2017, CIC-IDS2018 and UNSW-NB15, and records which network it
     came from.
  3. Does the same for the SAME flows before disguise, so the shift caused
     by the disguise is separated from where the attacks started.

    other_env_adv    = % of disguised flows whose nearest benign neighbour
                       is CIC-IDS2018 or UNSW-NB15 (not CIC-IDS2017)
    other_env_shift  = other_env_adv minus the same % before disguise

It also recomputes the clean margin on a scale that works for every
detector, including the Normalizing Flow whose scores can be negative:

    margin_iqr = (clean score - threshold) / IQR of CIC17 benign scores

and repeats the "does margin explain evasion?" check with it.

Uses saved grid results and existing checkpoints only. Keep it next to
pilot_evasion_ae.py and run_evasion_grid.py.

Run:
    python landing_analysis.py
    python landing_analysis.py --selftest
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch

import pilot_evasion_ae as P
import run_evasion_grid as G

warnings.filterwarnings("ignore")

REF_PER_ENV = 20000      # benign reference flows per network
ENV_LABELS = ["cic17", "cic18", "unsw"]


def build_reference(X_val, rng):
    """Pooled benign reference: equal-sized samples from each network's validation block."""
    n_block = len(X_val) // 3
    parts, labels = [], []
    for k, env in enumerate(ENV_LABELS):
        block = X_val[k * n_block:(k + 1) * n_block]
        idx = rng.choice(len(block), min(REF_PER_ENV, len(block)), replace=False)
        parts.append(block[idx])
        labels += [env] * len(idx)
    return np.vstack(parts).astype(np.float32), np.array(labels)


def nearest_env(Z, ref, ref_labels):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    dist, idx = nn.kneighbors(Z)
    return ref_labels[idx[:, 0]], dist[:, 0]


def rebuild_adversarial(X, flows, tf):
    """Disguised versions of the saved flows (same recomputation as the attack)."""
    z_clean = X[flows["index"].values].astype(np.float64)
    raw = torch.tensor(tf.to_raw(z_clean), dtype=torch.float64)
    pad = torch.tensor(flows["pad_bytes"].values, dtype=torch.float64)
    delay = torch.tensor(flows["delay_us"].values, dtype=torch.float64)
    raw_mod = P.apply_actions(raw, pad, delay, None)
    return P.build_input(z_clean, raw_mod, tf).numpy().astype(np.float32), z_clean.astype(np.float32)


def analyse_model(family, condition, seed, model, flows, X, y, tf, ref, ref_labels):
    with torch.no_grad():
        s_all = model.score(torch.tensor(X, dtype=torch.float32)).numpy()
    benign = s_all[y == 0]
    thr = np.percentile(benign, P.THRESHOLD_PERCENTILE)
    q1, q3 = np.percentile(benign, [25, 75])
    iqr = max(q3 - q1, 1e-12)

    flows = flows.copy()
    flows["margin_iqr"] = (s_all[flows["index"].values] - thr) / iqr
    flows["evaded"] = (flows.min_level >= 0).astype(int)

    row = {"family": family, "condition": condition, "seed": seed,
           "n_caught": len(flows), "evaded_pct": 100 * flows.evaded.mean(),
           "median_margin_iqr": float(flows.margin_iqr.median())}

    ev = flows[flows.evaded == 1]
    if len(ev):
        z_adv, z_clean = rebuild_adversarial(X, ev, tf)
        env_adv, d_adv = nearest_env(z_adv, ref, ref_labels)
        env_cln, _ = nearest_env(z_clean, ref, ref_labels)
        row.update({
            "n_evaded": len(ev),
            "other_env_adv": 100 * np.mean(env_adv != "cic17"),
            "other_env_clean": 100 * np.mean(env_cln != "cic17"),
            "near_cic18_adv": 100 * np.mean(env_adv == "cic18"),
            "near_unsw_adv": 100 * np.mean(env_adv == "unsw"),
            "median_nn_dist_adv": float(np.median(d_adv)),
        })
        row["other_env_shift"] = row["other_env_adv"] - row["other_env_clean"]
    return row, flows[["margin_iqr", "evaded"]].assign(family=family, condition=condition, seed=seed)


def report(models, flows_all, out_dir):
    pd.set_option("display.width", 220)
    models.to_csv(os.path.join(out_dir, "landing_per_model.csv"), index=False)
    cols = ["evaded_pct", "median_margin_iqr", "other_env_clean", "other_env_adv",
            "other_env_shift", "near_cic18_adv", "near_unsw_adv"]
    cols = [c for c in cols if c in models]
    print("\nSuccessfully disguised CIC-IDS2017 attacks: nearest benign traffic comes from which network?")
    print("(other_env = % nearest to CIC-IDS2018 or UNSW-NB15 benign; mean over seeds)\n")
    print(models.groupby(["family", "condition"])[cols].mean().round(1).to_string())

    import statsmodels.formula.api as smf
    m = models.dropna(subset=["other_env_adv"]).copy()
    m["multi"] = (m.condition == "multi").astype(int)
    m["sid"] = m.family + "_" + m.seed.astype(str)
    print("\nSingle vs Multi (paired mixed model: detector fixed, seed random)")
    for col in ["other_env_adv", "other_env_shift", "median_margin_iqr"]:
        if col not in m:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.mixedlm(f"{col} ~ multi + C(family)", m, groups=m.sid).fit(reml=True)
        lo, hi = r.conf_int().loc["multi"]
        print(f"  {col:18s}: Multi effect {r.params['multi']:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}), "
              f"p = {r.pvalues['multi']:.4f}")

    print("\nDoes margin explain evasion? (margin on IQR scale, valid for all four detectors)")
    f = flows_all.copy()
    f["multi"] = (f.condition == "multi").astype(int)
    f["lm"] = np.sign(f.margin_iqr) * np.log1p(np.abs(f.margin_iqr))
    for fam, g in f.groupby("family"):
        if g.evaded.nunique() < 2:
            continue
        m0 = smf.logit("evaded ~ multi", g).fit(disp=0)
        m1 = smf.logit("evaded ~ lm + multi", g).fit(disp=0)
        print(f"  {fam:5s}: margin coef {m1.params['lm']:+.2f} (p={m1.pvalues['lm']:.1e}); "
              f"Multi odds ratio {np.exp(m0.params['multi']):.2f} alone -> "
              f"{np.exp(m1.params['multi']):.2f} after margin")
    print(f"\nSaved {os.path.join(out_dir, 'landing_per_model.csv')}")


def run(jobs, X, y, tf, X_val, out_dir):
    ref, ref_labels = build_reference(X_val, np.random.default_rng(0))
    rows, flow_rows = [], []
    for i, (family, condition, seed, model, flows) in enumerate(jobs, 1):
        row, fr = analyse_model(family, condition, seed, model, flows, X, y, tf, ref, ref_labels)
        rows.append(row)
        flow_rows.append(fr)
        print(f"[{i}/{len(jobs)}] {family} {condition} seed{seed}: "
              f"{row.get('n_evaded', 0)} evaded, {row.get('other_env_adv', float('nan')):.1f}% land near other networks")
    os.makedirs(out_dir, exist_ok=True)
    report(pd.DataFrame(rows), pd.concat(flow_rows, ignore_index=True), out_dir)


def selftest(out_dir):
    _, tf, X_ben, syn = P.load_synthetic()
    X, y = syn["seen_cic17"]
    rng = np.random.default_rng(0)
    X_val = np.vstack([X_ben, X_ben + 0.5, X_ben - 0.5]).astype(np.float32)   # three "networks"
    jobs = []
    for family in G.FAMILIES:
        for condition in ("single", "multi"):
            for seed in (1, 2):
                torch.manual_seed(seed)
                m = G.new_model(family)
                m.eval()
                att = np.where(y == 1)[0][:200]
                flows = pd.DataFrame({"index": att, "min_level": rng.choice([-1, 1], len(att)),
                                      "pad_bytes": rng.uniform(0, 50, len(att)),
                                      "delay_us": rng.uniform(0, 500, len(att))})
                jobs.append((family, condition, seed, m, flows))
    run(jobs, X, y, tf, X_val, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=P.OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(os.path.join(args.out_dir, "landing_selftest"))
        return

    tf, X_val, tests = G.load_shared_data()
    X, y = tests["cic17"]
    flow_dir = os.path.join(args.out_dir, "grid", "flows")
    jobs = []
    for family in G.FAMILIES:
        for condition in ("single", "multi"):
            for seed in G.SEEDS:
                path = os.path.join(flow_dir, f"{family}_{condition}_seed{seed}_both.csv.gz")
                model, ckpt = G.load_model(family, condition, seed)
                if model is None or not os.path.exists(path):
                    print(f"SKIP {family} {condition} seed{seed} (missing {path if model else ckpt})")
                    continue
                f = pd.read_csv(path)
                f = f[f.domain == "seen_cic17/calibrated"]
                if len(f):
                    jobs.append((family, condition, seed, model, f))
    run(jobs, X, y, tf, X_val, args.out_dir)


if __name__ == "__main__":
    main()
