"""
Verify the two summary power-value claims in the Discussion's Limitations
paragraph, computed precisely rather than eyeballed:
  Claim 1: "post-hoc power is below 0.80 for 33 of 47 tests with a
            computable power value"
  Claim 2: "every finding reported as significant... has power at or
            above 0.73, and all but one exceed 0.85"

Claim 1 is computed by scanning every record in the saved file -- no
manual mapping needed.

Claim 2 requires identifying which records correspond to the 13
findings that survive FDR correction (the master table). This mapping
is stated EXPLICITLY below and printed in full, so it can be checked
by eye rather than trusted blindly.

Run from project root:
    python paper2\\pilot\\verify_discussion_power_claims.py
"""

import json
import os

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

FDR_SURVIVING_13 = [
    ("SMD, AE, H3",            "SMD_h3",            "ae"),
    ("IEC-104, Flow, Repr.",   "flow_representation", "IEC-104"),
    ("UNSW, AE, H3",           "UNSW-NB15_h3",      "ae"),
    ("UNSW, SVDD, H3",         "UNSW-NB15_h3",      "svdd"),
    ("SMD, Flow, Repr.",       "flow_smd",          "repr"),
    ("UNSW, Flow, H3",         "flow_UNSW_live",    "h3"),
    ("UNSW, SVDD, Repr.",      "UNSW-NB15_repr",    "svdd"),
    ("SMD, Flow, H3",          "flow_smd",          "h3"),
    ("SMD, SVDD, H1 (SHAP)",   "SMD_h1",            "svdd"),
    ("UNSW, Flow, H1 (SHAP)",  "flow_UNSW_live",    "h1"),
    ("UNSW, DAE, H3",          "UNSW-NB15_h3",      "dae"),
    ("SMD, Flow, H1 (SHAP)",   "flow_smd",          "h1"),
    ("IEC-104, SVDD, H3",      "IEC-104_h3",        "svdd"),
]


def safe_load(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    data = safe_load(os.path.join(PILOT_DIR, "strengthened_statistics_full.json"))
    if data is None:
        return

    print(f"{'=' * 90}\nCLAIM 1: post-hoc power below 0.80, out of all tests with a computable value\n{'=' * 90}")
    all_powers = []
    for section_key, section_val in data.items():
        if section_key == "fdr_correction" or not isinstance(section_val, dict):
            continue
        for model_key, record in section_val.items():
            if isinstance(record, dict) and "power" in record:
                p = record.get("power")
                if p is not None and p == p:
                    all_powers.append((section_key, model_key, p))

    n_total = len(all_powers)
    n_below_80 = sum(1 for _, _, p in all_powers if p < 0.80)
    print(f"  Total tests with a computable (non-NaN) power value: {n_total}")
    print(f"  Of these, number with power < 0.80: {n_below_80}")
    print(f"  Discussion claims: '33 of 47'")
    match1 = (n_total == 47 and n_below_80 == 33)
    print(f"  {'MATCH' if match1 else 'MISMATCH -- update the Discussion text to ' + str(n_below_80) + ' of ' + str(n_total)}")

    print(f"\n{'=' * 90}\nCLAIM 2: power values among the 13 FDR-surviving findings\n{'=' * 90}")
    print("Explicit mapping used (check this against the master table by eye):\n")
    found_powers = []
    n_nan = 0
    for label, section_key, model_key in FDR_SURVIVING_13:
        record = data.get(section_key, {}).get(model_key)
        if record is None:
            print(f"  [NOT FOUND] {label:28s} -> {section_key} / {model_key}  -- CHECK THIS MAPPING")
            continue
        power = record.get("power")
        p = record.get("p")
        if power is not None and power == power:
            power_str = f"{power:.4f}"
            found_powers.append((label, power))
        else:
            power_str = "NaN (effect too large to compute)"
            n_nan += 1
        print(f"  {label:28s} -> {section_key:22s} / {model_key:8s}  p={p:.5f}  power={power_str}")

    if found_powers:
        min_label, min_power = min(found_powers, key=lambda x: x[1])
        n_below_85 = sum(1 for _, p in found_powers if p < 0.85)
        n_below_73 = sum(1 for _, p in found_powers if p < 0.73)
        print(f"\n  Minimum power among the {len(found_powers)} findings with a computable value: "
              f"{min_power:.4f} ({min_label})")
        print(f"  Number below 0.85: {n_below_85}")
        print(f"  Number below 0.73: {n_below_73}")
        print(f"  Discussion claims: 'all >= 0.73, all but one exceed 0.85'")
        claim_ok = (n_below_73 == 0) and (n_below_85 <= 1)
        print(f"  {'MATCH' if claim_ok else 'MISMATCH -- update the Discussion text'}")
        print(f"  ({n_nan} of the 13 findings have NaN power, meaning effect size too large for the "
              f"power formula to return a finite value -- treat these as power approximately 1.0, "
              f"not as missing.)")


if __name__ == "__main__":
    main()
