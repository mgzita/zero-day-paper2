"""
Verify the post-hoc power value cited in the Discussion section
("UNSW-NB15 AE H1, power=0.13") and print every power value computed
by strengthen_statistics.py, for a full audit rather than checking
just one number in isolation.

Run from project root:
    python paper2\\pilot\\verify_power_values.py
"""

import json
import os

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")


def safe_load(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    data = safe_load(os.path.join(PILOT_DIR, "strengthened_statistics_full.json"))
    if data is None:
        print("\nCannot verify -- strengthened_statistics_full.json not found. "
              "Run strengthen_statistics.py first.")
        return

    print(f"{'=' * 90}\nSPECIFIC CHECK: UNSW-NB15, AE, H1 -- Discussion claims power=0.13\n{'=' * 90}")
    unsw_h1 = data.get("UNSW-NB15_h1", {})
    ae_record = unsw_h1.get("ae")
    if ae_record:
        actual_power = ae_record.get("power")
        print(f"  Found record: {json.dumps(ae_record, indent=2)}")
        if actual_power is not None:
            match = abs(actual_power - 0.13) < 0.01
            print(f"\n  Discussion claims: power=0.13")
            print(f"  Actual value found: power={actual_power:.4f}")
            print(f"  {'MATCH -- claim is correct' if match else 'MISMATCH -- Discussion text needs correcting'}")
        else:
            print("  WARNING: record found but has no 'power' field.")
    else:
        print("  NOT FOUND: no 'ae' record under 'UNSW-NB15_h1' in the saved file.")
        print(f"  Available keys under UNSW-NB15_h1: {list(unsw_h1.keys())}")

    print(f"\n{'=' * 90}\nFULL AUDIT: every post-hoc power value in the saved statistics\n{'=' * 90}")
    for section_key, section_val in data.items():
        if section_key == "fdr_correction" or not isinstance(section_val, dict):
            continue
        for model_key, record in section_val.items():
            if isinstance(record, dict) and "power" in record:
                power = record.get("power")
                p = record.get("p")
                power_str = f"{power:.4f}" if power is not None else "n/a"
                p_str = f"{p:.5f}" if p is not None else "n/a"
                flag = "  <-- LOW POWER (<0.80)" if (power is not None and power < 0.80) else ""
                print(f"  {section_key:30s} {model_key:10s} p={p_str:>10s}  power={power_str:>8s}{flag}")


if __name__ == "__main__":
    main()
