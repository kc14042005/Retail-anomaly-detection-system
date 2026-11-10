import argparse
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from anomaly_core import (BUSINESS_RULES, Detector, build_features, fit_state, generate_data, make_labelled_sample,
                          precision_at_top, recall_at_top, tune_alert_rate, tune_combiner, weighted_precision)

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    os.makedirs(a.models, exist_ok=True)
    if a.regenerate or not os.path.exists(a.data):
        df = generate_data(a.rows)
        df.to_csv(a.data, index=False)
    df = pd.read_csv(a.data, parse_dates=["ts"], dtype={"customer_id": "object", "price_basis": "object", "anomaly_type": "object"})
    df["anomaly_type"] = df["anomaly_type"].fillna("")
    t = time.time()
    state = fit_state(df)
    d = build_features(df, state).sort_values("line_id").reset_index(drop=True)
    print("features", round(time.time() - t, 1), "s", len(d), "rows")
    y = d["is_anomaly"].values
    det = Detector().fit(d)
    raw = det.raw_scores(d)
    base = raw["robust_z"]
    lab_idx, lab_y, lab_w = make_labelled_sample(base, y, 300)
    rng = np.random.default_rng(0)
    perm = rng.permutation(300)
    tune_i, hold_i = perm[:150], perm[150:]
    method, grid = tune_combiner(det, d, lab_idx[tune_i], lab_y[tune_i], lab_w[tune_i])
    print(grid.round(3).to_string(index=False))
    ens, raw = det.score(d, method)
    q, table = tune_alert_rate(ens, lab_idx[tune_i], lab_y[tune_i], lab_w[tune_i])
    thr = float(np.quantile(ens, 1 - q))
    audit_lower = float(np.quantile(ens, 1 - 3 * q))
    p_true = precision_at_top(ens, y, 0.01)
    p_hold = weighted_precision(ens, lab_idx[hold_i], lab_y[hold_i], lab_w[hold_i], 0.01)
    print("combiner", method, "alert_rate", q, "threshold", round(thr, 4))
    print("P@1% truth", round(p_true, 3), "P@1% held-out labels", round(p_hold, 3), "recall@1%", round(recall_at_top(ens, y, 0.01), 3))
    joblib.dump({"detector": det, "state": state, "method": method}, f"{a.models}/detector.joblib")
    json.dump({"score_threshold": thr, "audit_lower": audit_lower, "alert_rate": q,
               "alert_rate_cap": BUSINESS_RULES["alert_rate_cap"], "target_precision": BUSINESS_RULES["target_precision"]},
              open(f"{a.models}/thresholds.json", "w"), indent=2)
    pd.DataFrame({"line_id": d.loc[lab_idx, "line_id"].values, "label": lab_y, "weight": lab_w}).to_csv(f"{a.models}/labelled_sample.csv", index=False)


if __name__ == "__main__":
    main()
