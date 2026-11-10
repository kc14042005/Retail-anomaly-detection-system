import argparse
import json
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from anomaly_core import BUSINESS_RULES, init_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="review.db")
    ap.add_argument("--models", default="models")
    ap.add_argument("--lookback-days", type=int, default=28)
    ap.add_argument("--min-reviewed", type=int, default=60)
    a = ap.parse_args()

    con = init_db(a.db)
    th = json.load(open(f"{a.models}/thresholds.json"))
    r = pd.read_sql_query(
        "SELECT a.score, a.weight, a.is_audit, a.run_date, f.verdict FROM alerts a JOIN feedback f USING(alert_id) "
        "WHERE a.run_date >= date((SELECT max(run_date) FROM alerts), ?)", con, params=(f"-{a.lookback_days} day",))
    if len(r) < a.min_reviewed:
        con.execute("INSERT INTO threshold_log VALUES(?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), th["score_threshold"], th["audit_lower"], None, len(r), "kept: too few reviews"))
        con.commit()
        print("kept threshold, reviewed", len(r))
        return
    r["y"] = (r["verdict"] == "true").astype(int)
    all_scores = pd.read_sql_query("SELECT score, is_audit FROM alerts WHERE run_date >= date((SELECT max(run_date) FROM alerts), ?)", con, params=(f"-{a.lookback_days} day",))
    cands = np.unique(np.round(r["score"].values, 4))
    best_t, best_p = None, None
    for t in sorted(cands):
        m = r[r["score"] >= t]
        if m["weight"].sum() == 0 or len(m) < 30:
            continue
        p = float((m["weight"] * m["y"]).sum() / m["weight"].sum())
        vol = float(m["weight"].sum()) / max(r["run_date"].nunique(), 1)
        if p >= BUSINESS_RULES["target_precision"]:
            best_t, best_p = float(t), p
            break
    note = "lowered or kept" if best_t is not None and best_t <= th["score_threshold"] else "raised"
    if best_t is None:
        m = r[r["score"] >= th["score_threshold"]]
        best_p = float((m["weight"] * m["y"]).sum() / max(m["weight"].sum(), 1e-9))
        best_t = float(np.quantile(r["score"], 0.5)) if best_p < BUSINESS_RULES["target_precision"] else th["score_threshold"]
        note = "no threshold reaches target; raised to median reviewed score"
    th["audit_lower"] = float(min(th["audit_lower"], best_t - 0.02))
    th["score_threshold"] = best_t
    json.dump(th, open(f"{a.models}/thresholds.json", "w"), indent=2)
    con.execute("INSERT INTO threshold_log VALUES(?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), best_t, th["audit_lower"], best_p, len(r), note))
    con.commit()
    print("new threshold", round(best_t, 4), "precision", round(best_p, 3), "reviewed", len(r), note)


if __name__ == "__main__":
    main()
