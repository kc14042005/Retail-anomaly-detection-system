import argparse
import json
import joblib
import numpy as np
import pandas as pd
from anomaly_core import build_features, init_db, reason_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--date", required=True)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--db", default="review.db")
    ap.add_argument("--models", default="models")
    ap.add_argument("--history-days", type=int, default=60)
    ap.add_argument("--audit-size", type=int, default=15)
    a = ap.parse_args()

    art = joblib.load(f"{a.models}/detector.joblib")
    th = json.load(open(f"{a.models}/thresholds.json"))
    df = pd.read_csv(a.data, parse_dates=["ts"], dtype={"customer_id": "object", "price_basis": "object", "anomaly_type": "object"})
    con = init_db(a.db)
    for day in pd.date_range(a.date, a.end_date or a.date):
        run_day(df, day, art, th, con, a)
    con.close()


def run_day(df, day, art, th, con, a):
    ds = day.strftime("%Y-%m-%d")
    win = df[(df["ts"] >= day - pd.Timedelta(days=a.history_days)) & (df["ts"] < day + pd.Timedelta(days=1))]
    d = build_features(win, art["state"])
    today = d[d["ts"].dt.normalize() == day].copy()
    if today.empty:
        print("no rows for", ds)
        return
    ens, raw = art["detector"].score(today, art["method"])
    today["score"] = ens
    today = today.sort_values("score", ascending=False)
    cap = int(np.ceil(th["alert_rate_cap"] * len(today)))
    alerts = today[today["score"] >= th["score_threshold"]].head(cap).copy()
    band = today[(today["score"] >= th["audit_lower"]) & (today["score"] < th["score_threshold"])]
    audit = band.sample(min(a.audit_size, len(band)), random_state=int(day.strftime("%Y%m%d"))) if len(band) else band
    alerts["is_audit"] = 0
    alerts["weight"] = 1.0
    audit = audit.copy()
    audit["is_audit"] = 1
    audit["weight"] = len(band) / max(len(audit), 1)
    out = pd.concat([alerts, audit])
    out["rank_in_day"] = out["score"].rank(ascending=False, method="first").astype(int)
    codes, texts = reason_codes(out)
    out["reason_code"], out["reason_text"] = codes, texts
    out["alert_id"] = ds + "-" + out["line_id"].astype(str)
    out["run_date"] = ds
    out["ts"] = out["ts"].astype(str)
    cols = ["alert_id", "run_date", "line_id", "ts", "store_id", "product_id", "category", "customer_key", "qty",
            "unit_price", "amount", "payment_method", "score", "rank_in_day", "reason_code", "reason_text", "is_audit", "weight"]
    con.execute("DELETE FROM alerts WHERE run_date = ? AND alert_id NOT IN (SELECT alert_id FROM feedback)", (ds,))
    out[cols].to_sql("alerts_stage", con, if_exists="replace", index=False)
    con.execute("INSERT OR IGNORE INTO alerts(" + ",".join(cols) + ") SELECT " + ",".join(cols) + " FROM alerts_stage")
    con.execute("DROP TABLE alerts_stage")
    con.commit()
    print(ds, "rows", len(today), "alerts", len(alerts), "audit", len(audit))


if __name__ == "__main__":
    main()
