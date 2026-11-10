import json
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st
from anomaly_core import init_db

DB = "review.db"
st.set_page_config(page_title="Retail Anomaly Review", layout="wide")
con = init_db(DB)
alerts = pd.read_sql_query(
    "SELECT a.*, f.verdict, f.reviewer, f.reviewed_at FROM alerts a LEFT JOIN feedback f USING(alert_id)", con)
st.title("Retail Anomaly Review Dashboard")
if alerts.empty:
    st.info("No alerts yet. Run score_daily.py first.")
    st.stop()

dates = sorted(alerts["run_date"].unique())
sel = st.sidebar.select_slider("Run date range", options=dates, value=(dates[max(len(dates) - 7, 0)], dates[-1]))
reasons = st.sidebar.multiselect("Reason", sorted(alerts["reason_code"].unique()))
stores = st.sidebar.multiselect("Store", sorted(alerts["store_id"].unique()))
show_audit = st.sidebar.checkbox("Include audit sample", value=True)
only_pending = st.sidebar.checkbox("Only pending", value=True)

v = alerts[(alerts["run_date"] >= sel[0]) & (alerts["run_date"] <= sel[1])]
if reasons:
    v = v[v["reason_code"].isin(reasons)]
if stores:
    v = v[v["store_id"].isin(stores)]
if not show_audit:
    v = v[v["is_audit"] == 0]
reviewed = v[v["verdict"].notna() & (v["is_audit"] == 0)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts in range", int((v["is_audit"] == 0).sum()))
c2.metric("Pending review", int((v["verdict"].isna() & (v["is_audit"] == 0)).sum()))
c3.metric("Confirmed true rate", f"{(reviewed['verdict'] == 'true').mean():.0%}" if len(reviewed) else "n/a")
c4.metric("Value at stake (Rs)", f"{v.loc[(v['is_audit'] == 0), 'amount'].abs().sum():,.0f}")

l, r = st.columns(2)
l.subheader("Alerts per day")
l.bar_chart(v[v["is_audit"] == 0].groupby("run_date").size())
r.subheader("Alerts by reason")
r.bar_chart(v[v["is_audit"] == 0].groupby("reason_code").size())

st.subheader("Review queue")
q = v[v["verdict"].isna()] if only_pending else v
q = q.sort_values(["run_date", "score"], ascending=[False, False])
st.dataframe(q[["alert_id", "run_date", "ts", "store_id", "category", "qty", "amount", "payment_method",
                "reason_code", "reason_text", "score", "verdict"]], width="stretch", height=380)

st.subheader("Mark an alert")
pending_ids = q["alert_id"].tolist()
if pending_ids:
    with st.container():
        aid = st.selectbox("Alert", pending_ids)
        row = alerts[alerts["alert_id"] == aid].iloc[0]
        st.write(row["reason_text"])
        who = st.text_input("Reviewer name", "reviewer1")
        b1, b2, _ = st.columns([1, 1, 6])
        for label, key, col in (("Genuine anomaly", "true", b1), ("False alarm", "false", b2)):
            if col.button(label):
                con.execute("INSERT OR REPLACE INTO feedback VALUES(?,?,?,?)",
                            (aid, key, who, datetime.now().isoformat(timespec="seconds")))
                con.commit()
                st.rerun()

st.subheader("Threshold history")
st.dataframe(pd.read_sql_query("SELECT * FROM threshold_log ORDER BY updated_at DESC", con), width="stretch")
try:
    st.caption("Current threshold file: " + json.dumps(json.load(open("models/thresholds.json"))))
except FileNotFoundError:
    pass
