import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

CATEGORIES = ["Dairy", "Staples", "Snacks", "Beverages", "Personal Care", "Household",
              "Fresh Produce", "Bakery", "Frozen Foods", "Electronics"]
CAT_MAP = {
    "dairy": "Dairy", "staples": "Staples", "snacks": "Snacks", "snacks & munchies": "Snacks",
    "munchies & namkeen": "Snacks", "beverages": "Beverages", "drinks": "Beverages",
    "cold drinks & juices": "Beverages", "personal care": "Personal Care", "household": "Household",
    "fresh produce": "Fresh Produce", "bakery": "Bakery", "frozen foods": "Frozen Foods",
    "electronics": "Electronics",
}
GST = {"Dairy": 0.05, "Staples": 0.05, "Snacks": 0.12, "Beverages": 0.18, "Personal Care": 0.18,
       "Household": 0.18, "Fresh Produce": 0.05, "Bakery": 0.05, "Frozen Foods": 0.12, "Electronics": 0.18}
PAYMENTS = ["UPI", "Card", "Cash", "Wallet", "Gift Card"]
STORES = ["Connaught Place", "Saket", "Dwarka", "Rohini", "Lajpat Nagar", "Karol Bagh", "Janakpuri",
          "Vasant Kunj", "Pitampura", "Nehru Place", "Greater Kailash", "Mayur Vihar", "Rajouri Garden",
          "Hauz Khas", "Preet Vihar", "Noida Sector 18", "Gurgaon Sector 29", "Malviya Nagar", "Okhla",
          "Shahdara", "Punjabi Bagh", "Model Town", "Kalkaji", "Paschim Vihar", "Chandni Chowk"]
BUSINESS_RULES = {
    "false_alarm_cost_inr": 50,
    "missed_anomaly_cost_inr": 2000,
    "manual_review_rate": 0.02,
    "alert_rate_cap": 0.01,
    "target_precision": 0.60,
}
CONT = ["price_dev", "qty_dev_cust", "qty_peer_z", "amt_peer_z", "disc_dev", "rar_cat_pay",
        "rar_hour_amt", "rar_guest_amt", "log_amount"]
BIN = ["is_guest", "is_refund", "dup_flag", "guest_refund"]
BIN_SCALE = np.array([1.0, 3.0, 5.0, 5.0])
FEATURES = CONT + BIN
Z_FEATURES = ["price_dev", "qty_dev_cust", "qty_peer_z", "amt_peer_z", "disc_dev"]
START = pd.Timestamp("2026-01-01")
DAYS = 304
MODELS = ["iforest", "autoencoder", "robust_z"]
ANOMALY_TYPES = ["price_error", "qty_spike", "discount_abuse", "odd_hour_high_value",
                 "guest_refund", "rare_combo", "duplicate_charge"]


def generate_data(n=1_000_000, seed=7):
    rng = np.random.default_rng(seed)
    n_prod = 1500
    cat_idx = rng.integers(0, len(CATEGORIES), n_prod)
    cat_base = np.array([60, 120, 40, 55, 180, 150, 45, 50, 160, 600], dtype=float)
    base = cat_base[cat_idx] * np.exp(rng.normal(0, 0.5, n_prod))
    launch = np.where(rng.random(n_prod) < 0.05, rng.integers(30, DAYS - 20, n_prod), 0)
    rev_day = np.where(rng.random(n_prod) < 0.05, rng.integers(60, DAYS - 30, n_prod), 10 ** 6)
    pop = rng.pareto(1.5, n_prod) + 1
    pop = pop / pop.sum()
    store_basis = rng.choice(["incl", "excl", "mixed"], len(STORES), p=[0.4, 0.4, 0.2])
    n_cust = 60000
    cust_ids = np.array([f"C{i:06d}" for i in range(n_cust)])
    cust_w = rng.gamma(1.2, 1, n_cust)
    cust_w = cust_w / cust_w.sum()
    wholesale = rng.random(n_cust) < 0.01
    cust_qty_factor = np.where(wholesale, rng.uniform(5, 10, n_cust), np.exp(rng.normal(0, 0.15, n_cust)))

    m = int(n / 1.0025)
    day = rng.choice(DAYS, m, p=_day_weights())
    pid = rng.choice(n_prod, m, p=pop)
    day = np.maximum(day, launch[pid] + rng.integers(0, 5, m))
    day = np.minimum(day, DAYS - 1)
    cat = cat_idx[pid]
    store = rng.integers(0, len(STORES), m)
    hour = np.clip(np.round(rng.choice([11, 13, 18, 19], m) + rng.normal(0, 2.5, m)), 8, 21).astype(int)
    minute = rng.integers(0, 60, m)
    second = rng.integers(0, 60, m)
    guest = rng.random(m) < 0.30
    cidx = rng.choice(n_cust, m, p=cust_w)
    festival = (day >= 287) & (day <= 297)
    qty = 1 + rng.poisson(0.8, m)
    qty = np.where(cat == 9, 1 + (rng.random(m) < 0.1), qty).astype(float)
    qty = np.where(~guest, np.round(qty * cust_qty_factor[cidx]), qty)
    fest_bulk = festival & (rng.random(m) < 0.15)
    qty = np.where(fest_bulk, qty * rng.integers(3, 8, m), qty)
    qty = np.maximum(qty, 1)
    refund_legit = (~guest) & (rng.random(m) < 0.004)
    qty = np.where(refund_legit, -1, qty)
    disc = rng.choice([0, 5, 10, 15, 20], m, p=[0.55, 0.15, 0.15, 0.10, 0.05]).astype(float)
    disc = np.where(festival & (rng.random(m) < 0.3), np.minimum(disc + 10, 30), disc)
    pay_p = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    pay = rng.choice(len(PAYMENTS), m, p=pay_p)
    perishable = np.isin(cat, [0, 6, 7])
    pay = np.where(perishable & (pay == 4), rng.choice(4, m, p=pay_p[:4] / pay_p[:4].sum()), pay)
    price_ex = base[pid] * np.where(day >= rev_day[pid], 1.10, 1.0) * np.exp(rng.normal(0, 0.03, m))
    gst = np.array([GST[CATEGORIES[c]] for c in range(len(CATEGORIES))])[cat]
    basis_true = np.where(store_basis[store] == "mixed", rng.choice(["incl", "excl"], m), store_basis[store])
    unit_price = np.where(basis_true == "incl", price_ex * (1 + gst), price_ex)
    basis_label = np.where(rng.random(m) < 0.15, None, basis_true)

    cat_names = np.array(CATEGORIES, dtype=object)[cat]
    raw = cat_names.copy()
    snk = cat_names == "Snacks"
    bev = cat_names == "Beverages"
    r1 = day >= 90
    r2 = day >= 212
    raw[snk & r1 & ~r2] = "Snacks & Munchies"
    raw[snk & r2] = "Munchies & Namkeen"
    raw[bev & r1 & ~r2] = "Drinks"
    raw[bev & r2] = "Cold Drinks & Juices"
    noisy = rng.random(m) < 0.02
    raw = np.where(noisy, np.char.upper(raw.astype(str)), raw)
    raw = np.where(rng.random(m) < 0.01, np.char.add(raw.astype(str), " "), raw)

    ts = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 3600 + minute * 60 + second, unit="s")
    df = pd.DataFrame({
        "ts": ts, "store_id": np.array(STORES, dtype=object)[store], "product_id": pid,
        "category_raw": raw, "customer_id": np.where(guest, None, cust_ids[cidx]),
        "qty": qty, "unit_price": unit_price, "price_basis": basis_label, "discount_pct": disc,
        "payment_method": np.array(PAYMENTS, dtype=object)[pay],
    })
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""

    k = int(0.0105 * m)
    pick = rng.choice(m, k, replace=False)
    types = rng.choice(["price_error", "qty_spike", "discount_abuse", "odd_hour_high_value",
                        "guest_refund", "rare_combo"], k, p=[0.26, 0.17, 0.17, 0.14, 0.14, 0.12])
    for t in np.unique(types):
        ix = pick[types == t]
        if t == "price_error":
            df.loc[ix, "unit_price"] = df.loc[ix, "unit_price"].values * rng.choice([10, 100, 0.1, 0.01], len(ix))
        elif t == "qty_spike":
            df.loc[ix, "qty"] = np.abs(df.loc[ix, "qty"].values) * rng.integers(15, 60, len(ix)) + 10
        elif t == "discount_abuse":
            df.loc[ix, "discount_pct"] = rng.choice([60, 70, 80, 90], len(ix))
        elif t == "odd_hour_high_value":
            h = rng.integers(0, 5, len(ix))
            df.loc[ix, "ts"] = df.loc[ix, "ts"].dt.normalize().values + pd.to_timedelta(h * 3600 + rng.integers(0, 3600, len(ix)), unit="s")
            df.loc[ix, "payment_method"] = rng.choice(["Cash", "Wallet", "Gift Card"], len(ix))
            df.loc[ix, "qty"] = np.abs(df.loc[ix, "qty"].values) * rng.integers(6, 12, len(ix))
        elif t == "guest_refund":
            df.loc[ix, "customer_id"] = None
            df.loc[ix, "qty"] = -rng.integers(1, 9, len(ix)).astype(float)
            df.loc[ix, "payment_method"] = "Cash"
        elif t == "rare_combo":
            per = df.loc[ix, "category_raw"].str.lower().str.strip().map(CAT_MAP).isin(["Dairy", "Fresh Produce", "Bakery"])
            df.loc[ix, "payment_method"] = "Gift Card"
            df.loc[ix[~per.values], "category_raw"] = "Fresh Produce"
        df.loc[ix, "is_anomaly"] = 1
        df.loc[ix, "anomaly_type"] = t

    nd = int(0.0025 * m)
    src = rng.choice(m, nd, replace=False)
    dup = df.iloc[src].copy()
    dup["ts"] = dup["ts"].dt.floor("min") + pd.to_timedelta(rng.integers(31, 60, nd), unit="s")
    df.loc[df.index[src], "ts"] = df.loc[df.index[src], "ts"].dt.floor("min") + pd.to_timedelta(rng.integers(0, 30, nd), unit="s")
    dup["is_anomaly"] = 1
    dup["anomaly_type"] = "duplicate_charge"
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "line_id", np.arange(len(df)))
    df["unit_price"] = df["unit_price"].round(2)
    return df


def _day_weights():
    d = np.arange(DAYS)
    w = np.ones(DAYS)
    w[(START + pd.to_timedelta(d, unit="D")).dayofweek >= 5] = 1.35
    w[287:298] *= 1.5
    return w / w.sum()


def fit_state(df):
    d = df.copy()
    d["category"] = d["category_raw"].str.lower().str.strip().map(CAT_MAP)
    d["gst"] = d["category"].map(GST)
    known = d["price_basis"].notna()
    ex = np.where(d["price_basis"] == "incl", d["unit_price"] / (1 + d["gst"]), d["unit_price"])
    d["price_ex"] = ex
    kn = d[known]
    product_ref = kn.groupby("product_id")["price_ex"].median()
    cat_ref = kn.groupby("category")["price_ex"].median()
    pos = d[d["qty"] > 0]
    peer = {}
    lq = np.log1p(pos["qty"])
    for c, g in lq.groupby(pos["category"]):
        med = g.median()
        mad = max(1.4826 * (g - med).abs().median(), 0.25)
        peer[c] = {"qty_med": float(med), "qty_scale": float(mad), "qty_raw_med": float(pos.loc[g.index, "qty"].median())}
    amt = np.abs(d["qty"]) * d["price_ex"] * (1 - d["discount_pct"] / 100)
    la = np.log1p(amt)
    for c, g in la.groupby(d["category"]):
        med = g.median()
        mad = max(1.4826 * (g - med).abs().median(), 0.25)
        peer[c].update({"amt_med": float(med), "amt_scale": float(mad)})
    disc_ref = d.groupby("product_id")["discount_pct"].median()
    amt_edges = np.quantile(amt, np.linspace(0, 1, 11)[1:-1])
    band = np.searchsorted(amt_edges, amt)
    guest = d["customer_id"].isna().astype(int)
    N = len(d)

    t_cp = _key(d["category"], d["payment_method"]).value_counts()
    t_ha = _key(d["ts"].dt.hour, band).value_counts()
    t_ga = _key(guest, np.sign(d["qty"]).astype(int), band).value_counts()
    return {
        "product_ref": product_ref.to_dict(), "cat_ref": cat_ref.to_dict(), "peer": peer,
        "disc_ref": disc_ref.to_dict(), "amt_edges": amt_edges.tolist(), "N": N,
        "t_cp": {k: int(v) for k, v in t_cp.items()},
        "t_ha": {k: int(v) for k, v in t_ha.items()},
        "t_ga": {k: int(v) for k, v in t_ga.items()},
    }


def _key(*cols):
    parts = [pd.Series(np.asarray(c)).astype(str) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _rar(keys, table, N):
    cnt = pd.Series(keys.values).map(table).fillna(0).values
    return -np.log((cnt + 1) / (N + len(table)))


def build_features(df, state):
    d = df.copy()
    d["category"] = d["category_raw"].str.lower().str.strip().map(CAT_MAP)
    d["gst"] = d["category"].map(GST)
    d["is_guest"] = d["customer_id"].isna().astype(np.int8)
    d["customer_key"] = d["customer_id"].fillna("GUEST")
    ref = d["product_id"].map(state["product_ref"])
    ref = ref.fillna(d["category"].map(state["cat_ref"]))
    p = d["unit_price"].values
    cand_in = p / (1 + d["gst"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        use_ex = np.abs(np.log(p / ref.values)) <= np.abs(np.log(cand_in / ref.values))
    ex = np.where(d["price_basis"] == "incl", cand_in, np.where(d["price_basis"] == "excl", p, np.where(use_ex, p, cand_in)))
    d["price_ex"] = np.maximum(ex, 0.01)
    d["amount"] = d["qty"] * d["unit_price"] * (1 - d["discount_pct"] / 100)
    d["amount_ex"] = d["qty"].abs() * d["price_ex"] * (1 - d["discount_pct"] / 100)

    d = d.sort_values(["ts", "line_id"], kind="stable")
    lp = np.log(d["price_ex"])
    g = lp.groupby(d["product_id"])
    med = g.transform(lambda s: s.shift(1).rolling(50, min_periods=8).median())
    q1 = g.transform(lambda s: s.shift(1).rolling(50, min_periods=8).quantile(0.25))
    q3 = g.transform(lambda s: s.shift(1).rolling(50, min_periods=8).quantile(0.75))
    scale = np.maximum((q3 - q1) / 1.349, 0.04)
    has_base = med.notna()
    base_med = np.where(has_base, med, np.log(ref.reindex(d.index).values))
    base_scale = np.where(has_base, scale, 0.15)
    d["price_ratio"] = np.exp(lp.values - base_med)
    d["price_dev"] = np.clip((lp.values - base_med) / base_scale, -30, 30)
    d["price_dev"] = d["price_dev"].fillna(0)

    x = np.log1p(d["qty"].clip(lower=0))
    reg = d["is_guest"] == 0
    dev = pd.Series(0.0, index=d.index)
    xr = x[reg]
    gr = xr.groupby(d.loc[reg, "customer_key"])
    cn = gr.cumcount()
    cs = gr.cumsum() - xr
    cs2 = (xr ** 2).groupby(d.loc[reg, "customer_key"]).cumsum() - xr ** 2
    mean = cs / cn.replace(0, np.nan)
    var = (cs2 / cn.replace(0, np.nan)) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0))
    z = (xr - mean) / np.maximum(sd, 0.35)
    dev.loc[reg] = np.where(cn >= 3, z, 0.0)
    d["qty_dev_cust"] = np.clip(dev.fillna(0), -15, 15)

    pm = d["category"].map(lambda c: state["peer"][c]["qty_med"])
    ps = d["category"].map(lambda c: state["peer"][c]["qty_scale"])
    d["qty_peer_z"] = np.clip((np.log1p(d["qty"].abs()) - pm) / ps, -15, 15)
    d["qty_ratio"] = d["qty"].abs() / d["category"].map(lambda c: state["peer"][c]["qty_raw_med"])
    am = d["category"].map(lambda c: state["peer"][c]["amt_med"])
    asc = d["category"].map(lambda c: state["peer"][c]["amt_scale"])
    d["amt_peer_z"] = np.clip((np.log1p(d["amount_ex"]) - am) / asc, -15, 15)
    dref = d["product_id"].map(state["disc_ref"]).fillna(0)
    d["disc_ref"] = dref
    d["disc_dev"] = (d["discount_pct"] - dref) / 10.0

    band = np.searchsorted(np.array(state["amt_edges"]), d["amount_ex"].values)
    d["hour"] = d["ts"].dt.hour
    N = state["N"]
    d["rar_cat_pay"] = _rar(_key(d["category"], d["payment_method"]), state["t_cp"], N)
    d["rar_hour_amt"] = _rar(_key(d["hour"], band), state["t_ha"], N)
    sg = np.sign(d["qty"]).astype(int)
    d["rar_guest_amt"] = _rar(_key(d["is_guest"], sg, band), state["t_ga"], N)
    d["log_amount"] = np.log1p(d["amount_ex"])
    d["is_refund"] = (d["qty"] < 0).astype(np.int8)
    d["guest_refund"] = (d["is_refund"] * d["is_guest"]).astype(np.int8)
    key = d["store_id"].astype(str) + "|" + d["customer_key"] + "|" + d["product_id"].astype(str) + "|" + d["qty"].astype(str) + "|" + d["unit_price"].astype(str) + "|" + d["ts"].dt.floor("min").astype(str)
    d["dup_flag"] = (d.groupby(key).cumcount() > 0).astype(np.int8)
    return d


class Detector:
    def __init__(self, seed=0):
        self.seed = seed
        self.scaler = RobustScaler()

    def prep(self, d, fit=False):
        X = d[CONT].values.astype(float)
        X = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        X = np.clip(X, -12, 12)
        B = d[BIN].values.astype(float) * BIN_SCALE
        return np.hstack([X, B])

    def fit(self, d, n_fit=150_000, if_params=None, ae_epochs=40):
        s = d.sample(min(n_fit, len(d)), random_state=self.seed)
        X = self.prep(s, fit=True)
        params = dict(n_estimators=200, max_samples=512, contamination=0.01, random_state=self.seed, n_jobs=-1)
        params.update(if_params or {})
        self.iforest = IsolationForest(**params).fit(X)
        self.ae = MLPRegressor(hidden_layer_sizes=(12, 5, 12), activation="tanh", max_iter=ae_epochs,
                               batch_size=512, learning_rate_init=0.003, random_state=self.seed).fit(X, X)
        self.ref = {}
        sc = self.raw_scores(d.sample(min(200_000, len(d)), random_state=self.seed + 1))
        for k, v in sc.items():
            self.ref[k] = np.sort(v)
        return self

    def raw_scores(self, d):
        X = self.prep(d)
        out = {}
        out["iforest"] = -self.iforest.score_samples(X)
        out["autoencoder"] = ((self.ae.predict(X) - X) ** 2).mean(axis=1)
        out["robust_z"] = np.abs(d[Z_FEATURES].values).max(axis=1)
        return out

    def pct(self, name, x):
        r = self.ref[name]
        p = np.searchsorted(r, x, side="right") / len(r)
        hi = r[-1]
        spread = max(hi - r[int(0.99 * len(r))], 1e-9)
        tail = 1 + 0.05 * np.log1p(np.maximum(x - hi, 0) / spread)
        return np.where(x > hi, tail, p)

    def score(self, d, method="max"):
        raw = self.raw_scores(d)
        pcts = np.vstack([self.pct(k, raw[k]) for k in MODELS])
        return combine(pcts, method), raw


def combine(pcts, method):
    if method == "mean":
        return pcts.mean(axis=0)
    if method == "top2":
        return np.sort(pcts, axis=0)[1:].mean(axis=0)
    return pcts.max(axis=0) + 0.01 * pcts.mean(axis=0)


def tune_combiner(det, d, lab_idx, lab_y, lab_w, q=0.01):
    raw = det.raw_scores(d)
    pcts = np.vstack([det.pct(k, raw[k]) for k in MODELS])
    rows = []
    for m in ("mean", "top2", "max"):
        sc = combine(pcts, m)
        ps = [weighted_precision(sc, lab_idx, lab_y, lab_w, qq) for qq in (0.005, 0.0075, 0.01, 0.015, 0.02)]
        rows.append((m, weighted_precision(sc, lab_idx, lab_y, lab_w, q), float(np.nanmean(ps))))
    t = pd.DataFrame(rows, columns=["method", "weighted_p_at_1pct", "mean_p_over_0.5_to_2pct"])
    return t.sort_values("mean_p_over_0.5_to_2pct", ascending=False).iloc[0]["method"], t


def lof_scores(train_d, eval_d, det, n_neighbors=35, n_fit=20_000, seed=0):
    Xt = det.prep(train_d.sample(min(n_fit, len(train_d)), random_state=seed))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1).fit(Xt)
    return -lof.score_samples(det.prep(eval_d))


def precision_at_top(score, y, q):
    k = max(int(round(len(score) * q)), 1)
    idx = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y[idx]))


def recall_at_top(score, y, q):
    k = max(int(round(len(score) * q)), 1)
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def make_labelled_sample(base_score, y, n=300, seed=0):
    rng = np.random.default_rng(seed)
    N = len(base_score)
    order = np.argsort(-base_score, kind="stable")
    bands = [order[: int(0.01 * N)], order[int(0.01 * N): int(0.05 * N)], order[int(0.05 * N):]]
    per = n // 3
    idx, w = [], []
    for b in bands:
        s = rng.choice(b, per, replace=False)
        idx.append(s)
        w.append(np.full(per, len(b) / per))
    idx = np.concatenate(idx)
    return idx, y[idx], np.concatenate(w)


def weighted_precision(score_all, lab_idx, lab_y, lab_w, q):
    thr = np.quantile(score_all, 1 - q)
    m = score_all[lab_idx] >= thr
    if m.sum() == 0:
        return float("nan")
    return float((lab_w[m] * lab_y[m]).sum() / lab_w[m].sum())


def tune_alert_rate(score_all, lab_idx, lab_y, lab_w, grid=(0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)):
    N = len(score_all)
    pop_pos = float((lab_w * lab_y).sum()) * N / float(lab_w.sum())
    rows = []
    for q in grid:
        p = weighted_precision(score_all, lab_idx, lab_y, lab_w, q)
        tp = p * q * N if p == p else 0
        fp = q * N - tp
        cost = fp * BUSINESS_RULES["false_alarm_cost_inr"] + (pop_pos - tp) * BUSINESS_RULES["missed_anomaly_cost_inr"]
        rows.append({"alert_rate": q, "precision_est": p, "est_true_alerts": tp, "est_recall": tp / max(pop_pos, 1), "expected_cost_inr": cost})
    t = pd.DataFrame(rows)
    ok = t[(t["alert_rate"] <= BUSINESS_RULES["alert_rate_cap"]) & (t["precision_est"] >= BUSINESS_RULES["target_precision"])]
    best = ok.sort_values("expected_cost_inr").iloc[0]["alert_rate"] if len(ok) else t.sort_values("precision_est", ascending=False).iloc[0]["alert_rate"]
    return float(best), t


REASON_TEXT = {
    "PRICE_SPIKE": "Price is {pricephrase} the product's recent typical price. Likely a price-entry or GST mix-up error.",
    "REFUND_REVIEW": "Refund of Rs {amt:,.0f} on a registered customer account. Refunds are rare, confirm the original sale.",
    "QTY_SPIKE": "Quantity of {qty:.0f} is {qratio:.1f}x the usual for {category}. Check for a keying error or bulk fraud.",
    "DISCOUNT_HIGH": "Discount of {disc:.0f}% is far above the usual {dref:.0f}% for this product. Check authorisation.",
    "ODD_HOUR_VALUE": "{pay} payment of Rs {amt:,.0f} at {hour:02d}:00, outside normal trading hours.",
    "GUEST_REFUND": "Refund of Rs {amt:,.0f} with no customer ID attached. Check till and approver.",
    "DUPLICATE": "Identical purchase recorded more than once in the same minute. Possible double charge.",
    "RARE_COMBO": "Unusual combination: {category} paid by {pay} (rarely seen together).",
    "UNUSUAL_PATTERN": "Overall pattern is unusual compared with similar orders, with no single dominant cause.",
}


def reason_codes(d):
    out_code, out_text = [], []
    cols = ["price_dev", "price_ratio", "qty_peer_z", "qty_ratio", "disc_dev", "disc_ref", "discount_pct", "hour",
            "dup_flag", "is_refund", "is_guest", "amount", "qty", "category", "payment_method", "rar_cat_pay", "rar_hour_amt", "amount_ex"]
    for r in d[cols].itertuples(index=False):
        rr = r._asdict()
        cand = {
            "PRICE_SPIKE": abs(rr["price_dev"]) / 4 if abs(rr["price_dev"]) >= 4 else 0,
            "QTY_SPIKE": rr["qty_peer_z"] / 4 if rr["qty_peer_z"] >= 4 else 0,
            "DISCOUNT_HIGH": rr["disc_dev"] / 4 if rr["disc_dev"] >= 3.5 else 0,
            "ODD_HOUR_VALUE": 1.2 if (rr["hour"] < 6 and rr["rar_hour_amt"] > 9) else 0,
            "GUEST_REFUND": 1.1 if (rr["is_refund"] == 1 and rr["is_guest"] == 1) else 0,
            "REFUND_REVIEW": 0.9 if (rr["is_refund"] == 1 and rr["is_guest"] == 0) else 0,
            "DUPLICATE": 1.3 if rr["dup_flag"] == 1 else 0,
            "RARE_COMBO": 1.0 if rr["rar_cat_pay"] > 9 else 0,
        }
        code = max(cand, key=cand.get)
        if cand[code] <= 0:
            code = "UNUSUAL_PATTERN"
        pr = rr["price_ratio"]
        phrase = f"{pr:.1f}x" if pr >= 1 else f"{1 / max(pr, 1e-6):.0f}x lower than"
        txt = REASON_TEXT[code].format(
            pricephrase=phrase, qty=abs(rr["qty"]), qratio=rr["qty_ratio"], category=rr["category"],
            disc=rr["discount_pct"], dref=rr["disc_ref"], pay=rr["payment_method"], amt=abs(rr["amount"]), hour=int(rr["hour"]))
        out_code.append(code)
        out_text.append(txt)
    return out_code, out_text


def init_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS alerts(
        alert_id TEXT PRIMARY KEY, run_date TEXT, line_id INTEGER, ts TEXT, store_id TEXT, product_id INTEGER,
        category TEXT, customer_key TEXT, qty REAL, unit_price REAL, amount REAL, payment_method TEXT,
        score REAL, rank_in_day INTEGER, reason_code TEXT, reason_text TEXT, is_audit INTEGER, weight REAL,
        status TEXT DEFAULT 'pending');
    CREATE TABLE IF NOT EXISTS feedback(
        alert_id TEXT PRIMARY KEY, verdict TEXT, reviewer TEXT, reviewed_at TEXT);
    CREATE TABLE IF NOT EXISTS threshold_log(
        updated_at TEXT, score_threshold REAL, audit_lower REAL, precision_est REAL, n_reviewed INTEGER, note TEXT);
    """)
    con.commit()
    return con
