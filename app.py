import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ------------------------------------------------------------------ DATA
raw = pd.read_csv(os.path.join(BASE, "data", "hr.csv"), encoding="utf-8-sig")
raw.drop_duplicates(inplace=True)
raw["Attr"] = (raw["Attrition"] == "Yes").astype(int)

DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "Attrition", "Attr"]
X = pd.get_dummies(raw.drop(columns=DROP), drop_first=True)
y = raw["Attr"]
COLS = list(X.columns)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------ MODELS
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                            min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
comparison, fitted = [], {}
for name, m in candidates.items():
    m.fit(X_tr, y_tr)
    p, pr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    fitted[name] = m
    comparison.append(dict(model=name,
        accuracy=round(accuracy_score(y_te, p) * 100, 1),
        precision=round(precision_score(y_te, p, zero_division=0) * 100, 1),
        recall=round(recall_score(y_te, p) * 100, 1),
        f1=round(f1_score(y_te, p) * 100, 1),
        auc=round(roc_auc_score(y_te, pr) * 100, 1)))

MODEL = fitted["Logistic Regression"]    # best recall + AUC on imbalanced data -> used for prediction
RF = fitted["Random Forest"]             # notebook algorithm -> used for feature importance
best_pred = MODEL.predict(X_te)
tn, fp, fn, tp = confusion_matrix(y_te, best_pred).ravel()
imp = pd.Series(RF.feature_importances_, index=COLS).sort_values(ascending=False).head(12)

# defaults for the predictor form
NUM_DEF = raw.drop(columns=DROP).select_dtypes("number").median().to_dict()
CAT_DEF = {c: raw[c].mode()[0] for c in raw.drop(columns=DROP).select_dtypes("object").columns}

# ------------------------------------------------------------------ HELPERS
def rate_by(col, order=None):
    g = raw.groupby(col)["Attr"].agg(["mean", "count", "sum"])
    if order: g = g.reindex(order)
    return dict(labels=[str(i) for i in g.index], rate=[round(v * 100, 1) for v in g["mean"]],
                count=g["count"].astype(int).tolist(), left=g["sum"].astype(int).tolist())

def binned(col, bins, labels):
    b = pd.cut(raw[col], bins=bins, labels=labels, right=False)
    g = raw.groupby(b, observed=False)["Attr"].agg(["mean", "count"])
    return dict(labels=labels, rate=[round(v * 100, 1) if not np.isnan(v) else 0 for v in g["mean"]],
                count=g["count"].astype(int).tolist())

# ------------------------------------------------------------------ ROUTES
@app.route("/")
def index():
    opts = {c: sorted(raw[c].unique().tolist()) for c in
            ["JobRole", "Department", "MaritalStatus", "BusinessTravel", "OverTime", "Gender", "EducationField"]}
    return render_template("index.html", opts=opts)

@app.route("/api/overview")
def overview():
    left = raw[raw.Attr == 1]
    stay = raw[raw.Attr == 0]
    corr = raw.select_dtypes("number").drop(columns=["EmployeeCount", "StandardHours", "EmployeeNumber"]) \
              .corr()["Attr"].drop("Attr").sort_values()
    corr = pd.concat([corr.head(6), corr.tail(6)])
    return jsonify(
        kpis=dict(total=int(len(raw)), left=int(raw.Attr.sum()), rate=round(raw.Attr.mean() * 100, 1),
                  avg_age=round(raw.Age.mean(), 1), avg_income=int(raw.MonthlyIncome.mean()),
                  avg_tenure=round(raw.YearsAtCompany.mean(), 1),
                  inc_left=int(left.MonthlyIncome.mean()), inc_stay=int(stay.MonthlyIncome.mean())),
        split=dict(stay=int(len(stay)), left=int(len(left))),
        dept=rate_by("Department"), gender=rate_by("Gender"), overtime=rate_by("OverTime"),
        marital=rate_by("MaritalStatus"), travel=rate_by("BusinessTravel"),
        role=rate_by("JobRole"), field=rate_by("EducationField"),
        jobsat=rate_by("JobSatisfaction"), wlb=rate_by("WorkLifeBalance"),
        envsat=rate_by("EnvironmentSatisfaction"), joblevel=rate_by("JobLevel"),
        stock=rate_by("StockOptionLevel"),
        age=binned("Age", [18, 25, 30, 35, 40, 45, 50, 61], ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50+"]),
        tenure=binned("YearsAtCompany", [0, 2, 5, 10, 20, 50], ["0-1", "2-4", "5-9", "10-19", "20+"]),
        income=binned("MonthlyIncome", [0, 3000, 5000, 8000, 12000, 25000], ["<3k", "3-5k", "5-8k", "8-12k", "12k+"]),
        distance=binned("DistanceFromHome", [0, 5, 10, 20, 40], ["1-4", "5-9", "10-19", "20+"]),
        corr=dict(labels=corr.index.tolist(), values=[round(v, 3) for v in corr.values]),
    )

@app.route("/api/model")
def model_info():
    return jsonify(comparison=comparison, selected="Logistic Regression",
                   cm=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
                   importance=dict(labels=[i.replace("_", " ") for i in imp.index],
                                   values=[round(v * 100, 2) for v in imp.values]),
                   train=int(len(X_tr)), test=int(len(X_te)), features=len(COLS))

@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    row = {**NUM_DEF, **CAT_DEF}
    for k, v in d.items():
        if k in NUM_DEF:
            try: row[k] = float(v)
            except (TypeError, ValueError): pass
        elif k in CAT_DEF:
            row[k] = v
    df = pd.DataFrame([row])
    enc = pd.get_dummies(df).reindex(columns=COLS, fill_value=0).astype(float)
    prob = float(MODEL.predict_proba(enc)[0, 1])
    level = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"
    flags = []
    if row["OverTime"] == "Yes": flags.append("Works overtime - the strongest attrition signal in the data")
    if row["JobSatisfaction"] <= 2: flags.append("Low job satisfaction")
    if row["EnvironmentSatisfaction"] <= 2: flags.append("Low environment satisfaction")
    if row["WorkLifeBalance"] <= 2: flags.append("Poor work-life balance")
    if row["YearsAtCompany"] <= 2: flags.append("Very new to the company (0-2 years)")
    if row["MonthlyIncome"] < 3500: flags.append("Monthly income below 3,500")
    if row["DistanceFromHome"] >= 15: flags.append("Long commute (15+ km)")
    if row["StockOptionLevel"] == 0: flags.append("No stock options")
    if row["MaritalStatus"] == "Single": flags.append("Single employees leave more often")
    if row["BusinessTravel"] == "Travel_Frequently": flags.append("Travels frequently")
    if row["YearsSinceLastPromotion"] >= 5: flags.append("Not promoted for 5+ years")
    tips = []
    if row["OverTime"] == "Yes": tips.append("Reduce overtime load or offer compensation time")
    if row["JobSatisfaction"] <= 2 or row["EnvironmentSatisfaction"] <= 2: tips.append("Hold a 1-on-1 to understand engagement concerns")
    if row["MonthlyIncome"] < 3500: tips.append("Review pay against market benchmarks")
    if row["YearsSinceLastPromotion"] >= 5: tips.append("Discuss a career growth / promotion path")
    if row["StockOptionLevel"] == 0: tips.append("Consider long-term incentives such as stock options")
    if not tips: tips.append("Keep up regular check-ins and recognition")
    return jsonify(probability=round(prob * 100, 1), level=level, flags=flags, tips=tips)

@app.route("/api/employees")
def employees():
    q = request.args.get("q", "").lower().strip()
    dept = request.args.get("dept", "")
    st = request.args.get("status", "")
    page = max(int(request.args.get("page", 1)), 1)
    df = raw
    if dept: df = df[df.Department == dept]
    if st: df = df[df.Attrition == st]
    if q:
        m = df.JobRole.str.lower().str.contains(q) | df.EducationField.str.lower().str.contains(q) \
            | df.EmployeeNumber.astype(str).str.contains(q)
        df = df[m]
    total, per = len(df), 12
    part = df.iloc[(page - 1) * per: page * per]
    rows = part[["EmployeeNumber", "Age", "Gender", "Department", "JobRole", "MonthlyIncome",
                 "YearsAtCompany", "OverTime", "JobSatisfaction", "Attrition"]].to_dict("records")
    return jsonify(rows=rows, total=int(total), page=page, pages=max(1, -(-total // per)))

if __name__ == "__main__":
    print("\n  Open  http://127.0.0.1:5000  in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
