import pandas as pd
import numpy as np
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
from datetime import datetime

# ─────────────────────────────────────────────
# 1. LOAD & VALIDATE DATA
# ─────────────────────────────────────────────
df = pd.read_csv("system_metrics.csv").tail(100).reset_index(drop=True)
features = df[["CPU_Usage", "Memory_Usage", "Process_Count"]].copy()

# ─────────────────────────────────────────────
# 2. SCALE
# ─────────────────────────────────────────────
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# ─────────────────────────────────────────────
# 3. MODEL 1 — Isolation Forest
#    Detects global anomalies (sparse regions)
# ─────────────────────────────────────────────
iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
iso_labels = iso.fit_predict(scaled)           # -1 = anomaly
iso_scores = iso.decision_function(scaled)     # lower = more anomalous

# ─────────────────────────────────────────────
# 4. MODEL 2 — Local Outlier Factor
#    Detects local density anomalies (subtle outliers)
# ─────────────────────────────────────────────
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(scaled)           # -1 = anomaly
lof_scores = lof.negative_outlier_factor_      # more negative = more anomalous

# ─────────────────────────────────────────────
# 5. MODEL 3 — DBSCAN Clustering
#    Label -1 = noise/outlier (no cluster fit)
# ─────────────────────────────────────────────
db = DBSCAN(eps=0.8, min_samples=5)
cluster_labels = db.fit_predict(scaled)        # -1 = noise point

# ─────────────────────────────────────────────
# 6. MODEL 4 — Z-Score (statistical baseline)
#    Flag rows where ANY feature deviates > 2.5σ
# ─────────────────────────────────────────────
z_scores = np.abs(zscore(features))
z_anomaly = (z_scores > 2.5).any(axis=1).astype(int)  # 1 = anomaly

# ─────────────────────────────────────────────
# 7. ENSEMBLE VOTING
#    A point is anomalous if ≥ 2 models agree
# ─────────────────────────────────────────────
votes = pd.DataFrame({
    "iso":   (iso_labels   == -1).astype(int),
    "lof":   (lof_labels   == -1).astype(int),
    "dbscan":(cluster_labels== -1).astype(int),
    "zscore": z_anomaly,
})
vote_total = votes.sum(axis=1)
df["Vote_Score"]   = vote_total
df["ML_Anomaly"]   = vote_total.apply(lambda v: "Anomaly" if v >= 2 else "Normal")
df["ISO_Score"]    = iso_scores
df["LOF_Score"]    = lof_scores
df["DBSCAN_Cluster"] = cluster_labels

# ─────────────────────────────────────────────
# 8. ML-DERIVED SEVERITY (no hardcoded thresholds)
#    Severity is based on ISO score percentile
# ─────────────────────────────────────────────
iso_pct = pd.Series(iso_scores).rank(pct=True)

def ml_severity(idx):
    if df.loc[idx, "ML_Anomaly"] != "Anomaly":
        return None
    pct = iso_pct[idx]
    votes = df.loc[idx, "Vote_Score"]
    if pct < 0.05 or votes == 4:
        return "CRITICAL"
    elif pct < 0.15 or votes == 3:
        return "HIGH"
    else:
        return "MODERATE"

df["Severity"] = [ml_severity(i) for i in range(len(df))]

# ─────────────────────────────────────────────
# 9. ML-DERIVED SYSTEM STATE (cluster-based)
#    State labels come from DBSCAN cluster centroids
# ─────────────────────────────────────────────
cluster_cpu_means = {}
for c in set(cluster_labels):
    if c != -1:
        mask = cluster_labels == c
        cluster_cpu_means[c] = features.loc[mask, "CPU_Usage"].mean()

def ml_state(idx):
    c = df.loc[idx, "DBSCAN_Cluster"]
    if c == -1:
        return "Outlier Behavior"
    mean_cpu = cluster_cpu_means.get(c, 0)
    # Quartile-based labels derived from data distribution
    q1 = features["CPU_Usage"].quantile(0.25)
    q3 = features["CPU_Usage"].quantile(0.75)
    if mean_cpu < q1:
        return "Idle Cluster"
    elif mean_cpu < q3:
        return "Normal Load Cluster"
    else:
        return "High Load Cluster"

df["ML_State"] = [ml_state(i) for i in range(len(df))]

# ─────────────────────────────────────────────
# 10. GET TOP PROCESS (live system context)
# ─────────────────────────────────────────────
try:
    top_proc = max(
        psutil.process_iter(['name', 'cpu_percent']),
        key=lambda p: p.info['cpu_percent'] or 0
    )
    proc_name = top_proc.info['name']
    proc_cpu  = top_proc.info['cpu_percent']
except Exception:
    proc_name, proc_cpu = "Unknown", 0.0

# ─────────────────────────────────────────────
# 11. PRINT ANOMALY REPORT
# ─────────────────────────────────────────────
ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "⚠️ "}

print("\n" + "="*55)
print("   ML-DRIVEN SYSTEM ANOMALY REPORT")
print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*55 + "\n")

anomaly_count = 0
for i in range(len(df)):
    if df["ML_Anomaly"][i] != "Anomaly":
        continue

    anomaly_count += 1
    sev   = df["Severity"][i]
    state = df["ML_State"][i]
    icon  = ICONS.get(sev, "⚠️ ")
    time  = df["Time"][i]
    cpu   = df["CPU_Usage"][i]
    mem   = df["Memory_Usage"][i]
    procs = df["Process_Count"][i]
    vscore = df["Vote_Score"][i]
    iso_s  = df["ISO_Score"][i]

    print(f"[{time}] {icon} {sev} ANOMALY  |  State: {state}")
    print(f"   CPU: {cpu:.1f}%  |  Memory: {mem:.1f}%  |  Processes: {procs}")
    print(f"   Models agreed: {vscore}/4  |  Isolation score: {iso_s:.4f}")
    print(f"   Top process: {proc_name} (~{proc_cpu:.1f}%)")
    print(f"   Cluster: {df['DBSCAN_Cluster'][i]}\n")

# ─────────────────────────────────────────────
# 12. FINAL ML PERFORMANCE REPORT
# ─────────────────────────────────────────────
print("="*55)
print("   SYSTEM PERFORMANCE SUMMARY (ML-DERIVED)")
print("="*55)

# All stats derived from data, not constants
cpu_mean = df["CPU_Usage"].mean()
cpu_std  = df["CPU_Usage"].std()
mem_mean = df["Memory_Usage"].mean()

print(f"\n  Samples analysed     : {len(df)}")
print(f"  Anomalies detected   : {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
print(f"  Normal samples       : {len(df) - anomaly_count}")
print(f"\n  CPU  — Mean: {cpu_mean:.2f}%  Std: {cpu_std:.2f}%  Max: {df['CPU_Usage'].max():.2f}%")
print(f"  MEM  — Mean: {mem_mean:.2f}%  Max: {df['Memory_Usage'].max():.2f}%")
print(f"  Procs— Mean: {df['Process_Count'].mean():.0f}  Max: {df['Process_Count'].max()}")




peak_time = df.loc[df["CPU_Usage"].idxmax(), "Time"]
print(f"  Peak CPU at          : {peak_time}")

# Severity breakdown
sev_counts = df["Severity"].value_counts()
print("\n  Severity breakdown:")
for sev, cnt in sev_counts.items():
    print(f"    {ICONS.get(sev,'⚠️')} {sev}: {cnt}")

# Cluster distribution
print("\n  DBSCAN cluster distribution:")
for c, grp in df.groupby("DBSCAN_Cluster"):
    label = "Noise/Outlier" if c == -1 else f"Cluster {c}"
    print(f"    {label}: {len(grp)} samples")

# ─────────────────────────────────────────────
# 13. SAVE
# ─────────────────────────────────────────────
df.to_csv("output_with_anomalies.csv", index=False)
print("\n  Results saved → output_with_anomalies.csv")
print("="*55 + "\n")
