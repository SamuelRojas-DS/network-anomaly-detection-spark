#!/usr/bin/env python3
# streaming_demo_v2.py  — actualizado con cadena de transformación real
#
# Cadena completa:
#   batch_df
#     → StringIndexer      (S3: preprocessors/indexer)    label → label_index
#     → VectorAssembler    (reconstruido)                  78 cols → features
#     → StandardScaler     (S3: preprocessors/scaler)     features → scaled_features
#     → RF PipelineModel   (S3: model_optimized)           scaled_features → prediction
#     → KMeans PipelineModel (S3: model_optimized)         scaled_features → cluster_id
#     → UDF distancia      (calculado)                    cluster_id → distanceToCentroid
#
# Uso:
#   spark-submit --master yarn \
#     --conf spark.executor.cores=2 \
#     --conf spark.executor.memory=6g \
#     --conf spark.executor.instances=2 \
#     streaming_demo_v2.py --producer-host <EC2-PRIVATE-IP>

import json, time, os, argparse
import numpy as np
from collections import defaultdict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField,
                                DoubleType, StringType)
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel, VectorAssembler
from pyspark.sql.streaming import StreamingQueryListener

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--producer-host", default="localhost")
parser.add_argument("--producer-port", default=9999, type=int)
parser.add_argument("--trigger",       default=10,   type=int)
args = parser.parse_args()

# ── Rutas S3 ──────────────────────────────────────────────────────────────────
S3            = "s3://proyecto-anomalias-emr"
PATH_INDEXER  = f"{S3}/models/random_forest_pipeline/preprocessors/indexer"
PATH_SCALER   = f"{S3}/models/random_forest_pipeline/preprocessors/scaler"
PATH_RF       = f"{S3}/models/random_forest_pipeline/model_optimized"
PATH_KMEANS   = f"{S3}/models/kmeans_pipeline/model_optimized"

LOCAL_SUMMARY = "/tmp/batch_summary.json"
LOCAL_SLA     = "/tmp/sla_metrics.jsonl"

# ── Umbral KMeans (calibrado sobre IDS2017) ───────────────────────────────────
UMBRAL_DISTANCIA = 15.0

# ── 78 features — exactamente los del entrenamiento (features_meta.json) ─────
FEATURE_NAMES = [
    "ack_flag_count", "act_data_pkt_fwd", "active_max", "active_mean",
    "active_min", "active_std", "average_packet_size", "avg_bwd_segment_size",
    "avg_fwd_segment_size", "bwd_avg_bulk_rate", "bwd_avg_bytesbulk",
    "bwd_avg_packetsbulk", "bwd_header_length", "bwd_iat_max", "bwd_iat_mean",
    "bwd_iat_min", "bwd_iat_std", "bwd_iat_total", "bwd_packet_length_max",
    "bwd_packet_length_mean", "bwd_packet_length_min", "bwd_packet_length_std",
    "bwd_packetss", "bwd_psh_flags", "bwd_urg_flags", "cwe_flag_count",
    "destination_port", "downup_ratio", "ece_flag_count", "fin_flag_count",
    "flow_bytess", "flow_duration", "flow_iat_max", "flow_iat_mean",
    "flow_iat_min", "flow_iat_std", "flow_packetss", "fwd_avg_bulk_rate",
    "fwd_avg_bytesbulk", "fwd_avg_packetsbulk", "fwd_header_length34",
    "fwd_header_length55", "fwd_iat_max", "fwd_iat_mean", "fwd_iat_min",
    "fwd_iat_std", "fwd_iat_total", "fwd_packet_length_max",
    "fwd_packet_length_mean", "fwd_packet_length_min", "fwd_packet_length_std",
    "fwd_packetss", "fwd_psh_flags", "fwd_urg_flags", "idle_max", "idle_mean",
    "idle_min", "idle_std", "init_win_bytes_backward", "init_win_bytes_forward",
    "max_packet_length", "min_packet_length", "min_seg_size_forward",
    "packet_length_mean", "packet_length_std", "packet_length_variance",
    "psh_flag_count", "rst_flag_count", "subflow_bwd_bytes",
    "subflow_bwd_packets", "subflow_fwd_bytes", "subflow_fwd_packets",
    "syn_flag_count", "total_backward_packets", "total_fwd_packets",
    "total_length_of_bwd_packets", "total_length_of_fwd_packets",
    "urg_flag_count",
]  # 78 columnas

# ── Schema TCP: 78 features (Double) + label (String) ─────────────────────────
SCHEMA = StructType(
    [StructField(f, DoubleType(), nullable=True) for f in FEATURE_NAMES]
    + [StructField("label", StringType(), nullable=True)]
)

# from_csv requiere DDL string, no StructType
SCHEMA_STR = ", ".join(
    f"{f.name} {f.dataType.simpleString()}"
    for f in SCHEMA.fields
)

# ── ANSI colors ───────────────────────────────────────────────────────────────
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"
MAG = "\033[95m"; BLU = "\033[94m"

def bar(value, total, width=28, color=GRN):
    if total == 0: return " " * width
    filled = int(round(value / total * width))
    return color + "█" * filled + RST + "░" * (width - filled)

# ── SLA Listener ──────────────────────────────────────────────────────────────
class SLAListener(StreamingQueryListener):
    def onQueryStarted(self, e):    pass
    def onQueryTerminated(self, e): pass
    def onQueryProgress(self, e):
        p     = e.progress
        dur   = p.durationMs
        L     = dur.get("triggerExecution", 0)
        inp   = p.inputRowsPerSecond
        prc   = p.processedRowsPerSecond
        ratio = prc / inp if inp > 0 else 0.0
        status = ("VIOLATION" if L > 60_000 or ratio < 1.0 else
                  "CRITICAL"  if L > 35_000 or ratio < 1.1 else
                  "WARNING"   if L > 22_000 or ratio < 1.2 else "OK")
        rec = {
            "batchId":         p.batchId,
            "ts":              p.timestamp,
            "L_E2E_ms":        L,
            "inputRps":        round(inp, 1),
            "procRps":         round(prc, 1),
            "ratio":           round(ratio, 3),
            "numRows":         p.numInputRows,
            "status":          status,
            "t_read_ms":       dur.get("getBatch", 0),
            "t_add_batch_ms":  dur.get("addBatch", 0),
            "t_commit_ms":     dur.get("commitOffsets", 0),
        }
        with open(LOCAL_SLA, "a") as f:
            f.write(json.dumps(rec) + "\n")

# ── SparkSession ──────────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("AnomalyDetection-Streaming-v2")
         .config("spark.sql.streaming.metricsEnabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ── Cargar transformadores y modelos ──────────────────────────────────────────
print(f"\n{CYN}{BLD}[demo] Cargando transformadores y modelos...{RST}")

indexer  = StringIndexerModel.load(PATH_INDEXER)
scaler   = StandardScalerModel.load(PATH_SCALER)
rf_model = PipelineModel.load(PATH_RF)
km_model = PipelineModel.load(PATH_KMEANS)

# VectorAssembler — reconstruido con las 78 features del entrenamiento
assembler = VectorAssembler(
    inputCols  = FEATURE_NAMES,
    outputCol  = "features",
    handleInvalid = "keep",
)

# Centroides KMeans para calcular distancia euclidiana (no la exporta Spark)
km_centers = km_model.stages[-1].clusterCenters()
centers_bc = spark.sparkContext.broadcast(km_centers)

print(f"{GRN}[demo] Modelos cargados: RF ({rf_model.stages[-1].numTrees} árboles) | "
      f"KMeans ({len(km_centers)} centroides){RST}")
print(f"{CYN}[demo] Productor → {args.producer_host}:{args.producer_port} | "
      f"Trigger {args.trigger}s{RST}\n")

spark.streams.addListener(SLAListener())

# ── UDF: distancia euclidiana al centroide asignado ───────────────────────────
from pyspark.sql.types import DoubleType as DT
from pyspark.ml.linalg import Vectors

@F.udf(DT())
def dist_udf(features, cluster_id):
    if features is None or cluster_id is None:
        return None
    center = centers_bc.value[int(cluster_id)]
    arr    = features.toArray()
    return float(np.sqrt(np.sum((arr - center) ** 2)))

# ── readStream — socket TCP ───────────────────────────────────────────────────
raw = (spark.readStream
            .format("socket")
            .option("host", args.producer_host)
            .option("port", args.producer_port)
            .load())

parsed = (raw
          .select(F.from_csv(F.col("value"), SCHEMA_STR).alias("d"))
          .select("d.*"))

# ── foreachBatch ──────────────────────────────────────────────────────────────
_history = []

def procesar_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    t0 = time.time()

    # ── Cache ─────────────────────────────────────────────────────────────────
    batch_df.cache()

    # ── Cadena de preprocesamiento compartida por RF y KMeans ─────────────────
    # 1. StringIndexer: label → label_index
    df_indexed = indexer.transform(batch_df)

    # 2. VectorAssembler: 78 features → features
    df_assembled = assembler.transform(df_indexed)

    # 3. StandardScaler: features → scaled_features
    df_scaled = scaler.transform(df_assembled).cache()

    # ── Inferencia RF ──────────────────────────────────────────────────────────
    # RF espera: scaled_features, label_index
    pred_rf = rf_model.transform(df_scaled).cache()

    # ── Inferencia KMeans ──────────────────────────────────────────────────────
    # KMeans espera: scaled_features → cluster_id
    pred_km = km_model.transform(df_scaled)
    pred_km = pred_km.withColumn(
        "distanceToCentroid",
        dist_udf(F.col("scaled_features"), F.col("cluster_id"))
    ).withColumn(
        "es_anomalia",
        F.when(F.col("distanceToCentroid") > UMBRAL_DISTANCIA, 1).otherwise(0)
    ).cache()

    # ── Métricas RF ───────────────────────────────────────────────────────────
    n      = batch_df.count()
    rf_raw = pred_rf.groupBy("prediction", "label").count().collect()

    rf_counts = defaultdict(int)
    label_map = {}
    for row in rf_raw:
        k = int(row["prediction"])
        rf_counts[k] += row["count"]
        label_map[k]  = row["label"] or f"class_{k}"

    rf_anomaly = n - rf_counts.get(0, 0)

    # ── Métricas KMeans ───────────────────────────────────────────────────────
    km_stats = pred_km.agg(
        F.mean("distanceToCentroid").alias("dist_mean"),
        F.max("distanceToCentroid").alias("dist_max"),
        F.sum("es_anomalia").alias("anomalias"),
    ).collect()[0]

    km_anom      = int(km_stats["anomalias"]  or 0)
    km_dist_mean = float(km_stats["dist_mean"] or 0)
    km_dist_max  = float(km_stats["dist_max"]  or 0)

    # ── Persistir en S3 ───────────────────────────────────────────────────────
    (pred_rf
     .select("prediction", "label", "label_index", "flow_duration", "flow_bytess")
     .write.mode("append")
     .parquet(f"{S3}/results/streaming_metrics/rf/"))

    (pred_km
     .select("cluster_id", "distanceToCentroid", "es_anomalia", "label")
     .write.mode("append")
     .parquet(f"{S3}/results/streaming_metrics/kmeans/"))

    # ── Unpersist ─────────────────────────────────────────────────────────────
    pred_km.unpersist()
    pred_rf.unpersist()
    df_scaled.unpersist()
    batch_df.unpersist()

    t_ms = int((time.time() - t0) * 1000)

    # ── Resumen para monitor.py ───────────────────────────────────────────────
    summary = {
        "batchId":      batch_id,
        "ts":           time.strftime("%H:%M:%S"),
        "n":            n,
        "proc_ms":      t_ms,
        "rf_benign":    rf_counts.get(0, 0),
        "rf_anomaly":   rf_anomaly,
        "rf_dist":      dict(rf_counts),
        "label_map":    {str(k): v for k, v in label_map.items()},
        "km_anom":      km_anom,
        "km_normal":    n - km_anom,
        "km_dist_mean": round(km_dist_mean, 2),
        "km_dist_max":  round(km_dist_max, 2),
    }
    with open(LOCAL_SUMMARY, "w") as f:
        json.dump(summary, f)

    _history.append(summary)
    if len(_history) > 10:
        _history.pop(0)

    # ── Consola enriquecida ───────────────────────────────────────────────────
    rf_rate  = rf_anomaly / n * 100 if n else 0
    km_rate  = km_anom    / n * 100 if n else 0
    km_color = RED if km_dist_mean > UMBRAL_DISTANCIA else GRN
    rf_color = GRN if rf_rate < 10 else YLW if rf_rate < 30 else RED
    prev_rf  = (_history[-2]["rf_anomaly"] / _history[-2]["n"] * 100
                if len(_history) >= 2 and _history[-2]["n"] else 0)
    trend    = "↑" if rf_rate > prev_rf else ("↓" if rf_rate < prev_rf else "→")

    W = 70
    print(f"\n{BLD}{'═'*W}{RST}")
    print(f"{BLD}{CYN}  BATCH #{batch_id:>4}   {summary['ts']}   "
          f"filas: {n:>6,}   proc: {t_ms:>5} ms{RST}")
    print(f"{'─'*W}")

    print(f"\n  {BLD}{BLU}RANDOM FOREST — Clasificación supervisada{RST}")
    for idx in sorted(rf_counts.keys()):
        cnt   = rf_counts[idx]
        lbl   = label_map.get(idx, f"class_{idx}")[:22]
        pct   = cnt / n * 100 if n else 0
        color = GRN if idx == 0 else RED
        print(f"    {lbl:<23} {bar(cnt,n,24,color)}  {cnt:>6,}  ({pct:5.1f}%)")
    print(f"\n    Tasa anomalía RF : {rf_color}{BLD}{rf_rate:5.1f}% {trend}{RST}")

    print(f"\n  {BLD}{MAG}KMEANS — Detección no supervisada  "
          f"(umbral: {UMBRAL_DISTANCIA} | k=30){RST}")
    print(f"    {'Normal':<23} {bar(n-km_anom,n,24,GRN)}  "
          f"{n-km_anom:>6,}  ({100-km_rate:5.1f}%)")
    print(f"    {'Anomalía':<23} {bar(km_anom,n,24,RED)}  "
          f"{km_anom:>6,}  ({km_rate:5.1f}%)")
    print(f"\n    Dist. media → centroide : {km_color}{BLD}{km_dist_mean:7.2f}{RST}")
    print(f"    Dist. máxima en batch   : {km_color}{BLD}{km_dist_max:7.2f}{RST}")

    if len(_history) >= 2:
        print(f"\n  {BLD}TENDENCIA últimos {min(len(_history),5)} batches{RST}")
        print(f"    {'Batch':<8} {'Filas':>7} {'RF%':>7} {'KM%':>7} {'ms':>6}")
        print(f"    {'─'*38}")
        for h in _history[-5:]:
            rp = h["rf_anomaly"] / h["n"] * 100 if h["n"] else 0
            kp = h["km_anom"]    / h["n"] * 100 if h["n"] else 0
            rc = RED if rp>30 else YLW if rp>10 else GRN
            kc = RED if kp>30 else YLW if kp>10 else GRN
            print(f"    #{h['batchId']:<6}  {h['n']:>7,}  "
                  f"{rc}{rp:6.1f}%{RST}  {kc}{kp:6.1f}%{RST}  {h['proc_ms']:>5}")

    print(f"{BLD}{'═'*W}{RST}\n")


# ── Iniciar streaming ─────────────────────────────────────────────────────────
query = (parsed.writeStream
               .foreachBatch(procesar_batch)
               .option("checkpointLocation", f"{S3}/checkpoints/streaming/")
               .trigger(processingTime=f"{args.trigger} seconds")
               .start())

print(f"{GRN}{BLD}[demo] Query activo — ID: {query.id}{RST}")
print(f"{CYN}[demo] Spark UI  : http://localhost:4040{RST}")
print(f"{CYN}[demo] Dashboard : python3 monitor.py  (Terminal 3){RST}\n")

query.awaitTermination()
print(f"{YLW}[demo] Stream finalizado.{RST}")
