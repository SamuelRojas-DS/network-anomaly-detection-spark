#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_kmeans.py
==================================================================
PROPÓSITO:
    Ejecuta en secuencia los cuatro bloques del pipeline no supervisado
    de detección de anomalías basado en KMeans sobre el dataset
    CIC-IDS2017:

    BLOQUE 1 — ENTRENAMIENTO (03_train_kmeans.py):
        Entrena KMeans SOLO sobre tráfico benigno (lunes). El modelo
        aprende la estructura del tráfico normal; flujos lejanos al
        centroide asignado son candidatos a anomalía.

    BLOQUE 2 — CALIBRACIÓN DEL UMBRAL (03b_calibrate_kmeans.py):
        Halla el umbral de distancia euclidiana que maximiza F1 sobre
        los datos de entrenamiento etiquetados (miércoles/viernes),
        sin tocar el conjunto de prueba (martes/jueves).

    BLOQUE 3 — EVALUACIÓN BATCH (04b_evaluate_kmeans.py):
        Evalúa model_v1 sobre el conjunto de prueba con el umbral
        calibrado. Reporta métricas binarias, distribución de distancias
        por clase, análisis de clases NO vistas y métricas de sistema.

    BLOQUE 4 — OPTIMIZACIÓN (03c_optimize_kmeans.py):
        Búsqueda manual de hiperparámetros (KMeans no es compatible con
        CrossValidator de MLlib). Selecciona por Silhouette, re-entrena
        sobre el benigno completo, re-calibra el umbral y genera un
        reporte comparativo frente al modelo base.

INPUTS (rutas S3 de entrada):
    - s3://proyecto-anomalias-emr/data/cic-ids2017/train/processed/
    - s3://proyecto-anomalias-emr/data/cic-ids2017/test/processed/
    - s3://proyecto-anomalias-emr/models/random_forest_pipeline/preprocessors/indexer/

OUTPUTS (rutas S3 de salida):
    - .../models/kmeans_pipeline/model_v1/
    - .../models/kmeans_pipeline/model_optimized/
    - .../models/kmeans_pipeline/threshold/threshold_v1.json
    - .../models/kmeans_pipeline/threshold/threshold_opt.json
    - .../results/kmeans_evaluation/train_metrics_kmeans.json
    - .../results/kmeans_evaluation/test_metrics_kmeans.json
    - .../results/kmeans_evaluation/predictions_kmeans/       (Parquet)
    - .../results/kmeans_evaluation/optimization_report_kmeans.json

DEPENDENCIAS (ejecutar ANTES):
    - gestion_datos.py
    - pipeline_random_forest.py  (solo necesario si se carga el StringIndexer)

ORDEN DE EJECUCIÓN DEL PROYECTO:
    1. gestion_datos.py
    2. pipeline_random_forest.py
    3. pipeline_kmeans.py  ← este script
==================================================================
"""

# === IMPORTS ===
import time                                          # medición de tiempos
import json                                          # serialización de métricas
import logging                                       # logging estructurado
from urllib.parse import urlparse                    # parseo de rutas s3://

import boto3                                         # I/O de JSON con S3
import numpy as np                                   # grilla de umbrales

from pyspark.sql import SparkSession                 # punto de entrada de Spark
from pyspark.sql.functions import (                  # funciones SQL
    udf, col, when,
    sum as Fsum, mean as Fmean, stddev as Fstddev, count as Fcount,
)
from pyspark.sql.types import DoubleType            # tipo de la UDF de distancia
from pyspark.ml import Pipeline, PipelineModel       # pipeline de MLlib
from pyspark.ml.clustering import KMeans             # algoritmo de clustering
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel
from pyspark.ml.linalg import Vectors, VectorUDT     # vector para AUC
from pyspark.ml.evaluation import (
    ClusteringEvaluator,
    BinaryClassificationEvaluator,
)


# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PipelineKMeans")


# ==============================================================
# === CONFIGURACIÓN GLOBAL ===
# ==============================================================
BUCKET           = "s3://proyecto-anomalias-emr"             # bucket raíz
TRAIN_RAW        = f"{BUCKET}/data/cic-ids2017/train/"                   # entrenamiento
TEST_RAW         = f"{BUCKET}/data/cic-ids2017/test/"                    # prueba
RF_MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"           # preprocesadores compartidos (RF)
KM_MODELS_PATH   = f"{BUCKET}/models/kmeans_pipeline/"                  # modelos KMeans
RESULTS_PATH     = f"{BUCKET}/results/kmeans_evaluation/"               # resultados

# Rutas derivadas.
TRAIN_PROCESSED  = f"{TRAIN_RAW}processed/"                             # Parquet de entrenamiento
TEST_PROCESSED   = f"{TEST_RAW}processed/"                              # Parquet de prueba
SCALER_PATH      = f"{RF_MODELS_PATH}preprocessors/scaler/"            # StandardScaler ajustado en gestion_datos.py
INDEXER_PATH     = f"{RF_MODELS_PATH}preprocessors/indexer/"           # StringIndexer ajustado
MODEL_V1_OUT     = f"{KM_MODELS_PATH}model_v1/"                        # modelo base
MODEL_OPT_OUT    = f"{KM_MODELS_PATH}model_optimized/"                 # modelo optimizado
THRESHOLD_V1     = f"{KM_MODELS_PATH}threshold/threshold_v1.json"      # umbral base calibrado
THRESHOLD_OPT    = f"{KM_MODELS_PATH}threshold/threshold_opt.json"     # umbral optimizado
TRAIN_METRICS    = f"{RESULTS_PATH}train_metrics_kmeans.json"          # métricas de entrenamiento
TEST_METRICS_OUT = f"{RESULTS_PATH}test_metrics_kmeans.json"           # métricas de prueba
PREDICTIONS_OUT  = f"{RESULTS_PATH}predictions_kmeans/"                # predicciones Parquet
OPT_REPORT_OUT   = f"{RESULTS_PATH}optimization_report_kmeans.json"    # reporte comparativo

# --- Hiperparámetros Bloque 1 (modelo base) ---
K_CLUSTERS       = 10                                                   # nº de clústeres base
INIT_MODE        = "k-means||"                                          # inicialización paralela estable
MAX_ITER         = 50                                                   # iteraciones máximas
TOL              = 1e-4                                                 # tolerancia de convergencia
SEED             = 42                                                   # semilla reproducible

# --- Parámetros Bloque 4 (optimización) ---
SAMPLE_FRACTION    = 0.30                                               # 30% de muestra para búsqueda

# --- Parámetros de clúster ---
EXECUTOR_INSTANCES = 2                                                  # nº de executors (clúster base)
EXECUTOR_CORES     = 2                                                  # cores por executor
SHUFFLE_PARTITIONS = 16                                                 # particiones de shuffle


# ==============================================================
# === HELPERS COMPARTIDOS ===
# ==============================================================
def _parse_s3(ruta_s3):
    """Descompone s3://bucket/clave en (bucket, clave)."""
    p = urlparse(ruta_s3)
    return p.netloc, p.path.lstrip("/")


def escribir_json_s3(objeto, ruta_s3):
    """Escribe un dict como JSON en S3 con boto3."""
    bucket, clave = _parse_s3(ruta_s3)
    cuerpo = json.dumps(objeto, indent=2, ensure_ascii=False).encode("utf-8")
    boto3.client("s3").put_object(Bucket=bucket, Key=clave, Body=cuerpo)


def leer_json_s3(ruta_s3):
    """Lee un JSON desde S3 con boto3. Devuelve None si falla."""
    try:
        bucket, clave = _parse_s3(ruta_s3)
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=clave)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("No se pudo leer %s (%s). Se continúa sin métricas base.", ruta_s3, e)
        return None


def construir_spark():
    """SparkSession con TODOS los parámetros del clúster vía .config()."""
    spark = (
        SparkSession.builder
        .appName("ProyectoAnomalias-KMeans")                                      # nombre exacto requerido
        .config("spark.executor.cores", "2")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "1024")
        .config("spark.driver.cores", "2")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.instances", str(EXECUTOR_INSTANCES))
        .config("spark.default.parallelism", "16")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "256m")
        .config("spark.kryo.registrationRequired", "false")                       # evita warnings con clases MLlib
        .config("spark.memory.fraction", "0.75")
        .config("spark.memory.storageFraction", "0.40")
        .config("spark.task.maxFailures", "4")
        .config("spark.speculation", "true")
        .config("spark.speculation.multiplier", "1.5")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ==============================================================
# === HELPERS DE DISTANCIA Y CALIBRACIÓN (compartidos entre bloques) ===
# ==============================================================
def construir_udf_distancia(spark, kmeans_model):
    """Construye y devuelve una UDF de distancia euclidiana al centroide
    usando los centros del modelo dado (difundidos por broadcast)."""
    centros_lista = [c.tolist() for c in kmeans_model.clusterCenters()]
    centros_broadcast = spark.sparkContext.broadcast(centros_lista)

    @udf(DoubleType())
    def _distancia_udf(features, cluster_id):
        import numpy as np                                              # import dentro de la UDF
        centroide = np.array(centros_broadcast.value[cluster_id])
        vector    = np.array(features.toArray())
        diff      = vector - centroide
        return float(np.sqrt(np.dot(diff, diff)))

    return _distancia_udf


@udf(VectorUDT())
def distancia_a_vector_udf(distancia):
    """Convierte la distancia escalar en vector [-d, d] para el
    BinaryClassificationEvaluator (AUC-ROC)."""
    return Vectors.dense([-distancia, distancia])


def confusion_en_umbral(df, umbral):
    """Calcula TP, FP, FN, TN para un umbral dado en UNA sola pasada.
    - prediccion = 1 si distancia_centroide > umbral (anómalo)
    - label_binaria = 1 si la fila es un ataque real
    """
    fila = df.agg(
        Fsum(when((col("distancia_centroide") > umbral) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("distancia_centroide") > umbral) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("distancia_centroide") <= umbral) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("distancia_centroide") <= umbral) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    return int(fila["tp"]), int(fila["fp"]), int(fila["fn"]), int(fila["tn"])


def metricas_desde_confusion(tp, fp, fn, tn):
    """Deriva Precision, Recall, F1 y FPR a partir de la matriz de confusión."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


def _calibrar_umbral_interno(spark, model, df_train_completo):
    """Busca el umbral que maximiza F1 (empate -> menor FPR) sobre el
    DataFrame completo de entrenamiento. Función compartida entre
    los bloques 2 y 4."""
    kmeans_model = model.stages[-1]
    udf_dist = construir_udf_distancia(spark, kmeans_model)

    df = model.transform(df_train_completo)                            # cluster_id
    df = df.withColumn("distancia_centroide",
                       udf_dist(col("scaled_features"), col("cluster_id")))
    df = df.withColumn("label_binaria",
                       when(col("label_index") == 0.0, 0).otherwise(1))
    df = df.cache()
    df.count()                                                         # materializar

    # Percentiles sobre benigno para construir la grilla de umbrales.
    df_benigno = df.filter(col("label_binaria") == 0)
    p90, p95, p97, p99, p999 = df_benigno.approxQuantile(
        "distancia_centroide", [0.90, 0.95, 0.97, 0.99, 0.999], 0.001)

    candidatos = sorted(set([p90, p95, p97, p99, p999] + list(np.linspace(p90, p999, 20))))
    logger.info("Percentiles benignos -> P90=%.4f P95=%.4f P97=%.4f P99=%.4f P99.9=%.4f",
                p90, p95, p97, p99, p999)
    logger.info("Total de umbrales candidatos a evaluar: %d", len(candidatos))

    mejor = None
    for umbral in candidatos:
        tp, fp, fn, tn = confusion_en_umbral(df, float(umbral))
        precision, recall, f1, fpr = metricas_desde_confusion(tp, fp, fn, tn)
        cand = {"umbral": float(umbral), "precision": precision,
                "recall": recall, "f1": f1, "fpr": fpr}
        if (mejor is None or cand["f1"] > mejor["f1"]
                or (cand["f1"] == mejor["f1"] and cand["fpr"] < mejor["fpr"])):
            mejor = cand

    # AUC-ROC global (independiente del umbral).
    df = df.withColumn("raw_prediction_vec", distancia_a_vector_udf(col("distancia_centroide")))
    auc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df))

    df.unpersist()
    mejor["auc_roc"] = auc
    mejor["percentiles"] = {"p90": p90, "p95": p95, "p97": p97, "p99": p99, "p999": p999}
    return mejor


def _evaluar_en_test_interno(spark, model, df_test, umbral):
    """Calcula métricas binarias del modelo sobre el set de prueba con
    el umbral dado. Función compartida entre los bloques 3 y 4."""
    kmeans_model = model.stages[-1]
    udf_dist = construir_udf_distancia(spark, kmeans_model)

    df = model.transform(df_test)
    df = df.withColumn("distancia_centroide",
                       udf_dist(col("scaled_features"), col("cluster_id")))
    df = df.withColumn("prediccion_binaria",
                       (col("distancia_centroide") > umbral).cast("int"))
    df = df.withColumn("label_binaria",
                       when(col("label_index") == 0.0, 0).otherwise(1))
    df = df.cache()
    df.count()

    fila = df.agg(
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    tp, fp, fn, tn = int(fila["tp"]), int(fila["fp"]), int(fila["fn"]), int(fila["tn"])
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision, recall, f1, fpr = metricas_desde_confusion(tp, fp, fn, tn)

    df = df.withColumn("raw_prediction_vec", distancia_a_vector_udf(col("distancia_centroide")))
    auc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df))

    df.unpersist()
    return {"auc_roc": auc, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "fpr": fpr}


# ==============================================================
# === BLOQUE 1: ENTRENAMIENTO DEL MODELO BASE ===
# (origen: 03_train_kmeans.py)
# ==============================================================
def bloque1_entrenar_kmeans(spark):
    """Entrena KMeans SOLO sobre tráfico benigno (label_index == 0.0).
    Guarda el modelo base y sus métricas intrínsecas en S3."""
    logger.info("=" * 60)
    logger.info("BLOQUE 1 — ENTRENAMIENTO KMeans (model_v1, solo benigno)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # (c) CARGA DEL DATAFRAME DE ENTRENAMIENTO PROCESADO
    # ------------------------------------------------------------------
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Parquet de entrenamiento cargado desde %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        raise

    # ------------------------------------------------------------------
    # (d) FILTRADO CRÍTICO: KMeans aprende SOLO sobre tráfico benigno
    # ------------------------------------------------------------------
    # CORRECTO: incluir SOLO label_index == 0.0 (benigno, lunes).
    # INCORRECTO: incluir ataques contaminaría el modelo no supervisado.
    df_train_benigno = df_train.filter(col("label_index") == 0.0)       # solo clase mayoritaria benigna
    num_benigno = df_train_benigno.count()                              # conteo de registros benignos
    logger.info("Registros benignos para entrenamiento KMeans: %d", num_benigno)

    # ------------------------------------------------------------------
    # (e) VERIFICAR / GENERAR LA COLUMNA scaled_features
    # ------------------------------------------------------------------
    if "scaled_features" in df_train_benigno.columns:                   # camino normal: ya viene escalado
        logger.info("Columna 'scaled_features' presente: se reutiliza el escalado de gestion_datos.py.")
    else:
        # Camino alternativo: cargar el StandardScaler de RF y aplicarlo.
        logger.info("Columna 'scaled_features' ausente: se cargará el StandardScaler de RF para generarla.")
        try:
            scaler_model = StandardScalerModel.load(SCALER_PATH)
            df_train_benigno = scaler_model.transform(df_train_benigno)
            logger.info("StandardScaler aplicado desde %s", SCALER_PATH)
        except Exception as e:
            logger.error("ERROR al cargar/aplicar el StandardScaler (%s): %s", SCALER_PATH, e)
            raise

    # Persistir el subconjunto benigno: se reutiliza en fit + Silhouette + UDF.
    df_train_benigno = df_train_benigno.cache()
    df_train_benigno.count()                                            # materializar la caché

    # ------------------------------------------------------------------
    # (f)(g) PIPELINE CON KMEANS COMO ÚNICA ETAPA DE MODELADO
    # ------------------------------------------------------------------
    # El preprocesamiento (indexado/escalado) ya se realizó en gestion_datos.py,
    # por lo que el Pipeline contiene SOLO el estimador de clustering.
    kmeans = KMeans(
        k=K_CLUSTERS,                       # 10 clústeres iniciales
        initMode=INIT_MODE,                 # "k-means||": inicialización paralela y estable
        maxIter=MAX_ITER,                   # tope de iteraciones de Lloyd
        tol=TOL,                            # tolerancia de convergencia
        seed=SEED,                          # reproducibilidad
        featuresCol="scaled_features",      # vector de entrada ya escalado
        predictionCol="cluster_id",         # columna de clúster asignado
        distanceMeasure="euclidean",        # métrica de distancia
    )
    pipeline = Pipeline(stages=[kmeans])

    # ------------------------------------------------------------------
    # (h) ENTRENAMIENTO DISTRIBUIDO CON MEDICIÓN DE TIEMPO
    # ------------------------------------------------------------------
    logger.info("Iniciando entrenamiento KMeans (k=%d, maxIter=%d) SOLO sobre benigno...",
                K_CLUSTERS, MAX_ITER)
    t_inicio = time.time()
    model = pipeline.fit(df_train_benigno)
    t_fin = time.time()
    training_time = t_fin - t_inicio
    logger.info("Entrenamiento completado en %.2f s", training_time)

    kmeans_model = model.stages[-1]                                     # KMeansModel ajustado

    # ------------------------------------------------------------------
    # (i) GUARDADO DEL MODELO
    # ------------------------------------------------------------------
    try:
        model.write().overwrite().save(MODEL_V1_OUT)
        logger.info("Modelo KMeans guardado en %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR al guardar el modelo en %s: %s", MODEL_V1_OUT, e)
        raise

    # ------------------------------------------------------------------
    # (j) MÉTRICAS INTRÍNSECAS DE CLUSTERING
    # ------------------------------------------------------------------
    wssse = float(kmeans_model.summary.trainingCost)                    # coste de entrenamiento (WSSSE)
    logger.info("WSSSE (trainingCost): %.4f", wssse)

    cluster_sizes = [int(s) for s in kmeans_model.summary.clusterSizes]
    logger.info("Tamaños de clúster: %s", cluster_sizes)

    centros = kmeans_model.clusterCenters()
    logger.info("Nº de centroides: %d (dimensión = %d)", len(centros), len(centros[0]))

    # Transformar el set benigno para Silhouette y UDF de distancia.
    df_train_pred = model.transform(df_train_benigno)                   # añade cluster_id

    evaluador_silhouette = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="cluster_id",
        metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = float(evaluador_silhouette.evaluate(df_train_pred))
    logger.info("Silhouette score (squaredEuclidean): %.4f", silhouette)

    # ------------------------------------------------------------------
    # (k) UDF DE DISTANCIA EUCLIDIANA AL CENTROIDE ASIGNADO
    # ------------------------------------------------------------------
    # Se recuperan los centros ANTES de construir la UDF y se difunden con
    # broadcast para no serializar el array completo en cada tarea.
    udf_dist = construir_udf_distancia(spark, kmeans_model)
    df_train_pred = df_train_pred.withColumn(
        "distancia_centroide",
        udf_dist(col("scaled_features"), col("cluster_id")),
    )
    dist_min, dist_max = df_train_pred.selectExpr(
        "min(distancia_centroide)", "max(distancia_centroide)"
    ).first()
    logger.info("Distancia al centroide (benigno) -> min=%.4f, max=%.4f", dist_min, dist_max)

    # ------------------------------------------------------------------
    # (l) GUARDADO DE MÉTRICAS DE ENTRENAMIENTO
    # ------------------------------------------------------------------
    metricas = {
        "training_time_seconds": round(training_time, 4),
        "k": K_CLUSTERS,
        "init_mode": INIT_MODE,
        "max_iter": MAX_ITER,
        "wssse": wssse,
        "silhouette_score": silhouette,
        "cluster_sizes": cluster_sizes,
        "num_training_records": int(num_benigno),
        "num_partitions": SHUFFLE_PARTITIONS,
        "executor_instances": EXECUTOR_INSTANCES,
        "cluster_cores_total": EXECUTOR_INSTANCES * EXECUTOR_CORES,
    }
    try:
        escribir_json_s3(metricas, TRAIN_METRICS)
        logger.info("Métricas de entrenamiento guardadas en %s", TRAIN_METRICS)
    except Exception as e:
        logger.error("ERROR al escribir métricas en %s: %s", TRAIN_METRICS, e)
        raise

    df_train_benigno.unpersist()
    logger.info("BLOQUE 1 completado correctamente.")


# ==============================================================
# === BLOQUE 2: CALIBRACIÓN DEL UMBRAL ===
# (origen: 03b_calibrate_kmeans.py)
# ==============================================================
def bloque2_calibrar_kmeans(spark):
    """Calibra el umbral de distancia que maximiza F1 sobre los datos
    de entrenamiento etiquetados (miércoles/viernes de ataque).
    Los datos de prueba (martes/jueves) NO se tocan aquí."""
    logger.info("=" * 60)
    logger.info("BLOQUE 2 — CALIBRACIÓN DEL UMBRAL (model_v1, train etiquetado)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # (c) CARGA DEL MODELO Y DEL DATAFRAME COMPLETO DE ENTRENAMIENTO
    # ------------------------------------------------------------------
    # La calibración usa benigno + miércoles/viernes de ataque para hallar
    # el umbral. Los días de PRUEBA (martes y jueves) NO se tocan aquí:
    # así se preserva la validez metodológica (sin fuga del test).
    try:
        model = PipelineModel.load(MODEL_V1_OUT)
        logger.info("Modelo KMeans cargado desde %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR al cargar el modelo (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)          # Parquet completo (benigno + ataque)
        logger.info("Parquet de entrenamiento cargado desde %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        raise

    # ------------------------------------------------------------------
    # (e)-(j) CALIBRACIÓN MEDIANTE EL HELPER INTERNO
    # ------------------------------------------------------------------
    logger.info("Calibrando umbral con grilla de umbrales (>= 20 candidatos)...")
    mejor = _calibrar_umbral_interno(spark, model, df_train)
    umbral_opt = mejor["umbral"]

    logger.info("RESUMEN CALIBRACIÓN -> umbral=%.4f | F1=%.4f | P=%.4f | R=%.4f | FPR=%.4f | AUC=%.4f",
                umbral_opt, mejor["f1"], mejor["precision"], mejor["recall"],
                mejor["fpr"], mejor["auc_roc"])

    # ------------------------------------------------------------------
    # (k) GUARDADO DEL UMBRAL Y MÉTRICAS
    # ------------------------------------------------------------------
    p = mejor["percentiles"]
    salida = {
        "threshold_optimal": umbral_opt,
        "f1_at_threshold":   mejor["f1"],
        "precision":         mejor["precision"],
        "recall":            mejor["recall"],
        "fpr":               mejor["fpr"],
        "auc_roc":           mejor["auc_roc"],
        "percentil_90":  p["p90"],
        "percentil_95":  p["p95"],
        "percentil_97":  p["p97"],
        "percentil_99":  p["p99"],
        "percentil_999": p["p999"],
        "calibration_records": None,                          # calculado internamente
        "calibration_days": ["Wednesday", "Friday"],
    }
    try:
        escribir_json_s3(salida, THRESHOLD_V1)
        logger.info("Umbral calibrado guardado en %s", THRESHOLD_V1)
    except Exception as e:
        logger.error("ERROR al escribir el umbral en %s: %s", THRESHOLD_V1, e)
        raise

    logger.info("BLOQUE 2 completado correctamente.")


# ==============================================================
# === BLOQUE 3: EVALUACIÓN BATCH DEL MODELO BASE ===
# (origen: 04b_evaluate_kmeans.py)
# ==============================================================
def bloque3_evaluar_kmeans(spark):
    """Evalúa model_v1 sobre el conjunto de prueba (martes + jueves)
    con el umbral calibrado. Reporta métricas binarias, distribución de
    distancias por clase y análisis de clases NO vistas."""
    logger.info("=" * 60)
    logger.info("BLOQUE 3 — EVALUACIÓN BATCH KMeans (model_v1 sobre prueba)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # (b) CARGA DE MODELO, DATAFRAME DE PRUEBA, UMBRAL E INDEXER
    # ------------------------------------------------------------------
    try:
        model = PipelineModel.load(MODEL_V1_OUT)
        logger.info("Modelo KMeans cargado desde %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR al cargar el modelo (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Parquet de prueba cargado desde %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de prueba (%s): %s", TEST_PROCESSED, e)
        raise

    try:
        umbral_json = leer_json_s3(THRESHOLD_V1)
        threshold_optimal = float(umbral_json["threshold_optimal"])
        logger.info("Umbral calibrado leído: %.4f", threshold_optimal)
    except Exception as e:
        logger.error("ERROR al leer el umbral (%s): %s", THRESHOLD_V1, e)
        raise

    try:
        indexer_model = StringIndexerModel.load(INDEXER_PATH)
        etiquetas = list(indexer_model.labels)                         # labels[i] = nombre del índice i
        logger.info("StringIndexer cargado: %d clases conocidas.", len(etiquetas))
    except Exception as e:
        logger.error("ERROR al cargar el StringIndexer (%s): %s", INDEXER_PATH, e)
        raise

    kmeans_model = model.stages[-1]

    # ------------------------------------------------------------------
    # (c) UDF DE DISTANCIA (centros del modelo cargado)
    # ------------------------------------------------------------------
    udf_dist = construir_udf_distancia(spark, kmeans_model)

    # ------------------------------------------------------------------
    # (d) INFERENCIA DISTRIBUIDA CON MEDICIÓN DE TIEMPO
    # ------------------------------------------------------------------
    num_particiones = df_test.rdd.getNumPartitions()
    logger.info("Iniciando inferencia batch (particiones=%d)...", num_particiones)

    t_inicio = time.time()
    df_pred = model.transform(df_test)                                  # asigna cluster_id
    df_pred = df_pred.withColumn(                                       # añade distancia al centroide
        "distancia_centroide",
        udf_dist(col("scaled_features"), col("cluster_id")),
    )
    # (e) Clasificación binaria con el umbral calibrado.
    df_pred = df_pred.withColumn(
        "prediccion_binaria",
        (col("distancia_centroide") > threshold_optimal).cast("int"),
    )
    df_pred = df_pred.withColumn(
        "label_binaria",
        when(col("label_index") == 0.0, 0).otherwise(1),
    )
    df_pred = df_pred.cache()
    num_registros = df_pred.count()                                    # fuerza la inferencia
    t_fin = time.time()

    tiempo_inferencia = t_fin - t_inicio
    throughput = num_registros / tiempo_inferencia if tiempo_inferencia > 0 else 0.0
    latencia_us = (tiempo_inferencia / num_registros * 1_000_000) if num_registros > 0 else 0.0
    logger.info("Inferencia: %d registros en %.2f s | throughput=%.1f reg/s | latencia=%.2f µs/reg",
                num_registros, tiempo_inferencia, throughput, latencia_us)

    # ------------------------------------------------------------------
    # (f) MÉTRICAS BINARIAS
    # ------------------------------------------------------------------
    fila_cm = df_pred.agg(
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 1), 1).otherwise(0)).alias("tp"),
        Fsum(when((col("prediccion_binaria") == 1) & (col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 1), 1).otherwise(0)).alias("fn"),
        Fsum(when((col("prediccion_binaria") == 0) & (col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    tp, fp, fn, tn = int(fila_cm["tp"]), int(fila_cm["fp"]), int(fila_cm["fn"]), int(fila_cm["tn"])

    total = tp + fp + fn + tn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0               # FPR crítico para IDS

    # AUC-ROC usando la distancia como score continuo.
    df_pred = df_pred.withColumn(
        "raw_prediction_vec",
        distancia_a_vector_udf(col("distancia_centroide")),
    )
    auc_roc = float(BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df_pred))

    logger.info("MÉTRICAS BINARIAS -> AUC=%.4f | Acc=%.4f | P=%.4f | R(DR)=%.4f | F1=%.4f | FPR=%.4f",
                auc_roc, accuracy, precision, recall, f1, fpr)
    logger.info("Matriz de confusión -> TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)

    # ------------------------------------------------------------------
    # (g) DISTRIBUCIÓN DE DISTANCIAS POR CLASE REAL
    # ------------------------------------------------------------------
    df_por_clase = (
        df_pred.groupBy("label_index")
        .agg(
            Fmean("distancia_centroide").alias("media_dist"),
            Fstddev("distancia_centroide").alias("std_dist"),
            Fcount("*").alias("num_registros"),
            Fmean(col("prediccion_binaria").cast("double")).alias("tasa_deteccion"),
        )
        .orderBy("label_index")
    )

    filas_clase = df_por_clase.collect()
    tabla_por_clase = []
    logger.info("%-28s | %-10s | %-10s | %-12s | %-10s",
                "clase", "media_dist", "std_dist", "num_reg", "tasa_det")
    for f in filas_clase:
        idx = int(f["label_index"])
        nombre = etiquetas[idx] if 0 <= idx < len(etiquetas) else f"idx_{idx}"
        registro = {
            "clase": nombre, "label_index": idx,
            "media_dist": float(f["media_dist"]) if f["media_dist"] is not None else None,
            "std_dist": float(f["std_dist"]) if f["std_dist"] is not None else None,
            "num_registros": int(f["num_registros"]),
            "tasa_deteccion": float(f["tasa_deteccion"]) if f["tasa_deteccion"] is not None else None,
        }
        tabla_por_clase.append(registro)
        logger.info("%-28s | %-10.4f | %-10.4f | %-12d | %-10.4f",
                    nombre,
                    registro["media_dist"] or 0.0,
                    registro["std_dist"] or 0.0,
                    registro["num_registros"],
                    registro["tasa_deteccion"] or 0.0)

    # ------------------------------------------------------------------
    # (h) ANÁLISIS DE CLASES NO VISTAS (jueves: Web attacks + Infiltración)
    # ------------------------------------------------------------------
    indices_no_vistos = [i for i, nombre in enumerate(etiquetas)
                         if ("web attack" in nombre.lower()) or ("infiltration" in nombre.lower())]
    nombres_no_vistos = [etiquetas[i] for i in indices_no_vistos]
    logger.info("Clases NO vistas en entrenamiento/calibración: %s", nombres_no_vistos)

    analisis_no_vistos = []
    if indices_no_vistos:
        df_no_vistos = df_pred.filter(col("label_index").isin([float(i) for i in indices_no_vistos]))
        agg_nv = (
            df_no_vistos.groupBy("label_index")
            .agg(
                Fmean("distancia_centroide").alias("media_dist"),
                Fcount("*").alias("num_registros"),
                Fmean(col("prediccion_binaria").cast("double")).alias("tasa_deteccion"),
            )
            .collect()
        )
        for f in agg_nv:
            idx = int(f["label_index"])
            nombre = etiquetas[idx]
            tasa = float(f["tasa_deteccion"]) if f["tasa_deteccion"] is not None else 0.0
            media = float(f["media_dist"]) if f["media_dist"] is not None else 0.0
            analisis_no_vistos.append({
                "clase": nombre, "tasa_deteccion": tasa,
                "media_dist": media, "num_registros": int(f["num_registros"]),
            })
            detecta = "SÍ las detecta" if tasa >= 0.5 else "detección parcial/baja"
            logger.info("[NO VISTA] %s -> tasa_det=%.4f | media_dist=%.4f | %s",
                        nombre, tasa, media, detecta)
    else:
        logger.info("No se hallaron clases 'Web Attack'/'Infiltration' en el indexer.")

    # ------------------------------------------------------------------
    # (i) MÉTRICAS DE SISTEMA DISTRIBUIDO
    # ------------------------------------------------------------------
    metricas_sistema = {
        "throughput_registros_por_segundo": round(throughput, 2),
        "latencia_promedio_us": round(latencia_us, 4),
        "num_particiones": num_particiones,
        "tiempo_inferencia_segundos": round(tiempo_inferencia, 4),
    }
    logger.info("Métricas de sistema: %s", metricas_sistema)

    # ------------------------------------------------------------------
    # (j) GUARDADO DE MÉTRICAS Y PREDICCIONES
    # ------------------------------------------------------------------
    salida = {
        "threshold_usado": threshold_optimal,
        "binarias": {
            "auc_roc": auc_roc, "accuracy": accuracy,
            "precision": precision, "recall_detection_rate": recall,
            "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_registros": total,
        },
        "distribucion_por_clase": tabla_por_clase,
        "analisis_clases_no_vistas": analisis_no_vistos,
        "sistema_distribuido": metricas_sistema,
    }
    try:
        escribir_json_s3(salida, TEST_METRICS_OUT)
        logger.info("Métricas de prueba guardadas en %s", TEST_METRICS_OUT)
    except Exception as e:
        logger.error("ERROR al escribir métricas en %s: %s", TEST_METRICS_OUT, e)
        raise

    try:
        (df_pred
         .select("distancia_centroide", "cluster_id", "prediccion_binaria",
                 "label_binaria", "label_index")
         .write.mode("overwrite").parquet(PREDICTIONS_OUT))
        logger.info("Predicciones guardadas en %s", PREDICTIONS_OUT)
    except Exception as e:
        logger.error("ERROR al escribir predicciones en %s: %s", PREDICTIONS_OUT, e)
        raise

    df_pred.unpersist()
    logger.info("BLOQUE 3 completado correctamente.")


# ==============================================================
# === BLOQUE 4: OPTIMIZACIÓN DE HIPERPARÁMETROS ===
# (origen: 03c_optimize_kmeans.py)
# ==============================================================
def bloque4_optimizar_kmeans(spark):
    """Búsqueda manual sobre grilla (8 combinaciones) sobre una muestra
    del 30% del benigno. Selecciona por Silhouette (empate -> menor WSSSE),
    re-entrena sobre el benigno completo, re-calibra umbral y genera el
    reporte comparativo."""
    logger.info("=" * 60)
    logger.info("BLOQUE 4 — OPTIMIZACIÓN KMeans (búsqueda manual sobre grilla)")
    logger.info("=" * 60)

    # ADVERTENCIA: con spark.executor.instances=2 (clúster base), la búsqueda
    # sobre grilla es costosa. Se usa el 30% de muestra del benigno para la
    # búsqueda; el modelo ganador se re-entrena sobre el benigno completo.
    # KMeans NO es compatible con CrossValidator de MLlib -> búsqueda manual.

    # ------------------------------------------------------------------
    # (b) CARGA DEL BENIGNO + MUESTRA DEL 30%
    # ------------------------------------------------------------------
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Parquet de entrenamiento cargado desde %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        raise

    df_benigno = df_train.filter(col("label_index") == 0.0).cache()
    n_benigno = df_benigno.count()
    df_muestra = df_benigno.sample(withReplacement=False,
                                   fraction=SAMPLE_FRACTION, seed=SEED).cache()
    n_muestra = df_muestra.count()
    logger.info("Benigno total=%d | muestra (30%%)=%d", n_benigno, n_muestra)

    # ------------------------------------------------------------------
    # (c) GRILLA DE HIPERPARÁMETROS MANUAL (8 combinaciones)
    # ------------------------------------------------------------------
    param_grid = [
        {"k": 5,  "maxIter": 20, "tol": 1e-4},
        {"k": 10, "maxIter": 20, "tol": 1e-4},
        {"k": 15, "maxIter": 20, "tol": 1e-4},
        {"k": 20, "maxIter": 50, "tol": 1e-4},
        {"k": 30, "maxIter": 50, "tol": 1e-4},
        {"k": 10, "maxIter": 50, "tol": 1e-5},
        {"k": 15, "maxIter": 50, "tol": 1e-5},
        {"k": 20, "maxIter": 50, "tol": 1e-5},
    ]
    # Total: 8 combinaciones. Cada entrenamiento trabaja sobre el 30% del benigno.
    logger.info("Grilla de búsqueda: %d combinaciones sobre la muestra del 30%%.", len(param_grid))

    evaluador_silhouette = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="cluster_id",
        metricName="silhouette", distanceMeasure="squaredEuclidean")

    # ------------------------------------------------------------------
    # (d) BUCLE DE BÚSQUEDA
    # ------------------------------------------------------------------
    resultados_grilla = []
    for i, params in enumerate(param_grid, start=1):
        logger.info("[%d/%d] Entrenando k=%d maxIter=%d tol=%g sobre la muestra...",
                    i, len(param_grid), params["k"], params["maxIter"], params["tol"])
        km = KMeans(
            k=params["k"], maxIter=params["maxIter"], tol=params["tol"],
            seed=SEED, initMode="k-means||",
            featuresCol="scaled_features", predictionCol="cluster_id",
            distanceMeasure="euclidean")
        pipe = Pipeline(stages=[km])

        t0 = time.time()
        modelo_tmp = pipe.fit(df_muestra)
        t_entreno = time.time() - t0

        km_tmp = modelo_tmp.stages[-1]
        wssse = float(km_tmp.summary.trainingCost)

        df_eval = modelo_tmp.transform(df_muestra)
        silhouette = float(evaluador_silhouette.evaluate(df_eval))

        # Media/desviación de distancias sobre la muestra.
        udf_dist = construir_udf_distancia(spark, km_tmp)
        df_eval = df_eval.withColumn(
            "distancia_centroide",
            udf_dist(col("scaled_features"), col("cluster_id")))
        stats = df_eval.agg(
            Fmean("distancia_centroide").alias("m"),
            Fstddev("distancia_centroide").alias("s")).first()
        dist_media = float(stats["m"]) if stats["m"] is not None else 0.0
        dist_std   = float(stats["s"]) if stats["s"] is not None else 0.0

        resultado = {
            "k": params["k"], "maxIter": params["maxIter"], "tol": params["tol"],
            "silhouette": silhouette, "wssse": wssse,
            "dist_media": dist_media, "dist_std": dist_std,
            "tiempo_entreno_s": round(t_entreno, 4),
        }
        resultados_grilla.append(resultado)
        logger.info("   -> Silhouette=%.4f | WSSSE=%.2f | dist_media=%.4f | T=%.2fs",
                    silhouette, wssse, dist_media, t_entreno)

    # ------------------------------------------------------------------
    # (e) SELECCIÓN DEL MEJOR MODELO
    # ------------------------------------------------------------------
    # Maximizar Silhouette; empate (diferencia < 0.01) -> menor WSSSE.
    mejor = resultados_grilla[0]
    for r in resultados_grilla[1:]:
        diff = r["silhouette"] - mejor["silhouette"]
        if diff > 0.01:                                              # mejora clara de Silhouette
            mejor = r
        elif abs(diff) <= 0.01 and r["wssse"] < mejor["wssse"]:      # empate -> menor WSSSE
            mejor = r
    logger.info("MEJOR CONFIG: k=%d maxIter=%d tol=%g | Silhouette=%.4f | WSSSE=%.2f "
                "(criterio: máx. Silhouette, empate<0.01 -> mín. WSSSE)",
                mejor["k"], mejor["maxIter"], mejor["tol"], mejor["silhouette"], mejor["wssse"])

    # ------------------------------------------------------------------
    # (f) RE-ENTRENAMIENTO SOBRE EL BENIGNO COMPLETO
    # ------------------------------------------------------------------
    logger.info("Re-entrenando el mejor modelo sobre el benigno COMPLETO (%d registros)...", n_benigno)
    km_final = KMeans(
        k=mejor["k"], maxIter=mejor["maxIter"], tol=mejor["tol"],
        seed=SEED, initMode="k-means||",
        featuresCol="scaled_features", predictionCol="cluster_id",
        distanceMeasure="euclidean")
    pipe_final = Pipeline(stages=[km_final])

    t0 = time.time()
    model_opt = pipe_final.fit(df_benigno)
    tiempo_entreno_final = time.time() - t0
    logger.info("Re-entrenamiento completado en %.2f s", tiempo_entreno_final)

    km_opt = model_opt.stages[-1]
    silhouette_opt = float(evaluador_silhouette.evaluate(model_opt.transform(df_benigno)))
    wssse_opt = float(km_opt.summary.trainingCost)

    # ------------------------------------------------------------------
    # (g) RE-CALIBRACIÓN DEL UMBRAL (modelo optimizado)
    # ------------------------------------------------------------------
    logger.info("Re-calibrando umbral del modelo optimizado (DataFrame completo de entrenamiento)...")
    calib_opt = _calibrar_umbral_interno(spark, model_opt, df_train)
    umbral_opt = calib_opt["umbral"]
    logger.info("Umbral optimizado: %.4f (F1=%.4f, FPR=%.4f, AUC=%.4f)",
                umbral_opt, calib_opt["f1"], calib_opt["fpr"], calib_opt["auc_roc"])

    threshold_opt_json = {
        "threshold_optimal": umbral_opt,
        "f1_at_threshold": calib_opt["f1"],
        "precision": calib_opt["precision"],
        "recall": calib_opt["recall"],
        "fpr": calib_opt["fpr"],
        "auc_roc": calib_opt["auc_roc"],
        "percentil_90": calib_opt["percentiles"]["p90"],
        "percentil_95": calib_opt["percentiles"]["p95"],
        "percentil_97": calib_opt["percentiles"]["p97"],
        "percentil_99": calib_opt["percentiles"]["p99"],
        "percentil_999": calib_opt["percentiles"]["p999"],
        "calibration_days": ["Wednesday", "Friday"],
        "model": "model_optimized",
    }
    try:
        escribir_json_s3(threshold_opt_json, THRESHOLD_OPT)
        logger.info("Umbral optimizado guardado en %s", THRESHOLD_OPT)
    except Exception as e:
        logger.error("ERROR al escribir el umbral opt (%s): %s", THRESHOLD_OPT, e)
        raise

    # ------------------------------------------------------------------
    # (h) RE-EVALUACIÓN SOBRE EL CONJUNTO DE PRUEBA
    # ------------------------------------------------------------------
    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Parquet de prueba cargado desde %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de prueba (%s): %s", TEST_PROCESSED, e)
        raise

    logger.info("Evaluando el modelo optimizado sobre el set de prueba con umbral=%.4f...", umbral_opt)
    test_opt = _evaluar_en_test_interno(spark, model_opt, df_test, umbral_opt)
    logger.info("TEST optimizado -> AUC=%.4f | Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f | FPR=%.4f",
                test_opt["auc_roc"], test_opt["accuracy"], test_opt["precision"],
                test_opt["recall"], test_opt["f1"], test_opt["fpr"])

    # ------------------------------------------------------------------
    # (i) GUARDADO DEL MEJOR MODELO + REPORTE COMPARATIVO
    # ------------------------------------------------------------------
    try:
        model_opt.write().overwrite().save(MODEL_OPT_OUT)
        logger.info("Modelo optimizado guardado en %s", MODEL_OPT_OUT)
    except Exception as e:
        logger.error("ERROR al guardar el modelo optimizado (%s): %s", MODEL_OPT_OUT, e)
        raise

    # Cargar métricas del modelo base para la comparación.
    train_base = leer_json_s3(TRAIN_METRICS) or {}
    test_base  = leer_json_s3(TEST_METRICS_OUT) or {}
    test_base_bin = test_base.get("binarias", {})

    reporte = {
        "modelo_base": {
            "hiperparametros": {
                "k": train_base.get("k", 10),
                "maxIter": train_base.get("max_iter", 50),
                "tol": 1e-4,
                "initMode": train_base.get("init_mode", "k-means||"),
            },
            "silhouette": train_base.get("silhouette_score"),
            "wssse": train_base.get("wssse"),
            "auc_roc": test_base_bin.get("auc_roc"),
            "f1_binario": test_base_bin.get("f1"),
            "fpr": test_base_bin.get("fpr"),
            "accuracy_binaria": test_base_bin.get("accuracy"),
            "umbral_calibrado": test_base.get("threshold_usado"),
            "tiempo_entreno_s": train_base.get("training_time_seconds"),
        },
        "modelo_optimizado": {
            "hiperparametros": {
                "k": mejor["k"], "maxIter": mejor["maxIter"],
                "tol": mejor["tol"], "initMode": "k-means||",
            },
            "silhouette": silhouette_opt,
            "wssse": wssse_opt,
            "auc_roc": test_opt["auc_roc"],
            "f1_binario": test_opt["f1"],
            "fpr": test_opt["fpr"],
            "accuracy_binaria": test_opt["accuracy"],
            "umbral_calibrado": umbral_opt,
            "tiempo_entreno_s": round(tiempo_entreno_final, 4),
        },
        "grilla_busqueda": resultados_grilla,
        "criterio_seleccion": "max Silhouette; empate(<0.01) -> min WSSSE",
        "muestra_fraccion": SAMPLE_FRACTION,
    }
    try:
        escribir_json_s3(reporte, OPT_REPORT_OUT)
        logger.info("Reporte de optimización guardado en %s", OPT_REPORT_OUT)
    except Exception as e:
        logger.error("ERROR al escribir el reporte (%s): %s", OPT_REPORT_OUT, e)
        raise

    # ------------------------------------------------------------------
    # (j) TABLA COMPARATIVA EN CONSOLA
    # ------------------------------------------------------------------
    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/D"

    base = reporte["modelo_base"]
    opt  = reporte["modelo_optimizado"]
    logger.info("| Configuración       | Silhouette | AUC-ROC | F1 bin | FPR   | T. entreno |")
    logger.info("|---------------------|------------|---------|--------|-------|------------|")
    logger.info("| Modelo base (k=%-2s)  |   %-6s   |  %-5s  | %-5s  | %-5s | %8s s |",
                base["hiperparametros"]["k"], _fmt(base["silhouette"]), _fmt(base["auc_roc"]),
                _fmt(base["f1_binario"]), _fmt(base["fpr"]), _fmt(base["tiempo_entreno_s"]))
    logger.info("| Modelo optimizado   |   %-6s   |  %-5s  | %-5s  | %-5s | %8s s |",
                _fmt(opt["silhouette"]), _fmt(opt["auc_roc"]), _fmt(opt["f1_binario"]),
                _fmt(opt["fpr"]), _fmt(opt["tiempo_entreno_s"]))

    df_muestra.unpersist()
    df_benigno.unpersist()
    logger.info("BLOQUE 4 completado correctamente.")


# ==============================================================
# === MAIN: ORQUESTACIÓN COMPLETA DEL PIPELINE KMeans ===
# ==============================================================
def main():
    """Ejecuta en secuencia los cuatro bloques del pipeline KMeans con
    una única SparkSession compartida."""
    spark = construir_spark()
    logger.info("SparkSession 'ProyectoAnomalias-KMeans' iniciada.")
    logger.info("Iniciando pipeline completo de KMeans (4 bloques).")

    try:
        bloque1_entrenar_kmeans(spark)      # Entrenamiento del modelo base (solo benigno)
        bloque2_calibrar_kmeans(spark)      # Calibración del umbral de distancia
        bloque3_evaluar_kmeans(spark)       # Evaluación batch sobre prueba
        bloque4_optimizar_kmeans(spark)     # Optimización de hiperparámetros
    except Exception as e:
        logger.error("ERROR en el pipeline KMeans: %s", e)
        spark.stop()
        raise

    spark.stop()
    logger.info("Pipeline KMeans finalizado correctamente (4/4 bloques).")


if __name__ == "__main__":
    main()
