#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_random_forest.py
==================================================================
PROPÓSITO:
    Ejecuta en secuencia los tres bloques del pipeline supervisado
    de detección de anomalías basado en Random Forest sobre el dataset
    CIC-IDS2017:

    BLOQUE 1 — ENTRENAMIENTO (02_train_rf.py):
        Entrena el clasificador RF base (model_v1) sobre el Parquet
        procesado, usando pesos por clase para manejar el desbalance.
        Guarda el modelo y sus métricas de entrenamiento.

    BLOQUE 2 — EVALUACIÓN BATCH (04_evaluate_batch.py):
        Evalúa model_v1 sobre el conjunto de prueba (martes + jueves).
        Reporta métricas multiclase, binarias, matriz de confusión,
        análisis de clases NO vistas y métricas de sistema distribuido.

    BLOQUE 3 — OPTIMIZACIÓN (02b_optimize_rf.py):
        Busca los mejores hiperparámetros mediante CrossValidator sobre
        una muestra estratificada del 30%, re-entrena el modelo ganador
        sobre el entrenamiento completo, lo evalúa en prueba y genera
        un reporte comparativo base vs. optimizado.

INPUTS (rutas S3 de entrada):
    - s3://proyecto-anomalias-emr/data/cic-ids2017/train/processed/
    - s3://proyecto-anomalias-emr/data/cic-ids2017/test/processed/

OUTPUTS (rutas S3 de salida):
    - .../models/random_forest_pipeline/model_v1/
    - .../models/random_forest_pipeline/model_optimized/
    - .../results/batch_evaluation/train_metrics_rf.json
    - .../results/batch_evaluation/test_metrics_rf.json
    - .../results/batch_evaluation/predictions_rf/          (Parquet)
    - .../results/batch_evaluation/optimization_report_rf.json

DEPENDENCIAS (ejecutar ANTES):
    - gestion_datos.py   (Parquet procesados + preprocesadores en S3)

ORDEN DE EJECUCIÓN DEL PROYECTO:
    1. gestion_datos.py
    2. pipeline_random_forest.py  ← este script
    3. pipeline_kmeans.py
==================================================================
"""

# === IMPORTS ===
import time                                          # medición de tiempos
import json                                          # serialización de métricas
import logging                                       # logging estructurado
from urllib.parse import urlparse                    # parseo de rutas s3://

import boto3                                         # I/O de JSON con S3
import numpy as np                                   # argmax sobre métricas de CV

from pyspark.sql import SparkSession                 # punto de entrada de Spark
from pyspark.sql import functions as F               # funciones SQL
from pyspark.sql.functions import udf                 # decorador de UDF
from pyspark.sql.types import DoubleType            # tipo de UDF
from pyspark.ml import Pipeline, PipelineModel       # pipeline de MLlib
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexerModel    # nombres de clase
from pyspark.ml.linalg import Vectors, VectorUDT     # vector para AUC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.mllib.evaluation import MulticlassMetrics  # métricas multiclase (RDD)


# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PipelineRandomForest")


# ==============================================================
# === CONFIGURACIÓN GLOBAL ===
# ==============================================================
BUCKET        = "s3://proyecto-anomalias-emr"       # bucket raíz
TRAIN_RAW     = f"{BUCKET}/data/cic-ids2017/train/"             # entrenamiento
TEST_RAW      = f"{BUCKET}/data/cic-ids2017/test/"              # prueba
MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"     # modelos
RESULTS_PATH  = f"{BUCKET}/results/batch_evaluation/"          # resultados

# Rutas derivadas.
TRAIN_PROCESSED  = f"{TRAIN_RAW}processed/"                    # Parquet de entrenamiento
TEST_PROCESSED   = f"{TEST_RAW}processed/"                     # Parquet de prueba
INDEXER_PATH     = f"{MODELS_PATH}preprocessors/indexer/"     # StringIndexer ajustado
MODEL_V1_OUT     = f"{MODELS_PATH}model_v1/"                   # modelo base
MODEL_OPT_OUT    = f"{MODELS_PATH}model_optimized/"            # modelo optimizado
TRAIN_METRICS    = f"{RESULTS_PATH}train_metrics_rf.json"      # métricas de entrenamiento
TEST_METRICS_OUT = f"{RESULTS_PATH}test_metrics_rf.json"       # métricas de prueba
PREDICTIONS_OUT  = f"{RESULTS_PATH}predictions_rf/"            # predicciones Parquet
OPT_REPORT_OUT   = f"{RESULTS_PATH}optimization_report_rf.json"  # reporte comparativo

# --- Hiperparámetros Bloque 1 (modelo base) ---
NUM_TREES = 100                                                # nº de árboles
MAX_DEPTH = 10                                                 # profundidad máxima
FSS       = "sqrt"                                             # featureSubsetStrategy
SEED      = 42                                                 # semilla

# --- Parámetros Bloque 3 (optimización) ---
SAMPLE_FRACTION = 0.30                                         # fracción para CrossValidator
NUM_FOLDS       = 3                                            # folds del CrossValidator

# --- Parámetros de clúster ---
EXECUTOR_INSTANCES = 2                                          # nº de executors
EXECUTOR_CORES     = 2                                          # cores por executor
SHUFFLE_PARTITIONS = 16                                         # particiones de shuffle


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


def nombres_features(df, col_vec):
    """Recupera los nombres de las features desde la metadata ML del vector.
    Devuelve la lista en el orden de los índices del vector."""
    meta = df.schema[col_vec].metadata                         # metadata de la columna vectorial
    grupos = meta.get("ml_attr", {}).get("attrs", {})          # atributos (numeric/binary/nominal)
    indice_a_nombre = {}                                       # mapa idx -> nombre
    for lista in grupos.values():                              # recorrer cada grupo de atributos
        for attr in lista:
            indice_a_nombre[attr["idx"]] = attr["name"]
    if not indice_a_nombre:                                    # fallback si no hay metadata
        return None
    return [indice_a_nombre[i] for i in sorted(indice_a_nombre)]


def construir_spark():
    """SparkSession con TODOS los parámetros del clúster vía .config()."""
    spark = (
        SparkSession.builder
        .appName("ProyectoAnomalias-RandomForest")                                 # nombre exacto requerido
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
        .config("spark.kryo.registrationRequired", "false")
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
# === BLOQUE 1: ENTRENAMIENTO DEL MODELO BASE ===
# (origen: 02_train_rf.py)
# ==============================================================
def bloque1_entrenar_rf(spark):
    """Entrena el Random Forest base (model_v1) sobre el conjunto de
    entrenamiento procesado. Guarda el modelo y las métricas en S3."""
    logger.info("=" * 60)
    logger.info("BLOQUE 1 — ENTRENAMIENTO RF (model_v1)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # (c) CARGA DEL DATAFRAME DE ENTRENAMIENTO PROCESADO
    # ------------------------------------------------------------------
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)          # lectura de Parquet
        logger.info("Parquet de entrenamiento cargado desde %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        raise

    num_registros = df_train.count()                            # nº de registros de entrenamiento
    logger.info("Registros de entrenamiento: %d", num_registros)

    # ------------------------------------------------------------------
    # (d) DEFINICIÓN DEL CLASIFICADOR
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        numTrees=NUM_TREES,                     # 100 árboles
        maxDepth=MAX_DEPTH,                      # profundidad máxima 10
        featureSubsetStrategy=FSS,              # "sqrt"
        seed=SEED,                              # reproducibilidad
        labelCol="label_index",                 # etiqueta numérica
        featuresCol="scaled_features",          # features escaladas
        weightCol="class_weight",               # pesos para el desbalance
    )

    # ------------------------------------------------------------------
    # (e) PIPELINE con el clasificador como única etapa
    # ------------------------------------------------------------------
    pipeline = Pipeline(stages=[rf])                            # preprocesamiento ya hecho en gestion_datos.py

    # ------------------------------------------------------------------
    # (f) ENTRENAMIENTO DISTRIBUIDO CON MEDICIÓN DE TIEMPO
    # ------------------------------------------------------------------
    logger.info("Iniciando entrenamiento RF (numTrees=%d, maxDepth=%d, fss=%s)...",
                NUM_TREES, MAX_DEPTH, FSS)
    t0 = time.time()                                            # marca previa
    model = pipeline.fit(df_train)                              # entrenamiento
    t1 = time.time()                                            # marca posterior
    training_time = t1 - t0                                     # tiempo total (s)
    logger.info("Entrenamiento completado en %.2f s", training_time)

    rf_model = model.stages[-1]                                 # RandomForestClassificationModel

    # ------------------------------------------------------------------
    # (g) GUARDADO DEL MODELO
    # ------------------------------------------------------------------
    try:
        model.write().overwrite().save(MODEL_V1_OUT)            # serializar PipelineModel
        logger.info("Modelo RF guardado en %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR al guardar el modelo (%s): %s", MODEL_V1_OUT, e)
        raise

    # ------------------------------------------------------------------
    # (h) FEATURE IMPORTANCES ORDENADAS CON NOMBRES
    # ------------------------------------------------------------------
    importancias = rf_model.featureImportances.toArray()        # vector de importancias
    nombres = nombres_features(df_train, "scaled_features")     # nombres desde metadata
    if nombres is None:                                         # fallback si no hay metadata
        nombres = [f"f{i}" for i in range(len(importancias))]
    pares = sorted(zip(nombres, importancias), key=lambda x: x[1], reverse=True)  # orden desc
    logger.info("Top 15 features por importancia:")
    for nombre, imp in pares[:15]:                              # imprimir las 15 principales
        logger.info("   %-32s %.6f", nombre, imp)
    top_importancias = [{"feature": n, "importance": float(v)} for n, v in pares]  # para el JSON

    # ------------------------------------------------------------------
    # (i) GUARDADO DE MÉTRICAS DE ENTRENAMIENTO
    # ------------------------------------------------------------------
    metricas = {
        "training_time_seconds": round(training_time, 4),       # tiempo de entrenamiento
        "num_trees": NUM_TREES,                                 # nº de árboles
        "max_depth": MAX_DEPTH,                                 # profundidad máxima
        "feature_subset_strategy": FSS,                         # estrategia de subconjunto
        "num_partitions": SHUFFLE_PARTITIONS,                   # particiones de shuffle
        "num_training_records": int(num_registros),             # registros de entrenamiento
        "executor_instances": EXECUTOR_INSTANCES,               # nº de executors
        "cluster_cores_total": EXECUTOR_INSTANCES * EXECUTOR_CORES,  # cores totales
        "feature_importances": top_importancias,                # importancias ordenadas
    }
    try:
        escribir_json_s3(metricas, TRAIN_METRICS)               # escribir JSON a S3
        logger.info("Métricas de entrenamiento guardadas en %s", TRAIN_METRICS)
    except Exception as e:
        logger.error("ERROR al escribir métricas (%s): %s", TRAIN_METRICS, e)
        raise

    logger.info("BLOQUE 1 completado correctamente.")


# ==============================================================
# === BLOQUE 2: EVALUACIÓN BATCH DEL MODELO BASE ===
# (origen: 04_evaluate_batch.py)
# ==============================================================
def bloque2_evaluar_rf(spark):
    """Evalúa model_v1 sobre el conjunto de prueba. Reporta métricas
    multiclase, binarias, matriz de confusión y métricas de sistema."""
    logger.info("=" * 60)
    logger.info("BLOQUE 2 — EVALUACIÓN BATCH RF (model_v1 sobre prueba)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # (b) CARGA DE MODELO, PRUEBA E INDEXER
    # ------------------------------------------------------------------
    try:
        model = PipelineModel.load(MODEL_V1_OUT)                # modelo RF
        logger.info("Modelo RF cargado desde %s", MODEL_V1_OUT)
    except Exception as e:
        logger.error("ERROR al cargar el modelo (%s): %s", MODEL_V1_OUT, e)
        raise

    try:
        df_test = spark.read.parquet(TEST_PROCESSED)            # Parquet de prueba
        logger.info("Parquet de prueba cargado desde %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de prueba (%s): %s", TEST_PROCESSED, e)
        raise

    try:
        indexer_model = StringIndexerModel.load(INDEXER_PATH)   # nombres de clases conocidas
        etiquetas = list(indexer_model.labels)                 # labels[i] = nombre del índice i
        logger.info("StringIndexer cargado: %d clases conocidas.", len(etiquetas))
    except Exception as e:
        logger.error("ERROR al cargar el StringIndexer (%s): %s", INDEXER_PATH, e)
        raise

    # ------------------------------------------------------------------
    # (c) INFERENCIA DISTRIBUIDA CON MEDICIÓN DE TIEMPO
    # ------------------------------------------------------------------
    num_particiones = df_test.rdd.getNumPartitions()           # particiones del DataFrame
    logger.info("Iniciando inferencia batch (particiones=%d)...", num_particiones)

    t0 = time.time()                                           # marca previa
    df_pred = model.transform(df_test)                         # añade prediction/probability
    df_pred = df_pred.cache()                                  # cachear para reuso en métricas
    num_registros = df_pred.count()                            # acción que fuerza la inferencia
    t1 = time.time()                                          # marca posterior

    tiempo_inferencia = t1 - t0                                # tiempo total (s)
    throughput = num_registros / tiempo_inferencia if tiempo_inferencia > 0 else 0.0
    latencia_us = (tiempo_inferencia / num_registros * 1_000_000) if num_registros > 0 else 0.0
    logger.info("Inferencia: %d registros en %.2f s | throughput=%.1f reg/s | latencia=%.2f µs/reg",
                num_registros, tiempo_inferencia, throughput, latencia_us)

    # ------------------------------------------------------------------
    # (d) MÉTRICAS MULTICLASE Y BINARIAS
    # ------------------------------------------------------------------
    # MulticlassMetrics requiere un RDD de (prediccion, etiqueta) como floats.
    rdd_pred = df_pred.select("prediction", "label_index").rdd.map(
        lambda r: (float(r["prediction"]), float(r["label_index"])))
    mm = MulticlassMetrics(rdd_pred)                           # métricas multiclase

    accuracy = float(mm.accuracy)                              # exactitud global
    weighted_precision = float(mm.weightedPrecision)           # precision weighted
    weighted_recall = float(mm.weightedRecall)                 # recall weighted
    weighted_f1 = float(mm.weightedFMeasure())                 # F1 weighted

    # Macro = promedio simple de la métrica por clase (sin ponderar por soporte).
    etiquetas_presentes = sorted(
        [float(r["label_index"]) for r in df_pred.select("label_index").distinct().collect()])
    macro_precision = sum(mm.precision(l) for l in etiquetas_presentes) / len(etiquetas_presentes)
    macro_recall    = sum(mm.recall(l) for l in etiquetas_presentes) / len(etiquetas_presentes)
    macro_f1        = sum(mm.fMeasure(l) for l in etiquetas_presentes) / len(etiquetas_presentes)

    logger.info("MULTICLASE -> Acc=%.4f | F1(w)=%.4f F1(macro)=%.4f | P(w)=%.4f P(macro)=%.4f | R(w)=%.4f R(macro)=%.4f",
                accuracy, weighted_f1, macro_f1, weighted_precision, macro_precision,
                weighted_recall, macro_recall)

    # --- Métricas binarias (benigno=0 vs ataque>0) ---
    df_pred = df_pred.withColumn(                              # etiqueta binaria real
        "label_binaria", F.when(F.col("label_index") == 0.0, 0).otherwise(1))
    df_pred = df_pred.withColumn(                              # predicción binaria
        "prediccion_binaria", F.when(F.col("prediction") == 0.0, 0).otherwise(1))

    # AUC-ROC: score = probabilidad de ataque = 1 - P(benigno) (índice 0).
    @udf(VectorUDT())
    def prob_ataque_vec(prob):
        """Devuelve [P(benigno), P(ataque)] a partir del vector de probabilidad."""
        p_benigno = float(prob[0])                            # probabilidad de la clase 0
        return Vectors.dense([p_benigno, 1.0 - p_benigno])    # [neg, pos]

    df_pred = df_pred.withColumn("score_vec", prob_ataque_vec(F.col("probability")))
    auc_roc = float(BinaryClassificationEvaluator(
        rawPredictionCol="score_vec", labelCol="label_binaria",
        metricName="areaUnderROC").evaluate(df_pred))
    logger.info("AUC-ROC (benigno vs ataque): %.4f", auc_roc)

    # FPR global (binario): falsos positivos sobre total de benignos.
    cm_bin = df_pred.agg(
        F.sum(F.when((F.col("prediccion_binaria") == 1) & (F.col("label_binaria") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("prediccion_binaria") == 0) & (F.col("label_binaria") == 0), 1).otherwise(0)).alias("tn"),
    ).first()
    fp_g, tn_g = int(cm_bin["fp"]), int(cm_bin["tn"])
    fpr_global = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0
    logger.info("False Positive Rate global: %.4f", fpr_global)

    # ------------------------------------------------------------------
    # (e) MATRIZ DE CONFUSIÓN POR CLASE (con nombres originales)
    # ------------------------------------------------------------------
    # Nombre de clase: para clases conocidas usar labels del indexer; para
    # cualquier índice fuera de rango (etiquetas no vistas agrupadas por
    # handleInvalid='keep') usar un marcador genérico.
    def nombre_de_indice(idx):
        i = int(idx)
        return etiquetas[i] if 0 <= i < len(etiquetas) else "__no_visto__"

    filas_cm = (df_pred.groupBy("label_index", "prediction").count()
                .orderBy("label_index", "prediction").collect())
    matriz = []                                               # matriz para el JSON
    for r in filas_cm:
        matriz.append({
            "clase_real": nombre_de_indice(r["label_index"]),
            "label_index": int(r["label_index"]),
            "clase_predicha": nombre_de_indice(r["prediction"]),
            "prediction": int(r["prediction"]),
            "conteo": int(r["count"]),
        })
    logger.info("Matriz de confusión calculada (%d celdas no vacías).", len(matriz))

    # Métricas por clase (precision/recall/F1) usando MulticlassMetrics.
    por_clase = []
    logger.info("%-28s | %-9s | %-9s | %-9s", "clase", "precision", "recall", "f1")
    for l in etiquetas_presentes:
        nombre = nombre_de_indice(l)
        p, r_, f = mm.precision(l), mm.recall(l), mm.fMeasure(l)
        por_clase.append({"clase": nombre, "label_index": int(l),
                          "precision": p, "recall": r_, "f1": f})
        logger.info("%-28s | %-9.4f | %-9.4f | %-9.4f", nombre, p, r_, f)

    # ------------------------------------------------------------------
    # (f) ANÁLISIS DE CLASES NO VISTAS (Web attacks + Infiltración - jueves)
    # ------------------------------------------------------------------
    # Estas clases NO aparecen en entrenamiento (lun/mié/vie). Se identifican
    # por la columna ORIGINAL 'label' (string), siempre disponible en el
    # Parquet de prueba, ya que el indexer ajustado en train no las conoce.
    df_no_vistos = df_pred.filter(
        (F.lower(F.col("label")).contains("web attack")) |
        (F.lower(F.col("label")).contains("infiltration")))
    analisis_no_vistos = []
    grupos_nv = (df_no_vistos.groupBy("label")
                 .agg(F.count("*").alias("n"),
                      F.sum(F.col("prediccion_binaria")).alias("detectados"))
                 .collect())
    for g in grupos_nv:
        n = int(g["n"])
        detectados = int(g["detectados"])                     # nº predichos como ataque
        recall_clase = detectados / n if n > 0 else 0.0       # recall = detectados / total
        analisis_no_vistos.append({
            "clase": g["label"], "num_registros": n,
            "detectados_como_ataque": detectados,
            "recall_deteccion": recall_clase,
        })
        logger.info("[NO VISTA] %-28s | n=%-8d | recall_detección=%.4f",
                    g["label"], n, recall_clase)

    # ------------------------------------------------------------------
    # (g) MÉTRICAS DE SISTEMA DISTRIBUIDO
    # ------------------------------------------------------------------
    metricas_sistema = {
        "throughput_registros_por_segundo": round(throughput, 2),
        "latencia_promedio_us": round(latencia_us, 4),
        "num_particiones": num_particiones,
        "tiempo_inferencia_segundos": round(tiempo_inferencia, 4),
    }
    logger.info("Métricas de sistema: %s", metricas_sistema)

    # ------------------------------------------------------------------
    # (h) GUARDADO DE MÉTRICAS Y PREDICCIONES
    # ------------------------------------------------------------------
    salida = {
        "multiclase": {
            "accuracy": accuracy,
            "f1_weighted": weighted_f1, "f1_macro": macro_f1,
            "precision_weighted": weighted_precision, "precision_macro": macro_precision,
            "recall_weighted": weighted_recall, "recall_macro": macro_recall,
        },
        "binarias": {
            "auc_roc": auc_roc,
            "fpr_global": fpr_global,
        },
        "matriz_confusion": matriz,
        "metricas_por_clase": por_clase,
        "analisis_clases_no_vistas": analisis_no_vistos,
        "sistema_distribuido": metricas_sistema,
    }
    try:
        escribir_json_s3(salida, TEST_METRICS_OUT)              # JSON de métricas
        logger.info("Métricas de prueba guardadas en %s", TEST_METRICS_OUT)
    except Exception as e:
        logger.error("ERROR al escribir métricas (%s): %s", TEST_METRICS_OUT, e)
        raise

    try:
        (df_pred
         .select("label", "label_index", "prediction", "prediccion_binaria",
                 "label_binaria", "probability")
         .write.mode("overwrite").parquet(PREDICTIONS_OUT))     # predicciones Parquet
        logger.info("Predicciones guardadas en %s", PREDICTIONS_OUT)
    except Exception as e:
        logger.error("ERROR al escribir predicciones (%s): %s", PREDICTIONS_OUT, e)
        raise

    df_pred.unpersist()
    logger.info("BLOQUE 2 completado correctamente.")


# ==============================================================
# === BLOQUE 3: OPTIMIZACIÓN DE HIPERPARÁMETROS ===
# (origen: 02b_optimize_rf.py)
# ==============================================================
def bloque3_optimizar_rf(spark):
    """Busca los mejores hiperparámetros con CrossValidator sobre una
    muestra estratificada, re-entrena sobre el dataset completo y genera
    un reporte comparativo frente al modelo base (model_v1)."""
    logger.info("=" * 60)
    logger.info("BLOQUE 3 — OPTIMIZACIÓN RF (CrossValidator 30% muestra)")
    logger.info("=" * 60)

    # ADVERTENCIA: con spark.executor.instances=2 (clúster base de 2 core nodes),
    # el CrossValidator (18 combinaciones × 3 folds = 54 entrenamientos) es muy
    # costoso. Por eso se ejecuta sobre una muestra ESTRATIFICADA del 30% y solo
    # el modelo ganador se re-entrena sobre el conjunto completo.

    # ------------------------------------------------------------------
    # CARGA DEL ENTRENAMIENTO COMPLETO
    # ------------------------------------------------------------------
    try:
        df_train = spark.read.parquet(TRAIN_PROCESSED)
        logger.info("Parquet de entrenamiento cargado desde %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        raise

    # ------------------------------------------------------------------
    # (d) MUESTRA ESTRATIFICADA DEL 30% (sampleBy sobre label_index)
    # ------------------------------------------------------------------
    indices = [r["label_index"] for r in df_train.select("label_index").distinct().collect()]
    fracciones = {idx: SAMPLE_FRACTION for idx in indices}      # misma fracción por clase
    df_muestra = df_train.sampleBy("label_index", fractions=fracciones, seed=SEED).cache()
    n_muestra = df_muestra.count()
    logger.info("Muestra estratificada (30%%): %d registros sobre %d clases.", n_muestra, len(indices))

    # ------------------------------------------------------------------
    # (b)(c) GRILLA + CROSSVALIDATOR
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        labelCol="label_index", featuresCol="scaled_features",
        weightCol="class_weight", seed=SEED)
    pipeline = Pipeline(stages=[rf])                            # estimador del CV

    grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100, 200])               # 3 valores
            .addGrid(rf.maxDepth, [5, 10, 15])                  # 3 valores
            .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"])  # 2 valores
            .build())                                          # 3×3×2 = 18 combinaciones
    logger.info("Grilla: %d combinaciones × %d folds = %d entrenamientos.",
                len(grid), NUM_FOLDS, len(grid) * NUM_FOLDS)
    # Estimación de tiempo: si el entrenamiento base tarda T s, el CrossValidator
    # tardará aproximadamente 54 × T s (sobre la muestra, T es menor que en full).

    evaluador = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="f1")

    cv = CrossValidator(
        estimator=pipeline, estimatorParamMaps=grid,
        evaluator=evaluador, numFolds=NUM_FOLDS,
        parallelism=2, seed=SEED)                              # paralelismo acotado al clúster base

    # ------------------------------------------------------------------
    # AJUSTE DEL CROSSVALIDATOR SOBRE LA MUESTRA
    # ------------------------------------------------------------------
    logger.info("Ejecutando CrossValidator sobre la muestra...")
    t0 = time.time()
    cv_model = cv.fit(df_muestra)
    t_cv = time.time() - t0
    logger.info("CrossValidator completado en %.2f s", t_cv)

    # ------------------------------------------------------------------
    # (e) MEJORES HIPERPARÁMETROS
    # ------------------------------------------------------------------
    mejor_idx = int(np.argmax(cv_model.avgMetrics))            # índice de mejor F1 promedio
    mejor_map = cv_model.getEstimatorParamMaps()[mejor_idx]    # ParamMap ganador
    mejores = {}                                               # extraer por nombre de parámetro
    for param, valor in mejor_map.items():
        mejores[param.name] = valor
    logger.info("Mejores hiperparámetros: numTrees=%s, maxDepth=%s, fss=%s (F1_cv=%.4f)",
                mejores.get("numTrees"), mejores.get("maxDepth"),
                mejores.get("featureSubsetStrategy"), cv_model.avgMetrics[mejor_idx])

    # ------------------------------------------------------------------
    # (f) RE-ENTRENAMIENTO SOBRE EL ENTRENAMIENTO COMPLETO
    # ------------------------------------------------------------------
    rf_opt = RandomForestClassifier(
        labelCol="label_index", featuresCol="scaled_features", weightCol="class_weight",
        seed=SEED,
        numTrees=int(mejores.get("numTrees", 100)),
        maxDepth=int(mejores.get("maxDepth", 10)),
        featureSubsetStrategy=str(mejores.get("featureSubsetStrategy", "sqrt")))
    pipeline_opt = Pipeline(stages=[rf_opt])

    logger.info("Re-entrenando el mejor modelo sobre el entrenamiento COMPLETO...")
    t0 = time.time()
    model_opt = pipeline_opt.fit(df_train)
    t_opt = time.time() - t0
    logger.info("Re-entrenamiento completado en %.2f s", t_opt)

    # ------------------------------------------------------------------
    # (g) RE-EVALUACIÓN SOBRE EL CONJUNTO DE PRUEBA
    # ------------------------------------------------------------------
    try:
        df_test = spark.read.parquet(TEST_PROCESSED)
        logger.info("Parquet de prueba cargado desde %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR al leer el Parquet de prueba (%s): %s", TEST_PROCESSED, e)
        raise

    df_pred_opt = model_opt.transform(df_test)
    ev = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction")
    f1_opt = float(ev.setMetricName("f1").evaluate(df_pred_opt))
    acc_opt = float(ev.setMetricName("accuracy").evaluate(df_pred_opt))
    logger.info("TEST optimizado -> F1_weighted=%.4f | Accuracy=%.4f", f1_opt, acc_opt)

    # ------------------------------------------------------------------
    # (h) GUARDADO DEL MEJOR MODELO + REPORTE COMPARATIVO
    # ------------------------------------------------------------------
    try:
        model_opt.write().overwrite().save(MODEL_OPT_OUT)       # serializar modelo optimizado
        logger.info("Modelo optimizado guardado en %s", MODEL_OPT_OUT)
    except Exception as e:
        logger.error("ERROR al guardar el modelo optimizado (%s): %s", MODEL_OPT_OUT, e)
        raise

    # Métricas del modelo base (si existen).
    train_base = leer_json_s3(TRAIN_METRICS) or {}
    test_base  = leer_json_s3(TEST_METRICS_OUT) or {}
    f1_base  = test_base.get("multiclase", {}).get("f1_weighted")
    acc_base = test_base.get("multiclase", {}).get("accuracy")
    t_base   = train_base.get("training_time_seconds")

    reporte = {
        "modelo_base": {
            "hiperparametros": {
                "numTrees": train_base.get("num_trees", 100),
                "maxDepth": train_base.get("max_depth", 10),
                "featureSubsetStrategy": train_base.get("feature_subset_strategy", "sqrt"),
            },
            "f1_weighted": f1_base,
            "accuracy": acc_base,
            "tiempo_entreno_s": t_base,
        },
        "modelo_optimizado": {
            "hiperparametros": {
                "numTrees": int(mejores.get("numTrees", 100)),
                "maxDepth": int(mejores.get("maxDepth", 10)),
                "featureSubsetStrategy": str(mejores.get("featureSubsetStrategy", "sqrt")),
            },
            "f1_weighted": f1_opt,
            "accuracy": acc_opt,
            "tiempo_entreno_s": round(t_opt, 4),
            "f1_cv_promedio": float(cv_model.avgMetrics[mejor_idx]),
            "tiempo_crossvalidator_s": round(t_cv, 4),
        },
        "config_busqueda": {
            "numTrees": [50, 100, 200], "maxDepth": [5, 10, 15],
            "featureSubsetStrategy": ["sqrt", "log2"],
            "combinaciones": len(grid), "folds": NUM_FOLDS,
            "muestra_fraccion": SAMPLE_FRACTION,
        },
    }
    try:
        escribir_json_s3(reporte, OPT_REPORT_OUT)
        logger.info("Reporte de optimización guardado en %s", OPT_REPORT_OUT)
    except Exception as e:
        logger.error("ERROR al escribir el reporte (%s): %s", OPT_REPORT_OUT, e)
        raise

    # ------------------------------------------------------------------
    # (i) TABLA COMPARATIVA EN CONSOLA
    # ------------------------------------------------------------------
    def _fmt(x):
        """Formatea un número o devuelve 'N/D' si es None."""
        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/D"

    logger.info("| Configuración     | F1 weighted | Accuracy | T. entrenamiento |")
    logger.info("|-------------------|-------------|----------|------------------|")
    logger.info("| Modelo base       |    %-6s   |  %-6s  |     %8s s     |",
                _fmt(f1_base), _fmt(acc_base), _fmt(t_base))
    logger.info("| Modelo optimizado |    %-6s   |  %-6s  |     %8s s     |",
                _fmt(f1_opt), _fmt(acc_opt), _fmt(round(t_opt, 3)))

    df_muestra.unpersist()
    logger.info("BLOQUE 3 completado correctamente.")


# ==============================================================
# === MAIN: ORQUESTACIÓN COMPLETA DEL PIPELINE RF ===
# ==============================================================
def main():
    """Ejecuta en secuencia los tres bloques del pipeline RF con
    una única SparkSession compartida."""
    spark = construir_spark()
    logger.info("SparkSession 'ProyectoAnomalias-RandomForest' iniciada.")
    logger.info("Iniciando pipeline completo de Random Forest (3 bloques).")

    try:
        bloque1_entrenar_rf(spark)      # Entrenamiento del modelo base
        bloque2_evaluar_rf(spark)       # Evaluación batch sobre prueba
        bloque3_optimizar_rf(spark)     # Optimización de hiperparámetros
    except Exception as e:
        logger.error("ERROR en el pipeline RF: %s", e)
        spark.stop()
        raise

    spark.stop()
    logger.info("Pipeline Random Forest finalizado correctamente (3/3 bloques).")


if __name__ == "__main__":
    main()
