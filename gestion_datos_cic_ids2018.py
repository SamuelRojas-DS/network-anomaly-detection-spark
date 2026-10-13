#!/usr/bin/env python3


# === IMPORTS ===
import re
import json
import logging
from functools import reduce
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("GestionDatos2018")



# Bucket público del dataset CIC-IDS2018 (solo lectura, acceso anónimo).
BUCKET_2018  = "s3://cse-cic-ids2018"
PREFIJO_2018 = "Processed Traffic Data for ML Algorithms"

# Archivos para la demostración de streaming (Feb 20-21).
ARCHIVOS_DEMO = [
    f"{BUCKET_2018}/{PREFIJO_2018}/02-20-2018.csv",
    f"{BUCKET_2018}/{PREFIJO_2018}/02-21-2018.csv",
]

# Bucket propio del proyecto (lectura/escritura con rol del clúster EMR).
BUCKET_PROYECTO  = "s3://proyecto-anomalias-emr"
PREPROC_PATH     = f"{BUCKET_PROYECTO}/models/random_forest_pipeline/preprocessors"
FEATURES_META_IN = f"{PREPROC_PATH}/features_meta.json"

# Salida del Parquet homologado (input para tcp_sender.py).
STREAMING_OUT = f"{BUCKET_PROYECTO}/data/cic-ids2018/streaming/processed/"

# Columna de etiqueta (nombre normalizado, igual que en gestion_datos.py).
LABEL_COL = "label"

# Columnas que NUNCA entran en el vector de features.
EXCLUIR_FEATURES = {
    "label", "_source_file", "_split",
    "label_index", "class_weight", "features", "scaled_features",
    "timestamp",   # columna extra de IDS2018
    "flow_id",     # metadato de IDS2018
    "src_ip", "src_port", "dst_ip", "dst_port",  # metadatos de red
}

# Configuración Spark (igual que gestion_datos.py para consistencia).
EXECUTOR_INSTANCES = 2
SHUFFLE_PARTITIONS = 16

# ==================================================================
# DICCIONARIO DE HOMOLOGACIÓN DE COLUMNAS
# Mapea nombres de IDS2018 (ya normalizados con normalizar_nombre)
# a sus equivalentes en IDS2017 (también normalizados).
# Solo se listan las discrepancias — columnas iguales tras normalizar
# no necesitan entrada aquí.
# CORRECCIÓN: eliminadas entradas duplicadas de totlen_fwd/bwd_pkts.
# ==================================================================
HOMOLOGACION = {
    # Tasas de flujo
    # NOTA: normalizar_nombre() elimina '/' sin insertar '_' (p.ej. "Flow Bytes/s"
    # -> "flow_bytess"), por eso el destino NO lleva guion bajo antes de la "s".
    "flow_byts_s":      "flow_bytess",
    "flow_pkts_s":      "flow_packetss",
    "fwd_pkts_s":       "fwd_packetss",
    "bwd_pkts_s":       "bwd_packetss",
    # Longitud de paquetes — forward
    "fwd_pkt_len_max":  "fwd_packet_length_max",
    "fwd_pkt_len_min":  "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std":  "fwd_packet_length_std",
    # Longitud de paquetes — backward
    "bwd_pkt_len_max":  "bwd_packet_length_max",
    "bwd_pkt_len_min":  "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std":  "bwd_packet_length_std",
    # Longitud de paquetes — globales
    "pkt_len_max":      "max_packet_length",
    "pkt_len_min":      "min_packet_length",
    "pkt_len_mean":     "packet_length_mean",
    "pkt_len_std":      "packet_length_std",
    "pkt_len_var":      "packet_length_variance",
    "pkt_size_avg":     "average_packet_size",
    # Segmento
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_seg_size_min": "min_seg_size_forward",
    # Bulk
    # NOTA: igual que arriba — "Fwd Avg Bytes/Bulk" -> "fwd_avg_bytesbulk" (sin "_"
    # entre "bytes" y "bulk"). "Fwd Avg Bulk Rate" no tiene '/' así que sí conserva "_".
    "fwd_byts_b_avg":   "fwd_avg_bytesbulk",
    "fwd_pkts_b_avg":   "fwd_avg_packetsbulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg":   "bwd_avg_bytesbulk",
    "bwd_pkts_b_avg":   "bwd_avg_packetsbulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    # Flags
    "fin_flag_cnt":     "fin_flag_count",
    "syn_flag_cnt":     "syn_flag_count",
    "rst_flag_cnt":     "rst_flag_count",
    "psh_flag_cnt":     "psh_flag_count",
    "ack_flag_cnt":     "ack_flag_count",
    "urg_flag_cnt":     "urg_flag_count",
    "ece_flag_cnt":     "ece_flag_count",
    # Volumen de paquetes y bytes
    "tot_fwd_pkts":     "total_fwd_packets",
    "tot_bwd_pkts":     "total_backward_packets",
    "totlen_fwd_pkts":  "total_length_of_fwd_packets",   # sin duplicado
    "totlen_bwd_pkts":  "total_length_of_bwd_packets",   # sin duplicado
    # Subflow
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    # TCP init
    "init_bwd_win_byts": "init_win_bytes_backward",
    "init_fwd_win_byts": "init_win_bytes_forward",
    # Activos / segmentos
    "fwd_act_data_pkts": "act_data_pkt_fwd",
}


# ==================================================================
# === HELPERS S3 ===================================================
# ==================================================================

def _parse_s3(ruta_s3):
    """Descompone s3://bucket/clave en (bucket, clave)."""
    p = urlparse(ruta_s3)
    return p.netloc, p.path.lstrip("/")


def leer_features_meta():
    """
    Carga la lista de columnas de features del pipeline 2017 desde S3.
    Devuelve la lista si existe, o None si no se encontró.
    """
    bucket, clave = _parse_s3(FEATURES_META_IN)
    cliente = boto3.client("s3")
    try:
        resp = cliente.get_object(Bucket=bucket, Key=clave)
        datos = json.loads(resp["Body"].read().decode("utf-8"))
        feature_cols = datos.get("feature_cols", None)
        if feature_cols:
            logger.info(
                "features_meta.json cargado: %d columnas de features.",
                len(feature_cols),
            )
        return feature_cols
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning(
                "features_meta.json NO encontrado en %s. "
                "Se inferirán columnas desde los datos de IDS2018. "
                "Ver NOTA_METADATA en el docstring de este script.",
                FEATURES_META_IN,
            )
            return None
        raise


# ==================================================================
# === SPARK SESSION ================================================
# ==================================================================

def construir_spark():
    """
    SparkSession configurada para EMR.

    CORRECCIÓN: las credenciales anónimas se aplican SOLO al bucket
    público de CIC-IDS2018 usando configuración por bucket
    (spark.hadoop.fs.s3a.bucket.<nombre>.aws.credentials.provider).
    El bucket del proyecto usa las credenciales del rol IAM del EMR
    (comportamiento por defecto — sin configuración adicional).
    """
    spark = (
        SparkSession.builder
        .appName("ProyectoAnomalias-GestionDatos2018")
        # Recursos — igual que gestion_datos.py
        .config("spark.executor.cores",           "2")
        .config("spark.executor.memory",          "6g")
        .config("spark.executor.memoryOverhead",  "1024")
        .config("spark.driver.cores",             "2")
        .config("spark.driver.memory",            "6g")
        .config("spark.executor.instances",       str(EXECUTOR_INSTANCES))
        .config("spark.default.parallelism",      str(SHUFFLE_PARTITIONS))
        .config("spark.sql.shuffle.partitions",   str(SHUFFLE_PARTITIONS))
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "256m")
        .config("spark.kryo.registrationRequired", "false")
        .config("spark.memory.fraction",          "0.75")
        .config("spark.memory.storageFraction",   "0.40")
        .config("spark.task.maxFailures",         "4")
        .config("spark.speculation",              "true")
        .config("spark.speculation.multiplier",   "1.5")
        .config("spark.hadoop.fs.s3a.fast.upload","true")
        # CORRECCIÓN: acceso anónimo solo para el bucket público de IDS2018.
        # El bucket "proyecto-anomalias-emr" usa el rol IAM del EMR (default).
        .config(
            "spark.hadoop.fs.s3a.bucket.cse-cic-ids2018"
            ".aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ==================================================================
# === NORMALIZACIÓN DE NOMBRES DE COLUMNAS =========================
# ==================================================================

def normalizar_nombre(nombre):
    """
    Normaliza un nombre de columna de CICFlowMeter aplicando las
    mismas reglas que gestion_datos.py:
        1. strip() de espacios externos
        2. espacios internos → guion bajo
        3. conversión a minúsculas
        4. eliminación de caracteres no alfanuméricos (conserva '_')
    """
    n = nombre.strip()
    n = re.sub(r"\s+", "_", n)
    n = n.lower()
    n = re.sub(r"[^0-9a-z_]", "", n)
    return n


# ==================================================================
# === CARGA DE CSV =================================================
# ==================================================================

def cargar_y_normalizar(spark, rutas):
    """
    Carga los CSV de CIC-IDS2018 desde el bucket público S3 y
    normaliza inmediatamente los nombres de columna.
    """
    if not rutas:
        raise ValueError("Lista de rutas CSV vacía. Verificar ARCHIVOS_DEMO.")

    dataframes = []
    for ruta in rutas:
        logger.info("Cargando: %s", ruta)
        try:
            df = (
                spark.read
                .option("header",       True)
                .option("inferSchema",  False)    # todas como string
                .option("mode",         "PERMISSIVE")
                .option("encoding",     "UTF-8")
                .csv(ruta)
            )
        except Exception as e:
            logger.warning(
                "Fallo con UTF-8 en %s (%s). Reintentando con latin-1.", ruta, e)
            df = (
                spark.read
                .option("header",      True)
                .option("inferSchema", False)
                .option("mode",        "PERMISSIVE")
                .option("encoding",    "ISO-8859-1")
                .csv(ruta)
            )

        # Normalizar nombres inmediatamente tras la carga.
        for original in df.columns:
            nuevo = normalizar_nombre(original)
            if original != nuevo:
                df = df.withColumnRenamed(original, nuevo)

        # Trazabilidad por fila.
        df = df.withColumn("_source_file", F.input_file_name())

        logger.info(
            "  %s: %d columnas cargadas.",
            ruta.split("/")[-1], len(df.columns),
        )
        dataframes.append(df)

    df_union = reduce(
        lambda a, b: a.unionByName(b, allowMissingColumns=True),
        dataframes,
    )
    logger.info(
        "Dataset 2018 combinado: %d columnas tras unionByName.",
        len(df_union.columns),
    )
    return df_union


# ==================================================================
# === HOMOLOGACIÓN DE COLUMNAS =====================================
# ==================================================================

def homologar_columnas(df):
    """
    Renombra columnas de IDS2018 (normalizadas) a sus equivalentes
    en IDS2017. Solo renombra si la columna origen existe Y la
    columna destino NO existe (evita colisiones).
    """
    renombrados = []
    for nombre_2018, nombre_2017 in HOMOLOGACION.items():
        if nombre_2018 in df.columns and nombre_2017 not in df.columns:
            df = df.withColumnRenamed(nombre_2018, nombre_2017)
            renombrados.append(f"{nombre_2018} → {nombre_2017}")

    logger.info(
        "Homologación: %d columnas renombradas.", len(renombrados),
    )
    if renombrados:
        logger.debug("Detalle: %s", renombrados)
    return df


# ==================================================================
# === CAST NUMÉRICO ================================================
# ==================================================================

def castear_numericas(df, columnas_numericas):
    """
    Castea las columnas de features a DoubleType conservando solo
    las columnas necesarias para reducir el shuffle posterior.
    """
    cols_conservar = (
        list(columnas_numericas)
        + [c for c in ["label", "_source_file"] if c in df.columns]
    )
    cols_conservar = [c for c in cols_conservar if c in df.columns]
    df = df.select(*cols_conservar)

    set_num = set(columnas_numericas)
    df = df.select(*[
        F.col(c).cast(DoubleType()).alias(c)
        if c in set_num else F.col(c)
        for c in df.columns
    ])
    return df


# ==================================================================
# === LIMPIEZA =====================================================
# ==================================================================

def limpiar(df, columnas_numericas):
    """
    Aplica la misma limpieza que gestion_datos.py:
        Paso 1: eliminar nulos en columnas clave.
        Paso 2: reemplazar ±Inf por NaN y eliminar esas filas.
        Paso 3: eliminar duplicados exactos.

    CORRECCIÓN: se agrega df.cache() antes de los count() para
    evitar tres full-scans redundantes sobre el mismo DataFrame.
    """
    df = df.withColumn("_split", F.lit("streaming_demo"))
    columnas_clave = columnas_numericas + [LABEL_COL]

    # CORRECCIÓN: cachear antes de los count() encadenados.
    df.cache()

    n0 = df.count()
    logger.info("[IDS2018] Registros iniciales: %d", n0)

    # Paso 1: nulos.
    df = df.dropna(how="any", subset=columnas_clave)
    n1 = df.count()
    logger.info("[IDS2018] Tras eliminar nulos: %d (eliminados=%d)", n1, n0 - n1)

    # Paso 2: infinitos → NaN → filtrar.
    set_num = set(columnas_numericas)
    df = df.select(*[
        F.when(
            (F.col(c) == float("inf")) | (F.col(c) == float("-inf")),
            float("nan"),
        ).otherwise(F.col(c)).alias(c)
        if c in set_num else F.col(c)
        for c in df.columns
    ])
    cond_nan = reduce(
        lambda a, b: a | b,
        [F.isnan(F.col(c)) for c in columnas_numericas],
    )
    df = df.filter(~cond_nan)
    n2 = df.count()
    logger.info(
        "[IDS2018] Tras eliminar infinitos/NaN: %d (eliminados=%d)", n2, n1 - n2,
    )
    if (n1 - n2) > (n1 * 0.30):
        logger.warning(
            "[IDS2018] Se eliminó >30%% de registros por infinitos/NaN. "
            "Comportamiento conocido en CIC-IDS2018. Continuar."
        )

    # Paso 3: duplicados.
    df = df.dropDuplicates()
    n3 = df.count()
    logger.info(
        "[IDS2018] Tras eliminar duplicados: %d (eliminados=%d)", n3, n2 - n3,
    )

    df.unpersist()
    return df


# ==================================================================
# === ALINEACIÓN CON EL ESQUEMA 2017 ===============================
# ==================================================================

def alinear_con_esquema_2017(df, feature_cols_2017):
    """
    Garantiza que el DataFrame 2018 tenga exactamente las columnas
    que el PipelineModel 2017 espera.

    - Columnas presentes: se usan directamente.
    - Columnas ausentes: se imputan con 0.0 (sin actividad).
    - Columnas extra de 2018: se ignoran.
    """
    presentes          = set(df.columns)
    columnas_añadidas  = []

    for col in feature_cols_2017:
        if col not in presentes:
            df = df.withColumn(col, F.lit(0.0).cast(DoubleType()))
            columnas_añadidas.append(col)

    if columnas_añadidas:
        logger.warning(
            "%d columnas esperadas por el pipeline 2017 ausentes en IDS2018 "
            "→ imputadas con 0.0: %s",
            len(columnas_añadidas), columnas_añadidas,
        )
    else:
        logger.info(
            "Alineación perfecta: todas las columnas 2017 presentes en IDS2018."
        )

    excedentes = [
        c for c in presentes
        if c not in set(feature_cols_2017) and c not in EXCLUIR_FEATURES
    ]
    if excedentes:
        logger.info(
            "%d columnas de IDS2018 ignoradas (no están en esquema 2017): %s",
            len(excedentes), excedentes[:10],
        )

    return df


# ==================================================================
# === DETECCIÓN DE COLUMNAS NUMÉRICAS (fallback) ===================
# ==================================================================

def detectar_columnas_numericas(df, excluir):
    """
    Detecta columnas 100% casteables a DoubleType.
    Solo se usa como fallback si features_meta.json no existe.
    Idéntica a la función en gestion_datos.py.
    """
    columnas = [c for c in df.columns if c not in excluir]
    exprs = []
    for c in columnas:
        exprs.append(
            F.count(F.when(F.col(c).isNotNull(), True)).alias(f"nn__{c}"))
        exprs.append(
            F.count(F.when(F.col(c).cast("double").isNotNull(),
                           True)).alias(f"cd__{c}"))
    fila     = df.agg(*exprs).first()
    numericas = [
        c for c in columnas
        if fila[f"nn__{c}"] > 0 and fila[f"nn__{c}"] == fila[f"cd__{c}"]
    ]
    logger.info(
        "Columnas numéricas detectadas (fallback): %d de %d candidatas.",
        len(numericas), len(columnas),
    )
    return numericas


# ==================================================================
# === MAIN =========================================================
# ==================================================================

def main():
    """
    Orquesta la preparación del dataset CIC-IDS2018 para la demo.

    Pipeline:
        1. Cargar features_meta.json desde S3 (lista de columnas 2017).
        2. Cargar y normalizar CSV de IDS2018 desde bucket público.
        3. Homologar columnas IDS2018 → IDS2017.
        4. Determinar columnas de features a usar.
        5. Alinear con el esquema 2017 (imputar ausentes con 0.0).
        6. Cast a DoubleType + conservar solo columnas necesarias.
        7. Limpieza (nulos → infinitos → duplicados).
        8. Guardar Parquet en bucket del proyecto.
        9. Resumen de diagnóstico.
    """
    spark = construir_spark()
    logger.info("SparkSession 'ProyectoAnomalias-GestionDatos2018' iniciada.")

    # ── 1. Cargar lista de features del pipeline 2017 ─────────────────────────
    feature_cols_2017 = leer_features_meta()

    # ── 2. Cargar y normalizar CSV desde bucket público ───────────────────────
    try:
        df = cargar_y_normalizar(spark, ARCHIVOS_DEMO)
    except Exception as e:
        logger.error("ERROR al cargar CSV de IDS2018: %s", e)
        spark.stop()
        raise

    if LABEL_COL not in df.columns:
        logger.error(
            "Columna '%s' no encontrada tras normalizar. "
            "Columnas disponibles: %s", LABEL_COL, df.columns,
        )
        spark.stop()
        raise KeyError(LABEL_COL)

    # ── 3. Homologar columnas IDS2018 → IDS2017 ───────────────────────────────
    df = homologar_columnas(df)

    # ── 4. Determinar columnas de features ────────────────────────────────────
    if feature_cols_2017 is not None:
        logger.info(
            "Usando %d columnas de features del pipeline 2017 (features_meta.json).",
            len(feature_cols_2017),
        )
        columnas_numericas = feature_cols_2017
    else:
        logger.warning(
            "FALLBACK: infiriendo columnas numéricas desde IDS2018. "
            "Guardar features_meta.json en gestion_datos.py (ver NOTA_METADATA)."
        )
        columnas_numericas = sorted(
            detectar_columnas_numericas(df, EXCLUIR_FEATURES)
        )

    # ── 5. Alinear con esquema 2017 ───────────────────────────────────────────
    df = alinear_con_esquema_2017(df, columnas_numericas)

    # ── 6. Cast a DoubleType ──────────────────────────────────────────────────
    columnas_a_castear = [c for c in columnas_numericas if c in df.columns]
    df = castear_numericas(df, columnas_a_castear)

    # ── 7. Limpieza ───────────────────────────────────────────────────────────
    df = limpiar(df, columnas_a_castear)

    # ── 8. Guardar Parquet en S3 ──────────────────────────────────────────────
    # El Parquet contiene las 77 columnas numéricas + label + _source_file + _split.
    # NO contiene vectores (features, scaled_features) — eso lo hace el PipelineModel
    # dentro de streaming_demo_v2.py durante la demo.
    cols_salida = (
        columnas_a_castear
        + [c for c in ["label", "_source_file", "_split"] if c in df.columns]
    )
    cols_salida = [c for c in cols_salida if c in df.columns]
    df_out = df.select(*cols_salida)

    try:
        df_out.write.mode("overwrite").parquet(STREAMING_OUT)
        logger.info("Parquet homologado guardado en: %s", STREAMING_OUT)
    except Exception as e:
        logger.error("ERROR al guardar Parquet en %s: %s", STREAMING_OUT, e)
        spark.stop()
        raise

    # ── 9. Resumen de diagnóstico ─────────────────────────────────────────────
    n_final = df_out.count()
    distribucion = {
        r[LABEL_COL]: r["count"]
        for r in df_out.groupBy(LABEL_COL).count().collect()
    }
    logger.info(
        "RESUMEN FINAL | Registros: %d | Features: %d | Distribución: %s",
        n_final, len(columnas_a_castear), distribucion,
    )
    logger.info(
        "Parquet listo para streaming. tcp_sender.py debe leer desde: %s",
        STREAMING_OUT,
    )

    spark.stop()
    logger.info("gestion_datos_cic_ids2018.py completado correctamente.")


if __name__ == "__main__":
    main()
