#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gestion_datos.py
==================================================================
PROPÓSITO:
    Ingerir, limpiar, codificar, ensamblar, escalar y ponderar el
    dataset CIC-IDS2017 para alimentar TODOS los pipelines del proyecto
    (Random Forest y KMeans). Es el PRIMER script de la cadena: produce
    los Parquet procesados y los preprocesadores (StringIndexer +
    StandardScaler) que el resto de scripts reutilizan.

DESCARGA AUTOMÁTICA:
    Si los CSV aún no existen en S3, este script los descarga
    directamente desde Kaggle usando su API y los sube al bucket
    del proyecto. No se requiere tener el dataset en la máquina local.

    Credenciales de Kaggle (una de las dos opciones):
      1. Variables de entorno:    KAGGLE_USERNAME  y  KAGGLE_KEY
      2. AWS Secrets Manager:     secreto "kaggle/credentials"
                                  con campos {"username": ..., "key": ...}

    Dataset en Kaggle: chethuhn/network-intrusion-dataset
    (CSV nombrados por día: Monday-WorkingHours.pcap_ISCX.csv, Tuesday-...,
     Wednesday-..., los dos de Thursday y los tres de Friday — compatibles
     con la lógica de reparto por día de este script.
     Ver: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

INPUTS (rutas S3 de entrada):
    - s3://proyecto-anomalias-emr/data/cic-ids2017/train/   (CSV lunes/miércoles/viernes)
    - s3://proyecto-anomalias-emr/data/cic-ids2017/test/    (CSV martes/jueves)

OUTPUTS (rutas S3 de salida):
    - .../train/processed/                            (Parquet entrenamiento)
    - .../test/processed/                             (Parquet prueba)
    - .../models/random_forest_pipeline/preprocessors/indexer/  (StringIndexerModel)
    - .../models/random_forest_pipeline/preprocessors/scaler/   (StandardScalerModel)
    - .../models/random_forest_pipeline/preprocessors/features_meta.json
      Lista JSON con los nombres de columnas del VectorAssembler en orden
      alfabético determinístico. Requerido por gestion_datos_cic_ids2018.py
      para alinear el esquema de features de IDS2018 con el de IDS2017.

DEPENDENCIAS (ejecutar ANTES):
    - Ninguna. Este script DEBE ejecutarse el primero de todo el proyecto.

ORDEN DE EJECUCIÓN DEL PROYECTO:
    1. gestion_datos.py                  ← este script
    2. pipeline_random_forest.py
    3. pipeline_kmeans.py
    4. gestion_datos_cic_ids2018.py      (prepara dataset de demo streaming)
==================================================================
"""

# === IMPORTS ===
import os                                               # variables de entorno y paths locales
import re                                               # normalización de nombres de columnas
import json                                             # serialización auxiliar
import shutil                                           # limpieza de directorio temporal
import logging                                          # logging estructurado (NO print)
import subprocess                                       # instalación de kaggle y descarga CLI
import tempfile                                         # directorio temporal para descarga
from functools import reduce                            # union/condiciones acumuladas
from urllib.parse import urlparse                       # parseo de rutas s3://

import boto3                                            # listado de CSV en S3 y Secrets Manager

from pyspark.sql import SparkSession                    # punto de entrada de Spark
from pyspark import StorageLevel                        # niveles de persistencia (cache)
from pyspark.sql import functions as F                  # funciones SQL (alias F)
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler


# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("GestionDatos")


# === CONFIGURACIÓN ===
BUCKET        = "s3://proyecto-anomalias-emr"                   # bucket raíz del proyecto
TRAIN_RAW     = f"{BUCKET}/data/cic-ids2017/train/"             # CSV de entrenamiento
TEST_RAW      = f"{BUCKET}/data/cic-ids2017/test/"              # CSV de prueba
MODELS_PATH   = f"{BUCKET}/models/random_forest_pipeline/"     # modelos/preprocesadores
RESULTS_PATH  = f"{BUCKET}/results/batch_evaluation/"          # resultados
SCRIPTS_PATH  = f"{BUCKET}/scripts/"                           # scripts del proyecto

# Rutas derivadas.
TRAIN_PROCESSED   = f"{TRAIN_RAW}processed/"                   # salida Parquet entrenamiento
TEST_PROCESSED    = f"{TEST_RAW}processed/"                    # salida Parquet prueba
INDEXER_OUT       = f"{MODELS_PATH}preprocessors/indexer/"     # salida StringIndexer
SCALER_OUT        = f"{MODELS_PATH}preprocessors/scaler/"      # salida StandardScaler
FEATURES_META_OUT = f"{MODELS_PATH}preprocessors/features_meta.json"  # lista de features del VectorAssembler

EXECUTOR_INSTANCES = 2                                          # nº de executors (clúster base)
SHUFFLE_PARTITIONS = 16                                         # particiones de shuffle

# === CONFIGURACIÓN KAGGLE ===
# Slug del dataset en Kaggle (CSV nombrados por día, requeridos por _detectar_dia):
# https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
# Si usas otro dataset, asegúrate de que sus CSV incluyan el día en el nombre.
KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"

# Mapeo día-de-semana -> split (coincide con la convención de nombres de los CSV de CIC-IDS2017)
_DIAS_TRAIN = {"monday", "wednesday", "friday"}
_DIAS_TEST  = {"tuesday", "thursday"}


# === HELPERS S3 ===
def _parse_s3(ruta_s3):
    """Descompone s3://bucket/clave en (bucket, clave)."""
    p = urlparse(ruta_s3)
    return p.netloc, p.path.lstrip("/")


def listar_csv(prefijo_s3):
    """Lista todas las rutas s3:// de archivos .csv bajo un prefijo."""
    bucket, prefijo = _parse_s3(prefijo_s3)                    # bucket y prefijo
    cliente = boto3.client("s3")                               # cliente S3
    paginador = cliente.get_paginator("list_objects_v2")       # paginación de objetos
    rutas = []                                                 # acumulador de rutas
    for pagina in paginador.paginate(Bucket=bucket, Prefix=prefijo):
        for obj in pagina.get("Contents", []):                 # objetos de la página
            clave = obj["Key"]                                 # clave del objeto
            if clave.lower().endswith(".csv"):                 # solo CSV
                rutas.append(f"s3://{bucket}/{clave}")          # ruta completa
    return rutas


def construir_spark():
    """SparkSession con TODOS los parámetros del clúster vía .config()."""
    spark = (
        SparkSession.builder
        .appName("ProyectoAnomalias-GestionDatos")                                 # nombre exacto requerido
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
        .config("spark.kryo.registrationRequired", "false")                        # evita warnings de Kryo con MLlib
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


# =====================================================================
# DESCARGA AUTOMÁTICA DESDE KAGGLE
# =====================================================================

def _get_kaggle_credentials():
    """
    Configura las credenciales de Kaggle. Acepta, en orden de prioridad:
      1. KAGGLE_API_TOKEN  (token nuevo 'KGAT_...' por variable de entorno).
      2. KAGGLE_USERNAME + KAGGLE_KEY  (formato clásico de kaggle.json).
      3. ~/.kaggle/access_token  o  ~/.kaggle/kaggle.json  (el cliente los
         lee automáticamente, no hay que configurar nada).
      4. AWS Secrets Manager: secreto 'kaggle/credentials' con JSON
         {"api_token": "..."}  o  {"username": "...", "key": "..."}.

    Devuelve un dict que describe el método encontrado. El llamador se
    encarga de exportar al entorno lo que haga falta. Lanza RuntimeError
    si no hay credenciales por ninguna vía.
    """
    # 1. Token nuevo (KGAT_...) por variable de entorno.
    if os.environ.get("KAGGLE_API_TOKEN"):
        logger.info("Credenciales de Kaggle: KAGGLE_API_TOKEN (token nuevo) en el entorno.")
        return {"metodo": "api_token"}

    # 2. Usuario + key clásicos por variables de entorno.
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if username and key:
        logger.info("Credenciales de Kaggle: KAGGLE_USERNAME/KAGGLE_KEY en el entorno.")
        return {"metodo": "user_key", "username": username, "key": key}

    # 3. Archivos que el cliente kaggle lee solo.
    home = os.path.expanduser("~")
    if (os.path.exists(os.path.join(home, ".kaggle", "access_token"))
            or os.path.exists(os.path.join(home, ".kaggle", "kaggle.json"))):
        logger.info("Credenciales de Kaggle: archivo en ~/.kaggle/ (lectura automática del cliente).")
        return {"metodo": "archivo"}

    # 4. Fallback: AWS Secrets Manager.
    logger.info(
        "Sin credenciales en entorno ni en ~/.kaggle/. "
        "Intentando Secrets Manager (kaggle/credentials)..."
    )
    try:
        sm = boto3.client("secretsmanager")
        secreto = sm.get_secret_value(SecretId="kaggle/credentials")
        creds = json.loads(secreto["SecretString"])
        logger.info("Credenciales de Kaggle obtenidas desde Secrets Manager.")
        if creds.get("api_token"):
            return {"metodo": "api_token", "token": creds["api_token"]}
        return {"metodo": "user_key", "username": creds["username"], "key": creds["key"]}
    except Exception as exc:
        raise RuntimeError(
            "No se encontraron credenciales de Kaggle.\n"
            "Opciones disponibles:\n"
            "  1. export KAGGLE_API_TOKEN=KGAT_...          (token nuevo)\n"
            "  2. export KAGGLE_USERNAME=...  y  KAGGLE_KEY=...\n"
            "  3. archivo ~/.kaggle/access_token  o  ~/.kaggle/kaggle.json\n"
            "  4. AWS Secrets Manager: secreto 'kaggle/credentials'"
        ) from exc


def _detectar_dia(nombre_archivo):
    """
    Detecta el día de la semana a partir del nombre de un CSV de CIC-IDS2017.
    Los archivos tienen nombres como:
      Monday-WorkingHours.pcap_ISCX.csv
      Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Devuelve el día en minúsculas o None si no se reconoce.
    """
    nombre = nombre_archivo.lower()
    for dia in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if dia in nombre:
            return dia
    return None


def _listar_csvs_local(directorio):
    """Lista recursivamente todos los .csv dentro de un directorio local."""
    resultado = []
    for raiz, _, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.lower().endswith(".csv"):
                resultado.append(os.path.join(raiz, archivo))
    return resultado


def _tiene_csv_en_s3(cliente, bucket, prefijo):
    """Devuelve True si ya existe al menos un .csv bajo el prefijo dado."""
    resp = cliente.list_objects_v2(Bucket=bucket, Prefix=prefijo, MaxKeys=10)
    return any(
        obj["Key"].lower().endswith(".csv")
        for obj in resp.get("Contents", [])
    )


def descargar_y_subir_cic_ids2017():
    """
    Descarga el dataset CIC-IDS2017 desde Kaggle y sube los CSV a S3.

    Distribución por día de la semana:
      - Lunes, Miércoles, Viernes  ->  TRAIN_RAW  (entrenamiento)
      - Martes, Jueves             ->  TEST_RAW   (prueba)

    Idempotente: si ambos prefijos ya contienen CSV en S3, la función
    retorna inmediatamente sin volver a descargar ni subir nada.

    El paquete 'kaggle' se instala automáticamente en el nodo master
    si no está disponible (requiere acceso a pip/PyPI desde el EMR).
    """
    bucket, train_prefijo = _parse_s3(TRAIN_RAW)
    _,      test_prefijo  = _parse_s3(TEST_RAW)
    cliente_s3 = boto3.client("s3")

    # ── Comprobación de idempotencia ──────────────────────────────────
    train_ok = _tiene_csv_en_s3(cliente_s3, bucket, train_prefijo)
    test_ok  = _tiene_csv_en_s3(cliente_s3, bucket, test_prefijo)
    if train_ok and test_ok:
        logger.info(
            "CSV de CIC-IDS2017 ya presentes en S3 (train y test). "
            "Omitiendo descarga desde Kaggle."
        )
        return

    logger.info(
        "CSV ausentes en S3 (train_ok=%s, test_ok=%s). "
        "Iniciando descarga desde Kaggle [%s]...",
        train_ok, test_ok, KAGGLE_DATASET,
    )

    # ── Credenciales ──────────────────────────────────────────────────
    creds = _get_kaggle_credentials()
    if creds["metodo"] == "user_key":
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"]      = creds["key"]
    elif creds["metodo"] == "api_token" and creds.get("token"):
        # Token nuevo recuperado de Secrets Manager: exportarlo al entorno.
        os.environ["KAGGLE_API_TOKEN"] = creds["token"]
    # Para 'archivo' o 'api_token' ya presente en el entorno: nada que hacer,
    # el cliente kaggle lo lee automáticamente.

    # ── Instalar / actualizar el paquete kaggle ───────────────────────
    # El token nuevo (KGAT_..., variable KAGGLE_API_TOKEN) requiere kaggle>=1.7.
    # Se fuerza una versión reciente para evitar fallos de autenticación con
    # un paquete antiguo preinstalado en el nodo.
    logger.info("Instalando/actualizando el paquete 'kaggle' (>=1.7) via pip...")
    try:
        subprocess.check_call(
            ["pip", "install", "--quiet", "--upgrade", "kaggle>=1.7"],
            timeout=180,
        )
        logger.info("Paquete 'kaggle' listo.")
    except Exception as e:
        logger.warning("No se pudo actualizar 'kaggle' (%s). Se intentará con la versión presente.", e)

    # ── Descarga y descompresión en directorio temporal ───────────────
    tmpdir = tempfile.mkdtemp(prefix="cicids2017_")
    logger.info("Directorio temporal de descarga: %s", tmpdir)

    try:
        logger.info("Ejecutando: kaggle datasets download -d %s ...", KAGGLE_DATASET)
        subprocess.check_call(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", tmpdir,
                "--unzip",                              # descomprime el ZIP automáticamente
            ],
            timeout=3600,                               # 1 hora máx (dataset ~1 GB)
        )
        logger.info("Dataset descargado y descomprimido en %s.", tmpdir)

        # ── Subida de cada CSV al prefijo S3 correcto ─────────────────
        csvs = _listar_csvs_local(tmpdir)
        if not csvs:
            raise FileNotFoundError(
                f"No se encontraron archivos .csv en {tmpdir} tras la descarga. "
                f"Verifica el slug del dataset: {KAGGLE_DATASET}"
            )
        logger.info("CSV encontrados localmente: %d", len(csvs))

        n_train = n_test = n_omitidos = 0
        for ruta_local in csvs:
            nombre_archivo = os.path.basename(ruta_local)
            dia = _detectar_dia(nombre_archivo)

            if dia in _DIAS_TRAIN:
                prefijo_destino = train_prefijo
                n_train += 1
            elif dia in _DIAS_TEST:
                prefijo_destino = test_prefijo
                n_test += 1
            else:
                logger.warning(
                    "No se pudo clasificar '%s' por día — se omite.",
                    nombre_archivo,
                )
                n_omitidos += 1
                continue

            clave_s3 = prefijo_destino + nombre_archivo
            logger.info(
                "Subiendo %-60s -> s3://%s/%s",
                nombre_archivo, bucket, clave_s3,
            )
            cliente_s3.upload_file(ruta_local, bucket, clave_s3)

        logger.info(
            "Subida a S3 completa: %d train | %d test | %d omitidos.",
            n_train, n_test, n_omitidos,
        )

        # Validación mínima: al menos un CSV en cada split.
        if n_train == 0:
            raise RuntimeError(
                "No se subió ningún CSV de entrenamiento (lunes/miércoles/viernes). "
                f"Revisa los nombres de archivo en el dataset {KAGGLE_DATASET}."
            )
        if n_test == 0:
            raise RuntimeError(
                "No se subió ningún CSV de prueba (martes/jueves). "
                f"Revisa los nombres de archivo en el dataset {KAGGLE_DATASET}."
            )

    finally:
        # Siempre liberar espacio en el master, independientemente del resultado.
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("Directorio temporal eliminado: %s", tmpdir)


# =====================================================================
# FUNCIONES DE PROCESAMIENTO  (sin cambios respecto a la versión original)
# =====================================================================

def normalizar_nombre(nombre):
    """Normaliza un nombre de columna: strip, espacios->'_', minúsculas,
    y eliminación de caracteres no alfanuméricos (conservando '_')."""
    n = nombre.strip()                                # (1) quitar espacios externos
    n = re.sub(r"\s+", "_", n)                        # (2) espacios internos -> guion bajo
    n = n.lower()                                     # (3) minúsculas
    n = re.sub(r"[^0-9a-z_]", "", n)                  # (4) quitar no alfanuméricos
    return n


def cargar_y_normalizar(spark, prefijo_s3):
    """Carga TODOS los CSV del prefijo en UNA sola lectura (Spark los procesa
    en paralelo) y normaliza los nombres de columna una única vez. Mucho más
    eficiente que leer archivo por archivo y combinarlos con unionByName."""
    rutas = listar_csv(prefijo_s3)                    # lista explícita de .csv (excluye processed/)
    if not rutas:                                     # validar que haya archivos
        raise FileNotFoundError(f"No se encontraron CSV en {prefijo_s3}")
    logger.info("CSV encontrados en %s: %d (lectura en una sola pasada).", prefijo_s3, len(rutas))

    # Lectura SIN inferencia de esquema de la lista completa de archivos.
    # Pasar la lista evita leer la subcarpeta processed/ y evita el unionByName.
    df = (spark.read
          .option("header", True)
          .option("inferSchema", False)
          .csv(rutas))

    # (d) Normalización de nombres UNA sola vez sobre el esquema combinado.
    #
    # NOTA — columna duplicada conocida de CIC-IDS2017:
    # Los CSV originales de CICFlowMeter contienen la columna "Fwd Header Length"
    # DOS VECES (bug histórico de la herramienta, documentado en la literatura del
    # dataset). Tras normalizar_nombre() ambas colisionarían en el mismo nombre
    # "fwd_header_length", lo que vuelve ambigua cualquier referencia posterior por
    # nombre (VectorAssembler, select, etc.). Para evitarlo, si el nombre normalizado
    # ya se usó antes, se le añade como sufijo la posición (0-indexada) de la columna
    # en el esquema original — p.ej. "fwd_header_length34", "fwd_header_length55".
    nombres_originales = df.columns
    nombres_normalizados = [normalizar_nombre(c) for c in nombres_originales]
    vistos = {}
    for n in nombres_normalizados:
        vistos[n] = vistos.get(n, 0) + 1
    duplicados = {n for n, cnt in vistos.items() if cnt > 1}
    if duplicados:
        logger.warning(
            "Columnas duplicadas tras normalizar (conocido en CIC-IDS2017): %s. "
            "Se desambiguan añadiendo el índice de posición original.",
            duplicados,
        )

    nombres_finales = []
    for idx, (original, nuevo) in enumerate(zip(nombres_originales, nombres_normalizados)):
        if nuevo in duplicados:
            nuevo = f"{nuevo}{idx}"
        nombres_finales.append(nuevo)

    df = df.toDF(*nombres_finales)

    # (f) Trazabilidad: archivo de origen real de cada fila.
    df = df.withColumn("_source_file", F.input_file_name())
    return df


def detectar_columnas_numericas(df, excluir):
    """Detecta columnas casteables a double (numéricas reales). Lo hace sobre
    una MUESTRA acotada de filas: una columna es numérica o no lo es de forma
    consistente, así que no hace falta escanear los millones de filas — basta
    una muestra para clasificar el tipo y evitar un escaneo completo carísimo."""
    columnas = [c for c in df.columns if c not in excluir]     # candidatas
    muestra = df.select(*columnas).limit(20000)                # muestra suficiente para tipar
    exprs = []                                                 # agregaciones por columna
    for c in columnas:
        exprs.append(F.count(F.when(F.col(c).isNotNull(), True)).alias(f"nn__{c}"))            # no nulos
        exprs.append(F.count(F.when(F.col(c).cast("double").isNotNull(), True)).alias(f"cd__{c}"))  # casteables
    fila = muestra.agg(*exprs).first()                         # una sola acción sobre la muestra
    numericas = [c for c in columnas
                 if fila[f"nn__{c}"] > 0 and fila[f"nn__{c}"] == fila[f"cd__{c}"]]
    logger.info("Columnas numéricas detectadas: %d de %d candidatas",
                len(numericas), len(columnas))
    return numericas


def limpiar(df, columnas_numericas, etiqueta_split):
    """Limpieza en UNA sola pasada: un único filtro que descarta filas con
    label nulo, o con cualquier columna numérica nula / infinita / NaN; luego
    elimina duplicados. Reduce de 4 conteos (que recomputaban todo) a 2."""
    # (f) Columna de partición lógica.
    df = df.withColumn("_split", F.lit(etiqueta_split))
    n0 = df.count()                                            # conteo inicial (sobre cache)

    # Filtro combinado: label no nulo Y cada numérica válida (no null/inf/NaN).
    cond = F.col("label").isNotNull()
    for c in columnas_numericas:
        cond = (cond
                & F.col(c).isNotNull()
                & ~F.isnan(F.col(c))
                & (F.col(c) != float("inf"))
                & (F.col(c) != float("-inf")))
    df = df.filter(cond)                                       # una sola pasada de limpieza
    df = df.dropDuplicates()                                   # único shuffle pesado
    n1 = df.count()                                            # conteo final

    logger.info("[%s] Limpieza: %d -> %d registros (eliminados=%d: nulos/inf/NaN/duplicados)",
                etiqueta_split, n0, n1, n0 - n1)
    return df


def calcular_pesos_clase(df_train):
    """Calcula class_weight = (1/frecuencia_relativa) normalizado para que
    la suma de pesos por clase sea igual al número de clases."""
    conteos = df_train.groupBy("label_index").count().collect()   # conteos por clase (pequeño)
    total = sum(r["count"] for r in conteos)                       # total de registros
    num_clases = len(conteos)                                      # nº de clases

    pesos_brutos = {int(r["label_index"]): total / r["count"] for r in conteos}  # 1/freq = total/n
    suma_pesos = sum(pesos_brutos.values())                        # suma de pesos brutos
    factor = num_clases / suma_pesos                               # factor de normalización
    pesos_norm = {idx: w * factor for idx, w in pesos_brutos.items()}  # pesos normalizados

    logger.info("Pesos de clase normalizados (suma=%d clases): %s", num_clases,
                {k: round(v, 4) for k, v in pesos_norm.items()})

    # Construir DataFrame de mapeo y unir al de entrenamiento.
    spark = df_train.sql_ctx.sparkSession                          # sesión asociada
    filas = [(float(idx), float(w)) for idx, w in pesos_norm.items()]
    df_pesos = spark.createDataFrame(filas, ["label_index", "class_weight"])
    return df_train.join(df_pesos, on="label_index", how="left")


def guardar_features_meta(columnas_features):
    """
    Serializa la lista de columnas del VectorAssembler a un archivo JSON
    en S3 bajo la ruta FEATURES_META_OUT.

    Este archivo es requerido por gestion_datos_cic_ids2018.py para
    alinear exactamente las columnas de IDS2018 con las que el
    StandardScaler y el VectorAssembler fueron entrenados en IDS2017.
    Sin este archivo, el script 2018 opera en modo fallback (inferencia
    de columnas desde los datos), lo que puede producir discrepancias.

    El JSON tiene la forma:
        {"feature_cols": ["col_a", "col_b", ...]}
    donde las columnas están en orden alfabético determinístico, que
    es el mismo criterio usado por VectorAssembler en main().
    """
    bucket, clave = _parse_s3(FEATURES_META_OUT)                    # descompone la ruta S3
    cliente = boto3.client("s3")                                     # cliente S3 del rol EMR

    # Escribe el JSON en un archivo temporal local y lo sube a S3.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump({"feature_cols": list(columnas_features)}, tmp,
                  ensure_ascii=False, indent=2)
        ruta_tmp = tmp.name                                          # guarda la ruta para después

    try:
        cliente.upload_file(ruta_tmp, bucket, clave)                # sube a S3
        logger.info(
            "features_meta.json guardado en %s (%d columnas).",
            FEATURES_META_OUT, len(columnas_features),
        )
    except Exception as e:
        logger.error(
            "ERROR al guardar features_meta.json en %s: %s",
            FEATURES_META_OUT, e,
        )
        raise
    finally:
        os.remove(ruta_tmp)                                         # limpia el archivo temporal


def main():
    """Orquesta la ingesta y preparación completa del dataset."""

    # ------------------------------------------------------------------
    # PASO 0: DESCARGA AUTOMÁTICA DESDE KAGGLE (si los CSV no están en S3)
    # ------------------------------------------------------------------
    try:
        descargar_y_subir_cic_ids2017()
    except Exception as e:
        logger.error("ERROR en la descarga/subida desde Kaggle: %s", e)
        raise

    spark = construir_spark()
    logger.info("SparkSession 'ProyectoAnomalias-GestionDatos' iniciada.")

    # ------------------------------------------------------------------
    # (c)(d) CARGA + NORMALIZACIÓN DE ENTRENAMIENTO Y PRUEBA
    # ------------------------------------------------------------------
    try:
        df_train = cargar_y_normalizar(spark, TRAIN_RAW)        # CSV de entrenamiento
        df_test  = cargar_y_normalizar(spark, TEST_RAW)         # CSV de prueba
        logger.info("CSV de entrenamiento y prueba cargados y normalizados.")
    except Exception as e:
        logger.error("ERROR al cargar/normalizar CSV: %s", e)
        spark.stop()
        raise

    # ------------------------------------------------------------------
    # DETECCIÓN DE COLUMNAS NUMÉRICAS (sobre entrenamiento) + CAST
    # ------------------------------------------------------------------
    # Se excluyen explícitamente label y columnas de trazabilidad.
    excluir = {"label", "_source_file"}
    columnas_numericas = detectar_columnas_numericas(df_train, excluir)
    if "label" not in df_train.columns:                         # validar columna de etiqueta
        logger.error("No existe la columna 'label' tras la normalización. Abortando.")
        spark.stop()
        raise KeyError("label")

    def castear(df):
        """Castea columnas numéricas a double, descarta no-numéricas
        (salvo label/_source_file) y conserva el esquema esperado."""
        columnas_a_conservar = columnas_numericas + ["label", "_source_file"]
        df = df.select(*[c for c in columnas_a_conservar if c in df.columns])
        df = df.select(*[
            F.col(c).cast("double").alias(c) if c in set(columnas_numericas) else F.col(c)
            for c in df.columns
        ])
        return df

    df_train = castear(df_train).persist(StorageLevel.MEMORY_AND_DISK)   # cast + cache (lee S3 una vez)
    df_test  = castear(df_test).persist(StorageLevel.MEMORY_AND_DISK)     # cast + cache prueba

    # ------------------------------------------------------------------
    # (e) LIMPIEZA CON LOGGING + CACHE
    # ------------------------------------------------------------------
    # Los DataFrames cast se cachean para que los 2 conteos de limpiar() lean
    # de memoria y NO relean los CSV desde S3. El resultado limpio se vuelve a
    # cachear porque a partir de aquí lo reutilizan el StringIndexer, el
    # VectorAssembler, el StandardScaler, los pesos de clase y la escritura.
    # Sin estos cache cada acción recomputaba todo (causa de que tardara 1 hora).
    df_train_limpio = limpiar(df_train, columnas_numericas, "train").persist(StorageLevel.MEMORY_AND_DISK)
    df_test_limpio  = limpiar(df_test,  columnas_numericas, "test").persist(StorageLevel.MEMORY_AND_DISK)
    df_train_limpio.count(); df_test_limpio.count()   # materializa el cache limpio
    df_train.unpersist(); df_test.unpersist()          # libera el cache cast (ya no se usa)
    df_train, df_test = df_train_limpio, df_test_limpio

    # ------------------------------------------------------------------
    # (g) STRINGINDEXER (ajustar SOLO en entrenamiento)
    # ------------------------------------------------------------------
    # frequencyDesc -> la clase más frecuente (benigno) recibe índice 0.0.
    # handleInvalid='keep' permite que etiquetas de prueba NO vistas en
    # entrenamiento (Web attacks, Infiltración) no rompan el transform.
    indexer = StringIndexer(
        inputCol="label", outputCol="label_index",
        stringOrderType="frequencyDesc", handleInvalid="keep")
    indexer_model = indexer.fit(df_train)                       # AJUSTE solo en entrenamiento
    df_train = indexer_model.transform(df_train)                # aplicar a entrenamiento
    df_test  = indexer_model.transform(df_test)                 # aplicar a prueba
    logger.info("StringIndexer ajustado. Clases conocidas: %s", indexer_model.labels)

    # ------------------------------------------------------------------
    # (h) VECTORASSEMBLER (solo numéricas, orden alfabético determinístico)
    # ------------------------------------------------------------------
    columnas_features = sorted(columnas_numericas)              # orden alfabético estable
    assembler = VectorAssembler(
        inputCols=columnas_features, outputCol="features", handleInvalid="skip")
    df_train = assembler.transform(df_train)                    # ensamblar entrenamiento
    df_test  = assembler.transform(df_test)                     # ensamblar prueba
    logger.info("VectorAssembler: %d features ensambladas.", len(columnas_features))

    # ------------------------------------------------------------------
    # (i) STANDARDSCALER (ajustar SOLO en entrenamiento)
    # ------------------------------------------------------------------
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withMean=True, withStd=True)
    scaler_model = scaler.fit(df_train)                         # AJUSTE solo en entrenamiento (evita leakage)
    df_train = scaler_model.transform(df_train)                 # escalar entrenamiento
    df_test  = scaler_model.transform(df_test)                  # escalar prueba
    logger.info("StandardScaler ajustado (withMean=True, withStd=True).")

    # ------------------------------------------------------------------
    # (j) PESOS DE CLASE (solo entrenamiento)
    # ------------------------------------------------------------------
    df_train = calcular_pesos_clase(df_train)

    # ------------------------------------------------------------------
    # Selección de columnas finales a persistir.
    # ------------------------------------------------------------------
    cols_train = ["label", "label_index", "features", "scaled_features",
                  "class_weight", "_source_file", "_split"]
    cols_test  = ["label", "label_index", "features", "scaled_features",
                  "_source_file", "_split"]
    df_train_out = df_train.select(*cols_train)                 # esquema final de entrenamiento
    df_test_out  = df_test.select(*cols_test)                   # esquema final de prueba

    # ------------------------------------------------------------------
    # (k) GUARDADO EN S3
    # ------------------------------------------------------------------
    try:
        df_train_out.write.mode("overwrite").parquet(TRAIN_PROCESSED)   # Parquet entrenamiento
        logger.info("Entrenamiento procesado guardado en %s", TRAIN_PROCESSED)
    except Exception as e:
        logger.error("ERROR al guardar Parquet de entrenamiento (%s): %s", TRAIN_PROCESSED, e)
        spark.stop()
        raise

    try:
        df_test_out.write.mode("overwrite").parquet(TEST_PROCESSED)     # Parquet prueba
        logger.info("Prueba procesada guardada en %s", TEST_PROCESSED)
    except Exception as e:
        logger.error("ERROR al guardar Parquet de prueba (%s): %s", TEST_PROCESSED, e)
        spark.stop()
        raise

    try:
        indexer_model.write().overwrite().save(INDEXER_OUT)             # StringIndexer ajustado
        scaler_model.write().overwrite().save(SCALER_OUT)               # StandardScaler ajustado
        logger.info("Preprocesadores guardados en %s y %s", INDEXER_OUT, SCALER_OUT)
    except Exception as e:
        logger.error("ERROR al guardar preprocesadores: %s", e)
        spark.stop()
        raise

    # Guarda la lista de columnas del VectorAssembler como metadata JSON.
    # Este archivo es consumido por gestion_datos_cic_ids2018.py para
    # garantizar que el vector de features de IDS2018 tenga exactamente
    # la misma dimensión y orden que el pipeline entrenado sobre IDS2017.
    try:
        guardar_features_meta(columnas_features)
    except Exception as e:
        # No se detiene el pipeline: el script 2018 tiene un fallback.
        # Se registra como WARNING porque la ausencia del archivo no es
        # crítica para los pipelines de entrenamiento y evaluación.
        logger.warning(
            "No se pudo guardar features_meta.json (%s). "
            "gestion_datos_cic_ids2018.py operará en modo fallback.",
            e,
        )

    # ------------------------------------------------------------------
    # (m) CIERRE
    # ------------------------------------------------------------------
    spark.stop()
    logger.info("Ingesta y preparación finalizadas correctamente.")


if __name__ == "__main__":
    main()
