#!/usr/bin/env python3


import socket
import time
import argparse
import json
import sys
import boto3
from io import BytesIO

# ── ANSI colors ───────────────────────────────────────────────────────────────
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Productor TCP — lee Parquet de IDS2018 homologado y emite vía socket."
)
parser.add_argument("--host",   default="0.0.0.0",
                    help="IP de escucha del servidor TCP (default: 0.0.0.0)")
parser.add_argument("--port",   default=9999, type=int,
                    help="Puerto TCP (default: 9999)")
parser.add_argument("--rate",   default=500,  type=int,
                    help="Filas/s a emitir. 0 = máxima velocidad (default: 500)")
parser.add_argument("--bucket", default="proyecto-anomalias-emr",
                    help="Bucket S3 del proyecto")
parser.add_argument("--prefix", default="data/cic-ids2018/streaming/processed/",
                    help="Prefijo S3 del Parquet homologado")
parser.add_argument("--region", default="us-east-1")
parser.add_argument("--loop",   action="store_true",
                    help="Repetir el dataset indefinidamente")
args = parser.parse_args()

DELAY = 1.0 / args.rate if args.rate > 0 else 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg, color=CYN):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{RST}", flush=True)

def log_progress(enviadas, t_inicio):
    elapsed = time.time() - t_inicio
    tasa    = enviadas / elapsed if elapsed > 0 else 0
    log(f"{enviadas:>10,} filas | {tasa:>7.0f} filas/s | "
        f"{elapsed:>6.1f}s", GRN)


# ── Lectura de Parquet desde S3 ───────────────────────────────────────────────
def listar_parquet(bucket, prefix, region):
    """Lista todos los archivos .parquet bajo el prefijo S3."""
    s3   = boto3.client("s3", region_name=region)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    if not keys:
        raise FileNotFoundError(
            f"No se encontraron archivos .parquet en "
            f"s3://{bucket}/{prefix}\n"
            f"¿Ejecutaste gestion_datos_cic_ids2018.py primero?"
        )
    log(f"Parquet encontrado: {len(keys)} archivo(s) en s3://{bucket}/{prefix}")
    return keys


def leer_parquet_s3(bucket, key, region):
    """
    Descarga un archivo Parquet de S3 y devuelve un iterador de
    filas serializadas como CSV (string) listas para enviar por TCP.

    Se usa pyarrow para leer el Parquet sin necesidad de Spark.
    Las columnas se ordenan alfabéticamente — mismo orden que
    FEATURE_NAMES en streaming_demo_v2.py.
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        log("ERROR: pyarrow no está instalado.", RED)
        log("Instalar con: pip3 install pyarrow --break-system-packages", YLW)
        sys.exit(1)

    s3  = boto3.client("s3", region_name=region)
    log(f"Descargando {key.split('/')[-1]} ...")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    table   = pq.read_table(BytesIO(data))
    columns = table.column_names

    log(f"  {len(columns)} columnas | {table.num_rows:,} filas")

    # Ordenar columnas: features (alfabético) + label + metadatos
    meta_cols    = {"label", "_source_file", "_split"}
    feature_cols = sorted(c for c in columns if c not in meta_cols)
    ordered_cols = feature_cols + [c for c in ["label", "_source_file", "_split"]
                                   if c in columns]

    table = table.select(ordered_cols)

    # Generar CSV fila por fila
    for batch in table.to_batches(max_chunksize=1000):
        batch_dict = batch.to_pydict()
        n_rows     = batch.num_rows
        for i in range(n_rows):
            row = [str(batch_dict[c][i]) for c in ordered_cols]
            yield ",".join(row) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log(f"Productor TCP inicializado", BLD)
    log(f"Fuente  : s3://{args.bucket}/{args.prefix}")
    log(f"Host    : {args.host}:{args.port}")
    log(f"Tasa    : {'MÁXIMA' if args.rate == 0 else f'{args.rate} filas/s'}")
    log(f"Loop    : {'sí' if args.loop else 'no'}")
    print()

    # Verificar que el Parquet existe antes de abrir el socket
    try:
        parquet_keys = listar_parquet(args.bucket, args.prefix, args.region)
    except FileNotFoundError as e:
        log(str(e), RED)
        sys.exit(1)

    # Abrir socket TCP como servidor
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((args.host, args.port))
    except OSError as e:
        log(f"No se puede abrir el puerto {args.port}: {e}", RED)
        log("¿Hay otro proceso en ese puerto? Prueba: lsof -i :9999", YLW)
        sys.exit(1)

    srv.listen(1)
    log(f"Escuchando en {args.host}:{args.port} — esperando Spark...", YLW)

    conn, addr = srv.accept()
    log(f"Consumidor Spark conectado desde {addr}", GRN)
    print()

    enviadas = 0
    t_inicio = time.time()

    try:
        while True:
            for key in parquet_keys:
                for fila in leer_parquet_s3(args.bucket, key, args.region):
                    conn.sendall(fila.encode("utf-8"))
                    enviadas += 1
                    if DELAY > 0:
                        time.sleep(DELAY)
                    if enviadas % 10_000 == 0:
                        log_progress(enviadas, t_inicio)

            if not args.loop:
                break
            log("Dataset completo. Reiniciando...", YLW)

    except BrokenPipeError:
        log("Consumidor Spark desconectado.", YLW)
    except KeyboardInterrupt:
        log("Interrupción manual (Ctrl+C).", YLW)
    finally:
        elapsed = time.time() - t_inicio
        tasa    = enviadas / elapsed if elapsed > 0 else 0
        print()
        log(f"{'─'*48}")
        log(f"RESUMEN FINAL", BLD)
        log(f"  Filas enviadas : {enviadas:,}")
        log(f"  Tiempo total   : {elapsed:.1f}s")
        log(f"  Tasa promedio  : {tasa:.0f} filas/s")
        log(f"{'─'*48}")
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()