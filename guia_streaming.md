# Guía de Streaming — Sistema Distribuido de Detección de Anomalías de Red
**Proyecto Big Data · Universidad del Rosario · 2026**  
Sergio Andrés Guevara · Samuel David Rojas

guia de como visualizar el despliegue

---

## Índice

1. [Visión general del flujo](#1-visión-general-del-flujo)
2. [Infraestructura requerida](#2-infraestructura-requerida)
3. [Archivos del proyecto](#3-archivos-del-proyecto)
4. [Fase 0 — Preparación del entorno AWS](#4-fase-0--preparación-del-entorno-aws)
5. [Fase 1 — Entrenamiento previo (prerequisito)](#5-fase-1--entrenamiento-previo-prerequisito)
6. [Fase 2 — Homologación del dataset IDS2018](#6-fase-2--homologación-del-dataset-ids2018)
7. [Fase 3 — Ejecución del streaming](#7-fase-3--ejecución-del-streaming)
8. [Fase 4 — Monitoreo y visualización](#8-fase-4--monitoreo-y-visualización)
9. [Fase 5 — Cierre y recolección de resultados](#9-fase-5--cierre-y-recolección-de-resultados)
10. [Referencia rápida de comandos](#10-referencia-rápida-de-comandos)


---

## 1. Visión general del flujo

El pipeline de streaming se compone de **tres etapas secuenciales** separadas en el tiempo. Las fases 0–2 son preparación; la fase 3 es la demo activa.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PREPARACIÓN (offline — antes de la demo)                                   │
│                                                                             │
│  [gestion_datos_1.py]       [pipeline_random_forest.py + pipeline_kmeans.py]│
│   CIC-IDS2017 → Parquet  →   Modelos RF + KMeans → S3                       │
│                                                                             │
│  [gestion_datos_cic_ids2018.py]                                             │
│   CIC-IDS2018 público → Homologar → Parquet limpio → S3                    │
│                                                                             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ Parquet + Modelos en S3
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEMO EN STREAMING (online — tres terminales simultáneas)                   │
│                                                                             │
│  EC2 (Terminal 1)           EMR Master (Terminal 2)    Local (Terminal 3)   │
│                                                                             │
│  tcp_sender.py              streaming_demo_v2.py        monitor.py          │
│  Lee Parquet S3    TCP:9999  readStream socket          Lee /tmp/*.jsonl     │
│  Emite filas CSV ──────────► from_csv + PipelineModel   Dashboard en vivo   │
│  a tasa controlada          RF + KMeans                                     │
│                             Resultados → S3                                 │
│                             Métricas → Spark UI :4040                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Roles de cada componente

| Componente | Dónde corre | Qué hace |
|---|---|---|
| `gestion_datos_cic_ids2018.py` | Nodo master EMR | Homologa IDS2018, produce Parquet limpio en S3 |
| `tcp_sender.py` | EC2 separada (t3.medium) | Lee el Parquet y emite filas CSV por TCP |
| `streaming_demo_v2.py` | Nodo master EMR | Recibe el stream, aplica RF + KMeans, escribe en S3 |
| `monitor.py` | Nodo master EMR | Dashboard en terminal con métricas Big Data + ML |
| Spark UI `:4040` | Nodo master EMR | Visualización de series temporales de throughput y latencia |

---

## 2. Infraestructura requerida

### Clúster EMR

| Parámetro | Valor |
|---|---|
| Release | `emr-7.1.0` |
| Región | `us-east-1` |
| Nodo master | 1 × `m5.xlarge` (4 vCPU / 16 GiB) |
| Nodos core (base) | 2 × `m5.xlarge` |
| Aplicaciones | Spark 3.5.1, Hadoop 3.3.6 |
| S3 checkpoint | `s3://proyecto-anomalias-emr/checkpoints/streaming/` |

### EC2 productora

| Parámetro | Valor |
|---|---|
| Tipo | `t3.medium` (2 vCPU / 4 GiB) |
| Región | `us-east-1` (misma que el clúster) |
| Puerto TCP | `9999` (abierto hacia el Security Group del master EMR) |
| Dependencias | `python3`, `boto3`, `pyarrow` |

### Bucket S3

```
s3://proyecto-anomalias-emr/
├── data/
│   ├── cic-ids2017/
│   │   ├── train/          ← CSVs lunes, miércoles, viernes
│   │   └── test/           ← CSVs martes, jueves
│   └── cic-ids2018/
│       └── streaming/
│           └── processed/  ← Parquet producido por gestion_datos_cic_ids2018.py
├── models/
│   ├── random_forest_pipeline/
│   │   └── preprocessors/
│   │       ├── features_meta.json
│   │       ├── indexer/
│   │       └── scaler/
│   └── kmeans_pipeline/
├── checkpoints/
│   └── streaming/
├── results/
│   └── streaming_metrics/
│       ├── rf/
│       └── kmeans/
└── logs/
    ├── emr/
    └── spark_event_logs/
```

---

## 3. Archivos del proyecto

| Archivo | Fase | Descripción |
|---|---|---|
| `create_bucket.sh` | Preparación | Crea el bucket S3 con toda su estructura |
| `gestion_datos_1.py` | Fase 1 | Ingiere, limpia y preprocesa CIC-IDS2017 (train/test); produce los preprocesadores compartidos (StringIndexer, StandardScaler, `features_meta.json`) |
| `pipeline_random_forest.py` | Fase 1 | Entrena, evalúa y optimiza el modelo Random Forest |
| `pipeline_kmeans.py` | Fase 1 | Entrena, calibra, evalúa y optimiza el modelo KMeans |
| `gestion_datos_cic_ids2018.py` | Fase 2 | Homologa IDS2018 y produce el Parquet |
| `tcp_sender.py` | Fase 3 — Terminal 1 | Productor TCP (corre en la EC2) |
| `streaming_demo_v2.py` | Fase 3 — Terminal 2 | Consumidor Spark (corre en el master EMR) |
| `monitor.py` | Fase 4 — Terminal 3 | Dashboard de métricas en terminal |

---

## 4. Fase 0 — Preparación del entorno AWS

### 4.1 Crear el bucket S3

Desde tu máquina local con AWS CLI configurado:

```bash
# Sin acceso cross-account
bash create_bucket.sh proyecto-anomalias-emr us-east-1

# Con acceso cross-account (pasar ARN del rol EMR externo)
bash create_bucket.sh proyecto-anomalias-emr us-east-1 \
  arn:aws:iam::CUENTA_EXTERNA_ID:role/EMR_EC2_DefaultRole
```

Verificar que el bucket y las carpetas quedaron creados:

```bash
aws s3 ls s3://proyecto-anomalias-emr/ --recursive --human-readable
```

### 4.2 Crear el clúster EMR

```bash
aws emr create-cluster \
  --name 'ProyectoAnomalias-BD2026' \
  --release-label emr-7.1.0 \
  --region us-east-1 \
  --applications Name=Spark Name=Hadoop \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceType=m5.xlarge,InstanceCount=1 \
    InstanceGroupType=CORE,InstanceType=m5.xlarge,InstanceCount=2 \
  --ec2-attributes KeyName=<EC2_KEY_PAIR>,SubnetId=<SUBNET_ID> \
  --service-role EMR_DefaultRole \
  --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
  --log-uri s3://proyecto-anomalias-emr/logs/emr/ \
  --configurations file://emr-spark-config.json \
  --enable-debugging
```

Obtener el DNS público del master (necesario para SSH):

```bash
aws emr list-instances \
  --cluster-id <CLUSTER_ID> \
  --instance-group-types MASTER \
  --query 'Instances[0].PublicDnsName' \
  --output text
```

### 4.3 Crear la EC2 productora

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name <EC2_KEY_PAIR> \
  --region us-east-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tcp-sender}]'
```

Instalar dependencias en la EC2:

```bash
ssh -i <key.pem> ec2-user@<EC2-PUBLIC-IP>
pip3 install boto3 pyarrow --break-system-packages
```

### 4.4 Configurar Security Group para TCP:9999

```bash
# Obtener el SG del master EMR
EMR_MASTER_SG=$(aws emr describe-cluster \
  --cluster-id <CLUSTER_ID> \
  --query 'Cluster.Ec2InstanceAttributes.EmrManagedMasterSecurityGroup' \
  --output text)

# Permitir que el master EMR conecte al puerto 9999 de la EC2
aws ec2 authorize-security-group-ingress \
  --group-id <EC2-SG-ID> \
  --protocol tcp \
  --port 9999 \
  --source-group $EMR_MASTER_SG
```

---

## 5. Fase 1 — Entrenamiento previo (prerequisito)

> Esta fase debe completarse **antes** de la demo. Si los modelos ya están en S3, pasar directamente a la Fase 2.

### 5.1 Subir los datasets de entrenamiento

```bash
# CIC-IDS2017 — entrenamiento
aws s3 cp Monday-WorkingHours.pcap_ISCX.csv \
  s3://proyecto-anomalias-emr/data/cic-ids2017/train/
aws s3 cp Wednesday-WorkingHours.pcap_ISCX.csv \
  s3://proyecto-anomalias-emr/data/cic-ids2017/train/
aws s3 cp Friday-WorkingHours.pcap_ISCX.csv \
  s3://proyecto-anomalias-emr/data/cic-ids2017/train/

# CIC-IDS2017 — prueba
aws s3 cp Tuesday-WorkingHours.pcap_ISCX.csv \
  s3://proyecto-anomalias-emr/data/cic-ids2017/test/
aws s3 cp Thursday-WorkingHours.pcap_ISCX.csv \
  s3://proyecto-anomalias-emr/data/cic-ids2017/test/
```

### 5.2 Ejecutar el pipeline de entrenamiento

```bash
# Conectar al master EMR
ssh -i <key.pem> hadoop@<MASTER-DNS>

# Preprocesamiento IDS2017
spark-submit --master yarn gestion_datos_1.py

# Entrenamiento Random Forest
spark-submit --master yarn pipeline_random_forest.py

# Entrenamiento KMeans
spark-submit --master yarn pipeline_kmeans.py
```

### 5.3 Verificar que los modelos quedaron en S3

```bash
aws s3 ls s3://proyecto-anomalias-emr/models/random_forest_pipeline/
aws s3 ls s3://proyecto-anomalias-emr/models/kmeans_pipeline/
aws s3 ls s3://proyecto-anomalias-emr/models/random_forest_pipeline/preprocessors/features_meta.json
```

> **Importante:** si `features_meta.json` no existe, `gestion_datos_cic_ids2018.py` entra en modo fallback con un WARNING en los logs. El resultado es funcional pero puede haber discrepancias menores en el número de features. Se recomienda asegurarse de que `gestion_datos_1.py` lo genere.

---

## 6. Fase 2 — Homologación del dataset IDS2018

Esta fase corre **una sola vez**, antes de la demo. Produce el Parquet limpio que `tcp_sender.py` leerá durante el streaming.

### 6.1 Subir el script al master EMR

```bash
# Desde tu máquina local
aws s3 cp gestion_datos_cic_ids2018.py \
  s3://proyecto-anomalias-emr/scripts/

# En el master EMR
ssh -i <key.pem> hadoop@<MASTER-DNS>
aws s3 cp s3://proyecto-anomalias-emr/scripts/gestion_datos_cic_ids2018.py .
```

### 6.2 Ejecutar la homologación

```bash
spark-submit --master yarn \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memory=6g \
  --conf spark.executor.instances=2 \
  gestion_datos_cic_ids2018.py
```

Tiempo estimado con 2 nodos core: **15–25 minutos** (descarga de ~1.3M filas desde el bucket público + limpieza + escritura Parquet).

### 6.3 Verificar el Parquet producido

```bash
aws s3 ls s3://proyecto-anomalias-emr/data/cic-ids2018/streaming/processed/ \
  --human-readable
```

Se espera ver varios archivos `.parquet` con un tamaño total de **200–400 MB**.

### 6.4 Qué hace internamente el script

```
Carga CSV IDS2018 (bucket público, acceso anónimo)
    └── normalizar_nombre()          # snake_case + minúsculas
    └── homologar_columnas()         # 45 renombramientos IDS2018 → IDS2017
    └── alinear_con_esquema_2017()   # imputa columnas ausentes con 0.0
    └── castear_numericas()          # todas las features a DoubleType
    └── limpiar()                    # nulos → infinitos → duplicados
    └── Parquet → S3                 # 77 features + label + metadatos
```

> El script **NO aplica** VectorAssembler, StandardScaler ni StringIndexer. Esas transformaciones las aplica el `PipelineModel` de Spark dentro de `streaming_demo_v2.py` en cada micro-batch.

---

## 7. Fase 3 — Ejecución del streaming

Esta es la demo activa. Requiere **tres terminales abiertas simultáneamente**.

### 7.1 Preparar los scripts en cada máquina

**En la EC2 (Terminal 1):**
```bash
ssh -i <key.pem> ec2-user@<EC2-PUBLIC-IP>
aws s3 cp s3://proyecto-anomalias-emr/scripts/tcp_sender.py .
```

**En el master EMR (Terminal 2):**
```bash
ssh -i <key.pem> hadoop@<MASTER-DNS>
aws s3 cp s3://proyecto-anomalias-emr/scripts/streaming_demo_v2.py .
aws s3 cp s3://proyecto-anomalias-emr/scripts/monitor.py .
```

### 7.2 Paso 1 — Lanzar el productor TCP (Terminal 1 — EC2)

```bash
# Tráfico benigno (inicio de la demo)
python3 tcp_sender.py --rate 500

# Simulación de pico DDoS-HOIC (fase crítica)
python3 tcp_sender.py --rate 3500

# Velocidad máxima (prueba de throughput puro)
python3 tcp_sender.py --rate 0

# Loop continuo (para demos largas)
python3 tcp_sender.py --rate 500 --loop
```

La terminal mostrará:

```
[14:30:00] Productor TCP inicializado
[14:30:00] Fuente  : s3://proyecto-anomalias-emr/data/cic-ids2018/streaming/processed/
[14:30:00] Host    : 0.0.0.0:9999
[14:30:00] Tasa    : 500 filas/s
[14:30:00] Parquet encontrado: 3 archivo(s)
[14:30:01] Escuchando en 0.0.0.0:9999 — esperando Spark...
```

> **El productor queda bloqueado esperando la conexión de Spark.** No continúa hasta que el consumidor conecte. Lanzar el paso 2 antes de que pasen más de 60 segundos.

### 7.3 Paso 2 — Lanzar el consumidor Spark (Terminal 2 — master EMR)

```bash
spark-submit --master yarn \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memory=6g \
  --conf spark.executor.instances=2 \
  streaming_demo_v2.py \
  --producer-host <EC2-PRIVATE-IP> \
  --producer-port 9999 \
  --trigger 10
```

Parámetros disponibles:

| Parámetro | Default | Descripción |
|---|---|---|
| `--producer-host` | `localhost` | IP privada de la EC2 productora |
| `--producer-port` | `9999` | Puerto TCP del productor |
| `--trigger` | `10` | Duración del micro-batch en segundos |

La terminal mostrará por cada micro-batch:

```
══════════════════════════════════════════════════════════════════════
  BATCH #  12   14:32:10   filas:  5,000   proc:  3,420 ms  [cache: 3 jobs]
──────────────────────────────────────────────────────────────────────

  RANDOM FOREST — Clasificación supervisada
    benign                  ████████████████░░░░░░░░   4,100  ( 82.0%)
    ddos_loic_http          ████░░░░░░░░░░░░░░░░░░░░     700  ( 14.0%)
    ddos_hoic               ██░░░░░░░░░░░░░░░░░░░░░░     200  (  4.0%)

    Tasa anomalía RF :  18.0% ↑

  KMEANS — Detección no supervisada  (umbral: 15.0)
    Normal   ████████████░░░░░░░░░░░░   3,800  ( 76.0%)
    Anomalía ████████░░░░░░░░░░░░░░░░   1,200  ( 24.0%)

    Dist. media → centroide :   18.43
    Dist. máxima en batch   :   87.21
══════════════════════════════════════════════════════════════════════
```

### 7.4 Paso 3 — Lanzar el monitor (Terminal 3 — local o EMR)

```bash
# Refresco cada 5 segundos (default)
python3 monitor.py

# Refresco cada 3 segundos
python3 monitor.py --interval 3

# Conservar historial de 50 batches
python3 monitor.py --interval 5 --history 50
```

### 7.5 Paso 4 — Abrir Spark UI (Terminal local)

```bash
# Tunnel SSH para acceder a Spark UI
ssh -i <key.pem> \
    -L 4040:localhost:4040 \
    -N \
    hadoop@<MASTER-DNS>

# Abrir en el navegador:
# http://localhost:4040
```


---

