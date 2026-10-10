# network-anomaly-detection-spark
# Distributed System for Near Real-Time Network Anomaly Detection with Spark MLlib on AWS EMR

Distributed pipeline for detecting network traffic anomalies (DoS/DDoS, PortScan, Botnet) using Apache Spark, Spark MLlib, and AWS EMR, with a near real-time demonstration via Spark Structured Streaming on unseen data.

## Problem

Traditional centralized traffic analysis systems struggle with scalability, latency, and throughput as network volume grows. This project builds a **distributed, flow-based anomaly detection system** — analyzing statistical/temporal characteristics of network flows (no deep packet inspection) — to evaluate how Spark + EMR perform under high-volume, high-velocity, near real-time constraints.

Near real-time here means continuous processing in **10-second micro-batches** via Spark Structured Streaming.

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | Training & batch evaluation | 5 days of labeled traffic; Monday (benign) trains KMeans and contributes to Random Forest training; Wednesday (DoS/DDoS) and Friday (DDoS LOIT, PortScan, Botnet) complete supervised training and KMeans threshold calibration; Tuesday/Thursday held out as the batch test set |
| [CSE-CIC-IDS2018](https://registry.opendata.aws/cse-cic-ids2018) | Streaming demonstration only | Larger-scale successor dataset (50 attacking machines, 420+ victim machines), hosted on public S3 — consumed directly by the EMR cluster, never seen during training, used to validate detection on genuinely unknown data |

## Models

- **Random Forest (supervised)** — Spark MLlib. Chosen for robustness to noisy data, parallelizable training at scale, and relative interpretability.
- **KMeans (unsupervised)** — trained exclusively on benign traffic; anomalies are flagged via centroid-distance scoring against a calibrated threshold. Relevant when labels aren't available or when attacks are previously unknown.

The comparison prioritizes **distributed system behavior** (training time, horizontal scalability, inference latency) as much as classic ML metrics (Accuracy, Precision, Recall, F1, False Positive Rate).

## What I did

- Built a distributed ingestion pipeline with explicit schema control (no auto-inference) to avoid corrupt-value errors in the CICFlowMeter CSVs, using `unionByName` to safely merge files across days.
- Cleaned nulls/infinites/duplicates and split data respecting the dataset's temporal structure to avoid leakage.
- Engineered features: frequency-ordered label indexing, deterministic alphabetical vector assembly, and standardization (zero mean/unit variance) fit only on the training set — critical for KMeans, whose distance metric is scale-sensitive.
- Trained Random Forest and KMeans via Spark MLlib's Pipeline API on an AWS EMR cluster, evaluating both on a temporal train/test split.
- Implemented inference with **Spark Structured Streaming** in 10-second micro-batches over CSE-CIC-IDS2018 consumed from S3, measuring end-to-end latency (`tpredict - tingest`) via `StreamingQueryListener`.
- Ran horizontal scalability experiments by varying the EMR core-node count (2→5 nodes) and tracking throughput, parallelism, and resource usage.

## Architecture / Infrastructure

**EMR cluster topology**

| Role | EC2 type | vCPU | RAM | Storage | Count |
|---|---|---|---|---|---|
| Master | m5.xlarge | 4 | 16 GiB | 64 GiB EBS | 1 |
| Core | m5.xlarge | 4 | 16 GiB | 64 GiB EBS | 2 (min) |

**Key Spark configuration** (derived from m5.xlarge sizing heuristics — full file in [`/config/spark-defaults.conf`](./config/spark-defaults.conf)):

```
spark.executor.cores            2
spark.executor.memory           6g
spark.executor.instances        2
spark.default.parallelism       16
spark.sql.shuffle.partitions    16
spark.memory.fraction           0.75
spark.memory.storageFraction    0.40
spark.serializer                org.apache.spark.serializer.KryoSerializer
```

**Streaming configuration**

| Parameter | Value |
|---|---|
| Trigger interval | 10s micro-batches |
| Output mode | append |
| maxFilesPerTrigger | 5 |
| Watermark | 30s |
| Sink | console + S3 (parquet) |

Region: `us-east-1`, co-located with the public CSE-CIC-IDS2018 bucket to avoid cross-region transfer costs/latency.

## Tech stack

`Python` · `PySpark` · `Spark MLlib` · `Spark Structured Streaming` · `AWS EMR` · `AWS S3` · `AWS CLI`

## Repository structure

```
├── scripts/
│   ├── data_management_cic_ids2017.py
│   ├── data_management_cic_ids2018.py
│   ├── pipeline_random_forest.py
│   ├── pipeline_kmeans.py
│   ├── tcp_sender.py
│   ├── streaming_demo.py
│   └── monitor.py
├── config/
│   ├── spark-defaults.conf
│   └── emr-spark-config.json
├── notebooks/            # Exploración y validación de resultados
├── results/              # Métricas de evaluación batch y streaming, gráficas de escalabilidad
├── docs/
│   └── Proyecto_BD.pdf   # Especificación técnica completa del proyecto
└── README.md
```

## How to reproduce

**1. Provision the EMR cluster**
```bash
aws emr create-cluster \
  --name "ProjectAnomalies" \
  --release-label emr-7.1.0 \
  --region us-east-1 \
  --applications Name=Spark Name=Hadoop Name=JupyterEnterpriseGateway \
  --instance-groups \
      InstanceGroupType=MASTER,InstanceType=m5.xlarge,InstanceCount=1 \
      InstanceGroupType=CORE,InstanceType=m5.xlarge,InstanceCount=2 \
  --ec2-attributes KeyName=<EC2_KEY_PAIR>,SubnetId=<SUBNET_ID>,InstanceProfile=EMR_EC2_DefaultRole \
  --service-role EMR_DefaultRole \
  --log-uri s3://project-anomalies-emr/logs/emr/ \
  --bootstrap-actions Path=s3://project-anomalies-emr/scripts/bootstrap.sh \
  --configurations file://config/emr-spark-config.json \
  --enable-debugging \
  --auto-terminate
```

**2. Train the models**
```bash
spark-submit scripts/pipeline_random_forest.py
spark-submit scripts/pipeline_kmeans.py
```

**3. Run the streaming demonstration**
```bash
spark-submit scripts/streaming_demo.py
```

## Results

> _Completar con las métricas finales una vez evaluados ambos modelos: Accuracy/Precision/Recall/F1 (Random Forest), separación de scores (KMeans), throughput y latencia end-to-end del pipeline de streaming, y comparación de escalabilidad horizontal (2 a 5 nodos)._

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization.* ICISSP.
- CSE & CIC (2018). *CSE-CIC-IDS2018 on AWS.* Registry of Open Data on AWS.
- Meng, X. et al. (2016). *MLlib: Machine learning in Apache Spark.* JMLR.
- Armbrust, M. et al. (2018). *Structured streaming: A declarative API for real-time applications in Apache Spark.* ACM SIGMOD.

## Author
Samuel David Rojas Cárdenas — [samueldavidrojascardenas@gmail.com](mailto:samueldavidrojascardenas@gmail.com)
