#!/usr/bin/env bash
# create_bucket.sh
# ==================================================================
# PROPÓSITO:
#   Crea el bucket S3 del proyecto y toda su estructura de carpetas
#   (ver sección "Bucket S3" de guia_streaming.md), para que
#   gestion_datos_1.py, pipeline_random_forest.py, pipeline_kmeans.py,
#   gestion_datos_cic_ids2018.py y streaming_demo_v2.py puedan
#   escribir sin fallar por rutas inexistentes.
#
# USO:
#   bash create_bucket.sh <nombre-bucket> <región> [arn-rol-externo]
#
#   Ejemplos:
#     bash create_bucket.sh proyecto-anomalias-emr us-east-1
#     bash create_bucket.sh proyecto-anomalias-emr us-east-1 \
#         arn:aws:iam::CUENTA_EXTERNA_ID:role/EMR_EC2_DefaultRole
#
# REQUISITOS:
#   AWS CLI configurado con credenciales que tengan permisos
#   s3:CreateBucket, s3:PutObject y s3:PutBucketPolicy.
# ==================================================================

set -euo pipefail

BUCKET="${1:?Uso: bash create_bucket.sh <nombre-bucket> <región> [arn-rol-externo]}"
REGION="${2:?Uso: bash create_bucket.sh <nombre-bucket> <región> [arn-rol-externo]}"
EXTERNAL_ROLE_ARN="${3:-}"

echo "=== Creando bucket s3://${BUCKET} en ${REGION} ==="

# us-east-1 no acepta LocationConstraint en create-bucket.
if [ "${REGION}" == "us-east-1" ]; then
    aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}"
else
    aws s3api create-bucket \
        --bucket "${BUCKET}" \
        --region "${REGION}" \
        --create-bucket-configuration LocationConstraint="${REGION}"
fi

echo "=== Bloqueando acceso público (buena práctica por defecto) ==="
aws s3api put-public-access-block \
    --bucket "${BUCKET}" \
    --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "=== Creando estructura de carpetas ==="
# S3 no tiene carpetas reales; se crean "marcadores" (objetos vacíos
# terminados en /) para que la estructura sea visible y navegable,
# y para documentar el layout esperado por los scripts del proyecto.
PREFIXES=(
    "data/cic-ids2017/train/"
    "data/cic-ids2017/test/"
    "data/cic-ids2017/train/processed/"
    "data/cic-ids2017/test/processed/"
    "data/cic-ids2018/streaming/processed/"
    "models/random_forest_pipeline/preprocessors/indexer/"
    "models/random_forest_pipeline/preprocessors/scaler/"
    "models/random_forest_pipeline/model_v1/"
    "models/random_forest_pipeline/model_optimized/"
    "models/kmeans_pipeline/model_v1/"
    "models/kmeans_pipeline/model_optimized/"
    "models/kmeans_pipeline/threshold/"
    "checkpoints/streaming/"
    "results/batch_evaluation/"
    "results/batch_evaluation/predictions_rf/"
    "results/kmeans_evaluation/"
    "results/kmeans_evaluation/predictions_kmeans/"
    "results/streaming_metrics/rf/"
    "results/streaming_metrics/kmeans/"
    "logs/emr/"
    "logs/spark_event_logs/"
    "scripts/"
)

for prefix in "${PREFIXES[@]}"; do
    aws s3api put-object --bucket "${BUCKET}" --key "${prefix}" >/dev/null
    echo "  creado: s3://${BUCKET}/${prefix}"
done

# ── Acceso cross-account opcional (rol EMR en otra cuenta) ────────────────────
if [ -n "${EXTERNAL_ROLE_ARN}" ]; then
    echo "=== Configurando política de bucket para acceso cross-account ==="
    echo "    Rol externo: ${EXTERNAL_ROLE_ARN}"

    POLICY_FILE=$(mktemp)
    cat > "${POLICY_FILE}" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowExternalEMRRole",
      "Effect": "Allow",
      "Principal": { "AWS": "${EXTERNAL_ROLE_ARN}" },
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    }
  ]
}
EOF
    aws s3api put-bucket-policy --bucket "${BUCKET}" --policy "file://${POLICY_FILE}"
    rm -f "${POLICY_FILE}"
    echo "  política aplicada."
fi

echo ""
echo "=== Listo. Verifica con: ==="
echo "aws s3 ls s3://${BUCKET}/ --recursive --human-readable"
