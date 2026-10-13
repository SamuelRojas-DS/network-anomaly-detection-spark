#!/usr/bin/env python3
# monitor.py — Dashboard en terminal para el pipeline de streaming


import json, time, os, sys, argparse
from collections import deque

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--interval",  default=5,   type=int,
                    help="Segundos entre refrescos (default: 5)")
parser.add_argument("--history",   default=30,  type=int,
                    help="Batches a conservar en historial (default: 30)")
parser.add_argument("--sla-p50",   default=12000, type=int)
parser.add_argument("--sla-p95",   default=22000, type=int)
parser.add_argument("--sla-max",   default=60000, type=int)
args = parser.parse_args()

LOCAL_SUMMARY = "/tmp/batch_summary.json"
LOCAL_SLA     = "/tmp/sla_metrics.jsonl"

# ── ANSI ──────────────────────────────────────────────────────────────────────
GRN  = "\033[92m";  YLW  = "\033[93m";  RED  = "\033[91m"
CYN  = "\033[96m";  BLD  = "\033[1m";   RST  = "\033[0m"
MAG  = "\033[95m";  BLU  = "\033[94m";  WHT  = "\033[97m"
DIM  = "\033[2m"

def clr():   os.system("clear")
def mv(r,c): sys.stdout.write(f"\033[{r};{c}H")

def hl(val, lo, hi, fmt="{:.0f}"):
    """Colorea un valor según umbrales lo (verde) hi (rojo)."""
    s = fmt.format(val)
    if val <= lo:   return GRN + BLD + s + RST
    if val >= hi:   return RED + BLD + s + RST
    return YLW + BLD + s + RST

def bar(val, total, width=30, color=GRN):
    if total <= 0:
        return DIM + "░" * width + RST
    filled = max(0, min(width, int(round(val / total * width))))
    return color + "█" * filled + DIM + "░" * (width - filled) + RST

def pct_color(p):
    if p < 10:  return GRN
    if p < 30:  return YLW
    return RED

def sla_badge(status):
    return {
        "OK":        GRN + BLD + " ✔ OK        " + RST,
        "WARNING":   YLW + BLD + " ⚠ WARNING   " + RST,
        "CRITICAL":  RED + BLD + " ✖ CRITICAL  " + RST,
        "VIOLATION": RED + BLD + " ✖✖ VIOLATION" + RST,
    }.get(status, status)

# ── Lectura de archivos ────────────────────────────────────────────────────────
def read_summary():
    try:
        with open(LOCAL_SUMMARY) as f:
            return json.load(f)
    except Exception:
        return None

def read_sla_history():
    records = []
    try:
        with open(LOCAL_SLA) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return records[-args.history:]

# ── Cálculo de percentiles ─────────────────────────────────────────────────────
def percentile(data, p):
    if not data:
        return 0
    s   = sorted(data)
    idx = max(0, min(len(s)-1, int(len(s) * p / 100)))
    return s[idx]

# ── Render del dashboard ───────────────────────────────────────────────────────
def render(summary, sla_hist):
    W = 78    # ancho total del panel
    now = time.strftime("%Y-%m-%d  %H:%M:%S")

    print(f"{BLD}{CYN}╔{'═'*(W-2)}╗{RST}")
    title = f"  ANOMALY DETECTION — STREAMING DASHBOARD     {now}"
    print(f"{BLD}{CYN}║{WHT}{title:<{W-2}}{CYN}║{RST}")
    print(f"{BLD}{CYN}╠{'═'*(W-2)}╣{RST}")

    # ── Sin datos todavía ──────────────────────────────────────────────────────
    if summary is None and not sla_hist:
        print(f"{CYN}║{RST}{'':^{W-2}}{CYN}║{RST}")
        msg = "Esperando el primer batch...  (¿está corriendo streaming_demo_v2.py?)"
        print(f"{CYN}║{RST}{YLW}{msg:^{W-2}}{RST}{CYN}║{RST}")
        print(f"{CYN}║{RST}{'':^{W-2}}{CYN}║{RST}")
        print(f"{BLD}{CYN}╚{'═'*(W-2)}╝{RST}")
        return

    # ══ SECCIÓN 1: BIG DATA METRICS (desde SLA history) ═══════════════════════
    print(f"{BLD}{BLU}║  BIG DATA — MÉTRICAS DEL STREAM{' '*(W-34)}{CYN}║{RST}")
    print(f"{CYN}╠{'─'*(W-2)}╣{RST}")

    if sla_hist:
        total_batches = len(sla_hist)
        last          = sla_hist[-1]
        latencies     = [r["L_E2E_ms"] for r in sla_hist]
        input_rates   = [r["inputRps"]  for r in sla_hist]
        proc_rates    = [r["procRps"]   for r in sla_hist]
        ratios        = [r["ratio"]     for r in sla_hist]
        total_rows    = sum(r["numRows"] for r in sla_hist)
        ok_count      = sum(1 for r in sla_hist if r["status"]=="OK")

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        mx  = max(latencies)

        # Fila 1: contadores
        r1c1 = f"  Batches procesados : {BLD}{WHT}{total_batches:>5}{RST}"
        r1c2 = f"  Filas totales      : {BLD}{WHT}{total_rows:>10,}{RST}"
        r1c3 = f"  SLA cumplimiento   : {BLD}{WHT}{ok_count}/{total_batches}{RST}"
        print(f"{CYN}║{RST}{r1c1:<30}{r1c2:<30}{r1c3:<{W-62}}{CYN}║{RST}")

        # Fila 2: tasas
        inp_s  = f"{last['inputRps']:>7.1f} r/s"
        prc_s  = f"{last['procRps']:>7.1f} r/s"
        rat_c  = GRN if last["ratio"] >= 1.2 else YLW if last["ratio"] >= 1.0 else RED
        rat_s  = f"{rat_c}{BLD}{last['ratio']:.3f}{RST}"
        r2c1   = f"  Input rate (batch) : {BLD}{WHT}{inp_s}{RST}"
        r2c2   = f"  Proc  rate (batch) : {BLD}{WHT}{prc_s}{RST}"
        r2c3   = f"  Ratio proc/input   : {rat_s}"
        print(f"{CYN}║{RST}{r2c1:<36}{r2c2:<36}{r2c3:<{W-74}}{CYN}║{RST}")

        # Barra de ratio
        ratio_bar = bar(min(last["ratio"], 2.0), 2.0, width=38,
                        color=GRN if last["ratio"]>=1.2 else YLW if last["ratio"]>=1.0 else RED)
        print(f"{CYN}║{RST}  Ratio  0.0 {ratio_bar} 2.0+  {CYN}║{RST}")

        # Latencias
        p50c = hl(p50,  args.sla_p50*0.8,  args.sla_p50,  "{:.0f} ms")
        p95c = hl(p95,  args.sla_p95*0.8,  args.sla_p95,  "{:.0f} ms")
        p99c = hl(p99,  args.sla_p95,      args.sla_max,  "{:.0f} ms")
        mxc  = hl(mx,   args.sla_max*0.7,  args.sla_max,  "{:.0f} ms")
        print(f"{CYN}║{RST}  L_E2E  P50:{p50c:<25}  P95:{p95c:<25}  P99:{p99c:<25}  MAX:{mxc:<25}{CYN}║{RST}")

        # Estado SLA del último batch
        badge = sla_badge(last["status"])
        print(f"{CYN}║{RST}  Estado SLA (último batch) : {badge:<35}  "
              f"batch #{last['batchId']:>4}  @ {last['ts']}{CYN}║{RST}")
    else:
        print(f"{CYN}║{RST}  {YLW}Sin datos SLA todavía...{RST}{' '*(W-28)}{CYN}║{RST}")

    # ══ SECCIÓN 2: ML METRICS ═════════════════════════════════════════════════
    print(f"{CYN}╠{'═'*(W-2)}╣{RST}")
    half = (W-3)//2

    # Sub-cabeceras
    rf_hdr  = f"  {BLD}{BLU}RANDOM FOREST  (supervisado){RST}"
    km_hdr  = f"  {BLD}{MAG}KMEANS  (no supervisado){RST}"
    print(f"{CYN}║{RST}{rf_hdr:<{half+8}}{CYN}║{RST}{km_hdr:<{half+8}}{CYN}║{RST}")
    print(f"{CYN}╠{'─'*half}╦{'─'*(W-half-3)}╣{RST}")

    if summary:
        n        = summary["n"]
        rf_ben   = summary["rf_benign"]
        rf_anom  = summary["rf_anomaly"]
        rf_rate  = rf_anom / n * 100 if n else 0
        km_anom  = summary["km_anom"]
        km_norm  = summary["km_normal"]
        km_rate  = km_anom / n * 100 if n else 0
        km_dmean = summary["km_dist_mean"]
        km_dmax  = summary["km_dist_max"]

        # Filas RF
        rf_ben_pct  = rf_ben  / n * 100 if n else 0
        rf_anom_pct = rf_anom / n * 100 if n else 0
        rf_b_bar    = bar(rf_ben,  n, width=22, color=GRN)
        rf_a_bar    = bar(rf_anom, n, width=22, color=RED)

        rf_l1 = f"  Total filas batch : {BLD}{WHT}{n:>8,}{RST}"
        km_l1 = f"  Umbral distancia  : {BLD}{WHT}{UMBRAL_DISTANCIA:>8.1f}{RST}"
        rf_l2 = f"  BENIGN  {rf_b_bar}  {rf_ben:>7,}  ({rf_ben_pct:5.1f}%)"
        km_l2 = f"  Normal  {bar(km_norm, n, 22, GRN)}  {km_norm:>7,}  ({100-km_rate:5.1f}%)"
        rf_l3 = f"  Anomalía{rf_a_bar}  {rf_anom:>7,}  ({rf_anom_pct:5.1f}%)"
        km_l3 = f"  Anomalía{bar(km_anom, n, 22, RED)}  {km_anom:>7,}  ({km_rate:5.1f}%)"

        rc = pct_color(rf_rate);  kc = pct_color(km_rate)
        rf_l4 = f"  Tasa anomalía RF : {rc}{BLD}{rf_rate:6.1f}%{RST}"
        km_l4 = f"  Tasa anomalía KM : {kc}{BLD}{km_rate:6.1f}%{RST}"

        kdc = GRN if km_dmean < UMBRAL_DISTANCIA else RED
        km_l5 = f"  Dist. media → centroide : {kdc}{BLD}{km_dmean:7.2f}{RST}"
        km_l6 = f"  Dist. máxima en batch   : {kdc}{BLD}{km_dmax:7.2f}{RST}"

        rf_lines = [rf_l1, rf_l2, rf_l3, rf_l4, ""]
        km_lines = [km_l1, km_l2, km_l3, km_l4, km_l5]

        # Clases adicionales RF (más allá de benigno)
        label_map = summary.get("label_map", {})
        rf_dist   = summary.get("rf_dist", {})
        extras = [(int(k), rf_dist[k], label_map.get(k, f"class_{k}"))
                  for k in rf_dist if int(k) != 0]
        if extras:
            for (idx, cnt, lbl) in sorted(extras, key=lambda x: -x[1])[:3]:
                pct = cnt / n * 100 if n else 0
                rf_lines.append(
                    f"  {lbl[:16]:<16} {bar(cnt,n,22,RED)}  {cnt:>6,} ({pct:4.1f}%)")

        for i in range(max(len(rf_lines), len(km_lines))):
            rl = rf_lines[i] if i < len(rf_lines) else ""
            kl = km_lines[i] if i < len(km_lines) else ""
            print(f"{CYN}║{RST}{rl:<{half+3}}{CYN}║{RST}{kl:<{half+3}}{CYN}║{RST}")
    else:
        msg = f"  {YLW}Esperando primer batch ML...{RST}"
        print(f"{CYN}║{RST}{msg:<{half+10}}{CYN}║{RST}{' '*(half)}{CYN}║{RST}")

    # ══ SECCIÓN 3: HISTORIAL ÚLTIMOS BATCHES ══════════════════════════════════
    print(f"{CYN}╠{'═'*(W-2)}╣{RST}")
    print(f"{BLD}║  HISTORIAL — ÚLTIMOS BATCHES{' '*(W-31)}{CYN}║{RST}")
    print(f"{CYN}╠{'─'*(W-2)}╣{RST}")

    if sla_hist:
        hdr = f"  {'Batch':<7} {'Filas':>7}  {'Input r/s':>10}  {'Proc r/s':>9}  {'L_E2E':>8}  {'RF anom':>8}  {'Estado':<14}"
        print(f"{CYN}║{RST}{DIM}{hdr:<{W-2}}{RST}{CYN}║{RST}")

        # Unir SLA con ML summary del historial (solo disponemos del último summary)
        # Para el historial mostramos métricas Big Data (las ML solo del último batch)
        for r in sla_hist[-8:]:
            st  = r["status"]
            sc  = GRN if st=="OK" else YLW if st=="WARNING" else RED
            lc  = hl(r["L_E2E_ms"], args.sla_p50, args.sla_p95, "{:.0f}")
            rc_ = GRN if r["ratio"] >= 1.2 else YLW if r["ratio"] >= 1.0 else RED
            row = (f"  #{r['batchId']:<6} {r['numRows']:>7,}  "
                   f"{r['inputRps']:>9.1f}  {rc_}{r['procRps']:>8.1f}{RST}  "
                   f"{lc:>12}ms  {'—':>8}   {sc}{st:<12}{RST}")
            print(f"{CYN}║{RST}{row:<{W+10}}{CYN}║{RST}")
    else:
        print(f"{CYN}║{RST}  {DIM}Sin historial aún...{RST}{' '*(W-22)}{CYN}║{RST}")

    # ══ PIE ═══════════════════════════════════════════════════════════════════
    print(f"{BLD}{CYN}╠{'═'*(W-2)}╣{RST}")
    foot = (f"  Refresco: {args.interval}s   "
            f"Spark UI: http://localhost:4040   "
            f"SLA P50≤{args.sla_p50//1000}s  P95≤{args.sla_p95//1000}s  MAX≤{args.sla_max//1000}s   "
            f"[Ctrl+C para salir]")
    print(f"{BLD}{CYN}║{RST}{DIM}{foot:<{W-2}}{RST}{CYN}{BLD}║{RST}")
    print(f"{BLD}{CYN}╚{'═'*(W-2)}╝{RST}")


# ── Loop principal ─────────────────────────────────────────────────────────────
UMBRAL_DISTANCIA = 15.0   # debe coincidir con streaming_demo_v2.py

print(f"{CYN}{BLD}Monitor iniciado. Esperando datos de streaming...{RST}")
print(f"{DIM}Archivos leídos: {LOCAL_SUMMARY}  |  {LOCAL_SLA}{RST}\n")

try:
    while True:
        summary  = read_summary()
        sla_hist = read_sla_history()
        clr()
        render(summary, sla_hist)
        time.sleep(args.interval)
except KeyboardInterrupt:
    print(f"\n{YLW}Monitor detenido.{RST}")
