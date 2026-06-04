"""
Simulacao do ICOH com dados sinteticos de um pronto-socorro de grande porte.

Gera 72 horas de operacao com padroes realistas:
    - Pico de admissoes nas manhas (segundas) e finais de semana
    - Variacao circadiana na ocupacao
    - Eventos de surto epidemiologico (ex: pico de pneumonia no inverno)
    - Correlacao entre severidade radiologica e exames criticos pendentes

Calcula ICOH hora a hora e gera predicao rolante para as proximas 6 horas.
"""

import numpy as np
import matplotlib.pyplot as plt
from icoh_model import HospitalState, compute_icoh, predict_icoh_6h, classify_icoh


def generate_synthetic_hospital_data(n_hours: int = 72, seed: int = 42):
    """Gera serie temporal sintetica de indicadores hospitalares."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # Padrao circadiano: ocupacao sobe ao longo do dia, cai a noite
    circadian = 0.15 * np.sin(2 * np.pi * (t - 6) / 24)

    # Base de ocupacao com tendencia de crescimento ao longo dos 3 dias (surto)
    occupancy_base = 0.65 + circadian + 0.002 * t
    occupancy = np.clip(occupancy_base + rng.normal(0, 0.04, n_hours), 0.3, 0.98)

    # Exames pendentes correlacionados com ocupacao + ruido
    pending_exams = np.clip(
        0.4 + 0.5 * (occupancy - 0.65) + rng.normal(0, 0.06, n_hours),
        0, 1
    )

    # Tempo de espera com crescimento exponencial quando ocupacao > 0.85
    wait_ratio = np.where(
        occupancy > 0.85,
        1.0 + 4 * (occupancy - 0.85) ** 1.5,
        1.0 + 0.5 * occupancy
    ) + rng.normal(0, 0.1, n_hours)
    wait_ratio = np.clip(wait_ratio, 0.5, 4.0)

    # Fluxo de admissao com picos as 9h e 19h
    hour_of_day = t % 24
    admission_peaks = (
        np.exp(-((hour_of_day - 9) ** 2) / 8) +
        np.exp(-((hour_of_day - 19) ** 2) / 8)
    )
    admission_flow = np.clip(
        0.4 + 0.4 * admission_peaks + rng.normal(0, 0.05, n_hours),
        0, 1
    )

    # Fator epidemiologico crescente (simulando inverno)
    epidemic_factor = np.clip(0.3 + 0.005 * t + rng.normal(0, 0.03, n_hours), 0, 1)

    # Severidade radiologica correlacionada com ocupacao e epidemia
    radiology_severity = np.clip(
        0.35 + 0.3 * occupancy + 0.2 * epidemic_factor + rng.normal(0, 0.05, n_hours),
        0, 1
    )

    states = [
        HospitalState(
            occupancy=occupancy[i],
            pending_exams=pending_exams[i],
            wait_ratio=wait_ratio[i] / 4.0,  # normaliza para 0-1
            admission_flow=admission_flow[i],
            epidemic_factor=epidemic_factor[i],
            radiology_severity=radiology_severity[i],
        )
        for i in range(n_hours)
    ]

    return states, {
        "occupancy": occupancy,
        "pending_exams": pending_exams,
        "wait_ratio": wait_ratio,
        "admission_flow": admission_flow,
        "epidemic_factor": epidemic_factor,
        "radiology_severity": radiology_severity,
    }


def run_simulation():
    print("=" * 70)
    print("RadIA — Simulacao do Indice de Colapso Operacional Hospitalar")
    print("=" * 70)

    n_hours = 72
    states, raw_data = generate_synthetic_hospital_data(n_hours)

    # Calcula ICOH para cada hora
    icoh_series = np.array([compute_icoh(s) for s in states])

    # Predicao rolante: a cada 6h, prediz as proximas 6
    prediction_points = []
    for start in range(6, n_hours - 6, 6):
        history = states[:start]
        try:
            preds = predict_icoh_6h(history)
            for offset, p in enumerate(preds):
                prediction_points.append((start + offset, p))
        except ValueError:
            continue

    # Estatisticas
    print(f"\nPeriodo simulado: {n_hours} horas")
    print(f"ICOH medio: {icoh_series.mean():.1f}")
    print(f"ICOH maximo: {icoh_series.max():.1f}  (hora {icoh_series.argmax()})")
    print(f"ICOH minimo: {icoh_series.min():.1f}  (hora {icoh_series.argmin()})")

    print("\nDistribuicao por nivel de risco:")
    levels = [classify_icoh(v) for v in icoh_series]
    for level in ["rotina", "moderado", "alerta", "critico", "colapso iminente"]:
        count = levels.count(level)
        pct = 100 * count / n_hours
        bar = "#" * int(pct / 2)
        print(f"  {level:20s} {count:3d}h ({pct:5.1f}%) {bar}")

    # Estado final e predicao
    print("\n" + "=" * 70)
    print("ESTADO ATUAL (hora 72)")
    print("=" * 70)
    final = states[-1]
    icoh_final = compute_icoh(final)
    print(f"ICOH: {icoh_final}  [{classify_icoh(icoh_final)}]")
    print(f"  Ocupacao de leitos:    {final.occupancy*100:.1f}%")
    print(f"  Exames pendentes:      {final.pending_exams*100:.1f}%")
    print(f"  Razao tempo espera:    {final.wait_ratio*4:.2f}x baseline")
    print(f"  Fluxo de admissao:     {final.admission_flow*100:.1f}%")
    print(f"  Fator epidemiologico:  {final.epidemic_factor*100:.1f}%")
    print(f"  Severidade radiologica: {final.radiology_severity*100:.1f}%")

    print("\nPredicao para as proximas 6 horas:")
    future = predict_icoh_6h(states)
    for h, p in enumerate(future, 1):
        marker = "  ALERTA" if p >= 80 else "  ATENCAO" if p >= 60 else ""
        print(f"  +{h}h: ICOH = {p:.1f}  [{classify_icoh(p)}]{marker}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    ax1.plot(icoh_series, color="#1A6B9A", linewidth=2, label="ICOH historico")
    if prediction_points:
        px, py = zip(*prediction_points)
        ax1.scatter(px, py, color="#D85A30", s=15, alpha=0.5, label="Predicoes 6h")

    ax1.axhspan(0, 30, alpha=0.08, color="green")
    ax1.axhspan(30, 60, alpha=0.08, color="yellow")
    ax1.axhspan(60, 80, alpha=0.10, color="orange")
    ax1.axhspan(80, 100, alpha=0.12, color="red")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, n_hours)
    ax1.set_ylabel("ICOH (0-100)")
    ax1.set_title("RadIA — Indice de Colapso Operacional Hospitalar (72h)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(raw_data["occupancy"] * 100, label="Ocupacao (%)", linewidth=1.2)
    ax2.plot(raw_data["pending_exams"] * 100, label="Exames pendentes (%)", linewidth=1.2)
    ax2.plot(raw_data["wait_ratio"] * 25, label="Espera (x baseline, *25)", linewidth=1.2, alpha=0.7)
    ax2.set_xlim(0, n_hours)
    ax2.set_xlabel("Hora")
    ax2.set_ylabel("Indicadores")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = "../assets/icoh_output.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nGrafico salvo em: {output_path}")


if __name__ == "__main__":
    run_simulation()
