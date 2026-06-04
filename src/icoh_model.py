"""
ICOH — Indice de Colapso Operacional Hospitalar

Calculo do indice e predicao para horizonte de 6 horas.

Variaveis de entrada (todas normalizadas para 0-1):
    - occupancy: taxa de ocupacao de leitos
    - pending_exams: exames criticos pendentes / capacidade
    - wait_ratio: tempo medio de espera atual / baseline historico
    - admission_flow: admissoes nas ultimas 2h / media historica
    - epidemic_factor: fator sazonal/epidemiologico
    - radiology_severity: severidade media dos achados radiologicos recentes

Saida:
    ICOH 0-100, onde:
        0-29  : rotina
        30-59 : moderado
        60-79 : alerta
        80-94 : critico
        95-100: colapso iminente
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class HospitalState:
    occupancy: float
    pending_exams: float
    wait_ratio: float
    admission_flow: float
    epidemic_factor: float
    radiology_severity: float

    def to_vector(self):
        return np.array([
            self.occupancy,
            self.pending_exams,
            self.wait_ratio,
            self.admission_flow,
            self.epidemic_factor,
            self.radiology_severity,
        ])


# Pesos calibraveis por unidade hospitalar
# Estes sao os defaults para um pronto-socorro de grande porte
DEFAULT_WEIGHTS = np.array([
    0.22,  # ocupacao de leitos
    0.18,  # exames criticos pendentes
    0.20,  # tempo de espera
    0.15,  # fluxo de admissao
    0.10,  # sazonalidade
    0.15,  # severidade radiologica
])


def compute_icoh(state: HospitalState, weights: np.ndarray = None) -> float:
    """Calcula o ICOH a partir do estado hospitalar atual."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    vector = state.to_vector()

    # Combinacao linear ponderada
    base = float(np.dot(vector, weights))

    # Fator de amplificacao nao-linear: quando varias variaveis estao altas
    # simultaneamente, o risco real de colapso cresce mais do que a media linear
    high_count = int(np.sum(vector > 0.7))
    amplification = 1 + (0.08 * high_count)

    icoh = min(100.0, base * 100 * amplification)
    return round(icoh, 1)


def classify_icoh(icoh: float) -> str:
    if icoh < 30:
        return "rotina"
    if icoh < 60:
        return "moderado"
    if icoh < 80:
        return "alerta"
    if icoh < 95:
        return "critico"
    return "colapso iminente"


def predict_icoh_6h(history: List[HospitalState], weights: np.ndarray = None) -> List[float]:
    """
    Predicao do ICOH para as proximas 6 horas a partir do historico.
    Usa regressao linear simples sobre a serie de cada variavel.
    """
    if len(history) < 6:
        raise ValueError("historico precisa ter pelo menos 6 horas")

    vectors = np.array([s.to_vector() for s in history])
    n_hours = vectors.shape[0]
    n_vars = vectors.shape[1]

    # Para cada variavel, ajusta regressao linear sobre o tempo
    t = np.arange(n_hours)
    predictions_per_var = []

    for v in range(n_vars):
        series = vectors[:, v]
        # Regressao linear simples y = a*t + b
        a, b = np.polyfit(t, series, 1)
        future_t = np.arange(n_hours, n_hours + 6)
        future_values = a * future_t + b
        # Clip em [0, 1.2] — variaveis podem extrapolar 1 em situacoes extremas
        future_values = np.clip(future_values, 0, 1.2)
        predictions_per_var.append(future_values)

    predicted_vectors = np.array(predictions_per_var).T

    icoh_predictions = []
    for vec in predicted_vectors:
        state = HospitalState(*vec)
        icoh_predictions.append(compute_icoh(state, weights))

    return icoh_predictions


def recommend_actions(icoh: float, state: HospitalState) -> List[str]:
    """Recomendacoes operacionais baseadas no ICOH e no estado."""
    actions = []

    if icoh >= 80:
        actions.append("Ativar equipe de sobreaviso imediatamente")
        actions.append("Acionar protocolo de transferencia inter-hospitalar")
    elif icoh >= 60:
        actions.append("Preparar equipe de sobreaviso")
        actions.append("Avaliar pacientes elegiveis para alta antecipada")

    if state.occupancy > 0.85:
        actions.append(f"Liberar leitos de observacao (ocupacao {state.occupancy*100:.0f}%)")

    if state.pending_exams > 0.7:
        actions.append("Priorizar fila de exames criticos pendentes")

    if state.wait_ratio > 1.5:
        actions.append(f"Tempo de espera {state.wait_ratio:.1f}x acima do baseline")

    if not actions:
        actions.append("Operacao dentro dos parametros normais")

    return actions


if __name__ == "__main__":
    # Teste rapido
    state = HospitalState(
        occupancy=0.78,
        pending_exams=0.65,
        wait_ratio=1.4,
        admission_flow=0.72,
        epidemic_factor=0.55,
        radiology_severity=0.61,
    )
    icoh = compute_icoh(state)
    print(f"ICOH atual: {icoh}")
    print(f"Classificacao: {classify_icoh(icoh)}")
    print("Recomendacoes:")
    for action in recommend_actions(icoh, state):
        print(f"  - {action}")
