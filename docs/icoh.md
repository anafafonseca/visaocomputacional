# Índice de Colapso Operacional Hospitalar (ICOH)

## O que é

O ICOH é uma métrica preditiva no intervalo de 0 a 100 que estima o risco de sobrecarga sistêmica em um pronto-socorro ou hospital nas próximas 6 horas. Diferentemente de modelos existentes de predição de superlotação — que tipicamente usam apenas dados de fluxo de pacientes — o ICOH integra um sinal radiológico (severidade média dos achados nos últimos exames processados) ao cálculo operacional.

## Variáveis de entrada

| Variável | Descrição | Fonte |
|----------|-----------|-------|
| `occupancy` | Taxa de ocupação de leitos | HIS/RIS |
| `pending_exams` | Exames críticos pendentes de revisão / capacidade do plantão | RIS |
| `wait_ratio` | Tempo médio de espera atual / baseline histórico do mesmo dia/hora | HIS |
| `admission_flow` | Admissões nas últimas 2 horas / média histórica | HIS |
| `epidemic_factor` | Fator sazonal/epidemiológico (índice histórico ou externo) | DataSUS, séries históricas |
| `radiology_severity` | Severidade média dos achados nos últimos N exames analisados pelo RadIA | Camada de visão computacional do RadIA |

## Fórmula

O ICOH é calculado em duas etapas:

1. Combinação linear ponderada das variáveis normalizadas:

```
base = Σ (w_i · x_i)
```

Pesos `w_i` calibráveis por unidade hospitalar. Os defaults documentados em `app/icoh_model.py` representam um pronto-socorro de grande porte.

2. Amplificação não-linear:

```
amplification = 1 + 0.08 · (número de variáveis acima de 0.7)
icoh = min(100, base · 100 · amplification)
```

A amplificação reflete o fato empírico de que múltiplos indicadores simultaneamente elevados aumentam o risco de colapso mais do que a soma linear sugere.

## Predição para 6 horas

A predição usa regressão linear simples sobre a série temporal de cada variável das últimas N horas, projeta os valores para t+1 a t+6 e recalcula o ICOH em cada ponto futuro.

Esta abordagem é intencionalmente simples na prova de conceito. Em produção, a substituição por modelos de série temporal mais robustos (SARIMAX, XGBoost ou LSTM) é direta — a literatura existente (Vural et al., 2025; Antczak et al., 2026) mostra ganhos significativos com esses modelos em predição de fluxo de pronto-socorro.

## Classificação de risco

| ICOH | Nível | Ação típica |
|------|-------|-------------|
| 0–29 | Rotina | Operação normal |
| 30–59 | Moderado | Monitoramento ativo |
| 60–79 | Alerta | Preparação de equipe de sobreaviso |
| 80–94 | Crítico | Ativação imediata de protocolos |
| 95–100 | Colapso iminente | Acionamento de transferência inter-hospitalar |

## Diferencial em relação a trabalhos existentes

A literatura recente cobre amplamente predição de superlotação de pronto-socorro com aprendizado de máquina. O que o ICOH propõe de novo é a integração direta de um sinal radiológico no índice operacional, criando um circuito fechado:

```
   imagens radiológicas  ──►  achados/severidade  ──►  ICOH  ──►  priorização
            ▲                                                          │
            └──────────────────────────────────────────────────────────┘
```

A severidade radiológica funciona como um leading indicator: um aumento na proporção de exames com achados graves precede frequentemente o aumento de admissões e ocupação de leitos. Capturar esse sinal antecipa a predição em comparação a modelos que dependem apenas de fluxo.

## Estado atual

A prova de conceito implementa o cálculo do ICOH e a predição a 6 horas com dados sintéticos realistas. A próxima etapa é a integração com um modelo de análise de RX de tórax (CheXNet ou TorchXRayVision) para fechar o circuito acima com dados reais do dataset público NIH ChestX-ray14.

## Referências relevantes

- Vural, O. et al. (2025). *An Artificial Intelligence-Based Framework for Predicting Emergency Department Overcrowding.* arXiv:2504.18578
- Antczak, J. et al. (2026). *Early predicting of hospital admission using machine learning algorithms.* arXiv:2601.15481
- Moreno-Sánchez, P.A. et al. (2024). *Prediction of patient flow in the emergency department using explainable artificial intelligence.*
