# RadIA Triage Platform

> Plataforma de inteligência hospitalar que integra análise de imagens radiológicas por IA com predição de sobrecarga operacional em um pipeline único.

**Status:** Prova de conceito — em desenvolvimento

**Autores:** Anderson Ferreira dos Santos · Ana Flavia Arcanjo da Fonseca

---

## Proposta

A maioria das soluções existentes trata diagnóstico assistido por imagem e gestão operacional hospitalar como problemas separados. O RadIA propõe a integração dos dois em uma plataforma única, com três camadas complementares:

1. **Análise radiológica assistida** — classificação de achados em imagens de raio-X de tórax com explicabilidade visual (mapas de calor sobre as regiões suspeitas).
2. **Interação em linguagem natural** — o profissional médico pode fazer perguntas sobre o exame e receber respostas contextualizadas àquela imagem específica.
3. **Índice de Colapso Operacional Hospitalar (ICOH)** — métrica preditiva multivariada que cruza indicadores radiológicos, ocupação de leitos e fluxo de exames para antecipar sobrecarga sistêmica com horizonte de 6 horas.

O diferencial está na conexão entre as três camadas: os achados radiológicos alimentam a fila clínica, que alimenta o índice operacional, que retroalimenta a priorização de exames.

---

## O que está implementado

Este repositório contém a prova de conceito da camada de predição operacional (ICOH) e o esqueleto da API de análise radiológica.

```
radia-triage-platform/
├── app/
│   ├── icoh_simulation.py     # Simulação do ICOH com dados sintéticos
│   ├── icoh_model.py          # Lógica do índice e predição 6h
│   └── api.py                 # FastAPI — endpoints da plataforma
├── notebooks/
│   └── icoh_demo.ipynb        # Demonstração do índice
├── docs/
│   ├── arquitetura.md
│   └── icoh.md
└── requirements.txt
```

---

## Rodar a simulação do ICOH

```bash
pip install -r requirements.txt
python app/icoh_simulation.py
```

A simulação gera 72 horas de dados operacionais sintéticos de um pronto-socorro (ocupação de leitos, exames críticos pendentes, tempo médio de espera, volume de admissões) e calcula o ICOH hora a hora, com predição rolante para as próximas 6 horas. Saída visual em `assets/icoh_output.png`.

---

## Roadmap

| Fase | Entrega | Status |
|------|---------|--------|
| 1 | Simulação do ICOH com dados sintéticos | ✓ |
| 2 | API REST com endpoints de análise e predição | em progresso |
| 3 | Integração com modelo pré-treinado para RX de tórax | planejado |
| 4 | Camada de RAG sobre achados radiológicos | planejado |
| 5 | Dashboard operacional integrado | planejado |
| 6 | Piloto com dataset público (NIH ChestX-ray14) | planejado |

---

## Sobre o ICOH

O Índice de Colapso Operacional Hospitalar combina seis variáveis em uma métrica única no intervalo de 0 a 100:

- Taxa de ocupação de leitos
- Volume de exames críticos pendentes
- Tempo médio de espera vs. baseline histórico
- Fluxo de admissões nas últimas 2 horas
- Sazonalidade epidemiológica
- Severidade média dos achados radiológicos recentes

A predição para o horizonte de 6 horas é feita por regressão sobre a série temporal das próprias variáveis, com pesos calibráveis por unidade hospitalar.

A integração entre achados radiológicos (severidade) e indicadores operacionais é o que diferencia o ICOH de modelos existentes de predição de superlotação, que tratam apenas o fluxo de pacientes.

Detalhes técnicos completos em [`docs/icoh.md`](docs/icoh.md).

---

## Tecnologias

Python 3.11 · FastAPI · NumPy · Pandas · Matplotlib · scikit-learn · PyTorch (planejado para RX)

---

## Licença

Uso acadêmico. Todos os direitos sobre o conceito do ICOH reservados aos autores.
