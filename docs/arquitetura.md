# Arquitetura do RadIA Triage Platform

## Visão geral

O RadIA é organizado em três camadas independentes que se comunicam por uma API REST central:

```
┌────────────────────────────────────────────────────────────────┐
│                  Camada de apresentação                        │
│            (interface médica + dashboard operacional)          │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│                       API REST (FastAPI)                       │
└────────────────────────────────────────────────────────────────┘
       │                      │                      │
┌──────────────┐      ┌──────────────┐      ┌────────────────┐
│   Camada 1   │      │   Camada 2   │      │    Camada 3    │
│    Visão     │      │      IA      │      │   Operacional  │
│ Computacional│      │  Generativa  │      │     (ICOH)     │
└──────────────┘      └──────────────┘      └────────────────┘
       │                      │                      ▲
       └──────────────────────┴──────────────────────┘
              sinal radiológico alimenta o ICOH
```

## Camada 1 — Visão Computacional

**Entrada:** imagens no padrão DICOM ou PNG/JPEG.

**Modelo planejado:** TorchXRayVision ou CheXNet (modelos pré-treinados no NIH ChestX-ray14 dataset) para classificação de 14 patologias torácicas.

**Saída:**
- Probabilidades por patologia
- Score de severidade global (0–1) — alimenta o ICOH
- Mapa de calor (Grad-CAM) sobre a imagem
- Score de confiança do modelo

## Camada 2 — IA Generativa

**Entrada:** achados da camada 1 + pergunta do médico em linguagem natural.

**Abordagem:** RAG (Retrieval-Augmented Generation) onde os "documentos" recuperados são os achados estruturados do exame em análise, não conhecimento médico genérico.

**Saída:** resposta contextualizada à imagem específica, sem alucinação sobre dados que não estão no exame.

## Camada 3 — Operacional (ICOH)

**Entrada:** indicadores operacionais do HIS/RIS + severidade radiológica vinda da camada 1.

**Processamento:** cálculo do ICOH (ver `icoh.md`) e predição rolante para 6 horas.

**Saída:**
- Valor atual do índice
- Predição para próximas 6 horas
- Recomendações operacionais
- Alertas para gestão hospitalar

## Stack tecnológico

| Componente | Tecnologia | Por quê |
|------------|------------|---------|
| API | FastAPI | Tipagem nativa, geração automática de OpenAPI, performance |
| Modelos de imagem | PyTorch / TorchXRayVision | Modelos pré-treinados open-source no domínio |
| Predição operacional | NumPy / scikit-learn | Cálculo do ICOH e regressões |
| Storage de imagens | AWS S3 ou MinIO local | Padrão para dados volumosos |
| Banco operacional | PostgreSQL | Histórico de exames e indicadores |
| Cache | Redis | Filas e estados temporários |
| Frontend | React + TailwindCSS | Componentes médicos reutilizáveis |
| Dashboard | Microsoft Power BI | Padrão de mercado para gestão hospitalar |
| Conformidade | LGPD, HL7 FHIR, DICOM 3.0 | Interoperabilidade hospitalar |

## Conformidade regulatória

A plataforma é projetada desde o início para operar no marco regulatório aplicável a sistemas de auxílio ao diagnóstico médico:

- **LGPD** — anonimização de imagens DICOM (remoção de metadados PHI)
- **CFM Resolução 2.299/2021** — IA como suporte ao médico, nunca substituta
- **HL7 FHIR R4** — interoperabilidade com HIS/RIS/PACS
- **DICOM 3.0** — formato padrão para ingestão de imagens

## Próximos passos técnicos

1. Integração da camada 1 com TorchXRayVision usando o dataset NIH ChestX-ray14
2. Pipeline DICOM → análise → severidade → ICOH end-to-end
3. Endpoint FastAPI para upload de imagem e retorno do laudo + score
4. Dashboard mínimo em Streamlit para visualização do ICOH em tempo real
