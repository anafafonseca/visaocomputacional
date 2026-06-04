"""
RadIA — API REST

Endpoints da plataforma. Esta e a especificacao inicial; a integracao
com modelos de visao computacional sera adicionada na proxima fase.

Para rodar:
    uvicorn app.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from icoh_model import (
    HospitalState,
    compute_icoh,
    predict_icoh_6h,
    classify_icoh,
    recommend_actions,
)


app = FastAPI(
    title="RadIA Triage Platform",
    description="Plataforma de inteligencia hospitalar baseada em IA",
    version="0.1.0",
)


class HospitalStateInput(BaseModel):
    occupancy: float
    pending_exams: float
    wait_ratio: float
    admission_flow: float
    epidemic_factor: float
    radiology_severity: float


class ICOHResponse(BaseModel):
    icoh: float
    classification: str
    recommendations: List[str]


class PredictionRequest(BaseModel):
    history: List[HospitalStateInput]


class PredictionResponse(BaseModel):
    current_icoh: float
    current_classification: str
    forecast_6h: List[float]
    forecast_classifications: List[str]


@app.get("/")
def root():
    return {
        "service": "RadIA Triage Platform",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "/icoh": "Calcula ICOH a partir do estado hospitalar atual",
            "/forecast": "Predicao do ICOH para as proximas 6 horas",
            "/health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/icoh", response_model=ICOHResponse)
def calculate_icoh(state_input: HospitalStateInput):
    """Calcula o ICOH a partir do estado operacional atual."""
    state = HospitalState(**state_input.dict())
    icoh_value = compute_icoh(state)
    return ICOHResponse(
        icoh=icoh_value,
        classification=classify_icoh(icoh_value),
        recommendations=recommend_actions(icoh_value, state),
    )


@app.post("/forecast", response_model=PredictionResponse)
def forecast_icoh(request: PredictionRequest):
    """Predicao do ICOH para as proximas 6 horas."""
    if len(request.history) < 6:
        raise HTTPException(
            status_code=400,
            detail="Historico precisa ter pelo menos 6 horas de dados",
        )

    states = [HospitalState(**s.dict()) for s in request.history]
    current = compute_icoh(states[-1])

    try:
        forecast = predict_icoh_6h(states)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(
        current_icoh=current,
        current_classification=classify_icoh(current),
        forecast_6h=forecast,
        forecast_classifications=[classify_icoh(v) for v in forecast],
    )
