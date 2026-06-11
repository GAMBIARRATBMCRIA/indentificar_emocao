from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
    texto: str = Field(..., min_length=1, max_length=5000, description="Texto para análise de emoção em PT-BR")

class EmotionDetail(BaseModel):
    emocao: str = Field(..., description="Nome da emoção")
    confianca: float = Field(..., description="Probabilidade (0.0 a 1.0)")

class PredictResponse(BaseModel):
    emocoes_detectadas: list[EmotionDetail] = Field(..., description="Lista de emoções detectadas que ultrapassaram o limiar (threshold)")
    probabilidades: Dict[str, float] = Field(..., description="Dicionário com todas as emoções e suas respectivas probabilidades cruas")
