from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
    texto: str = Field(..., min_length=1, max_length=5000, description="Texto para análise de emoção em PT-BR")

class PredictResponse(BaseModel):
    emocao_predominante: str = Field(..., description="A emoção com maior probabilidade identificada no texto")
    confianca: float = Field(..., description="Valor da maior probabilidade (0.0 a 1.0)")
    probabilidades: Dict[str, float] = Field(..., description="Dicionário com as 6 emoções e suas respectivas probabilidades")
