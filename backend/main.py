from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest, PredictResponse
from predictor import predictor

app = FastAPI(
    title="Reconhecimento de Emoções em Texto",
    description="API para classificar emoções predominantes em textos em PT-BR usando Deep Learning",
    version="1.0.0"
)

# Configuração de CORS (permitir chamadas do frontend em localhost)
origins = [
    "http://localhost:5500", 
    "http://127.0.0.1:5500", 
    "http://localhost:8000",
    "*" # Permite qualquer origem para facilitar o desenvolvimento local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict_emotion(request: PredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Serviço Indisponível: O modelo de IA não foi carregado. Certifique-se de executar o treinamento primeiro."
        )
    
    try:
        resultado = predictor.predict(request.texto)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno durante a predição: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modelo_carregado": predictor.is_loaded,
        "versao": "1.0.0"
    }
