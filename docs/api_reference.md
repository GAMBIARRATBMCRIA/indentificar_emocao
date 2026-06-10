# API Reference — SenseAI Backend

A plataforma utiliza o **FastAPI** providenciando suporte a concorrência assíncrona, e se beneficiando do Pydantic para validação estrita dos contratos.

> **Base URL**: `http://localhost:8000`

---

## 1. Verificação de Saúde
Endpoint utilizado para instrumentação, conferência de estabilidade e verificação do status dos pesos da Inteligência Artificial.

### `GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "versao": "1.0.0"
}
```

---

## 2. Predição de Emoção
O endpoint vital do sistema que consome a arquitetura Deep Learning previamente treinada.

### `POST /predict`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "texto": "Esta conquista foi extremamente maravilhosa para todos nós!"
}
```

#### Regras de Validação Pydantic
| Campo | Tipo | Obrigatoriedade | Regra |
|---|---|---|---|
| `texto` | `string` | Obrigatório | Min: 1 carac. Max: 5000 carac. |

**Response — Sucesso (`200 OK`)**:
```json
{
  "emocao_predominante": "joy",
  "confianca": 0.9412,
  "probabilidades": {
    "anger": 0.005,
    "disgust": 0.012,
    "fear": 0.002,
    "joy": 0.9412,
    "sadness": 0.010,
    "surprise": 0.0298
  }
}
```

**Response — Erro Semântico/Validação (`422 Unprocessable Entity`)**:
Este erro ocorre caso a requisição omita a chave "texto", ou forneça textos acima do comprimento requisitado.
```json
{
  "detail": [
    {
      "loc": ["body", "texto"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Response — Modelo Indisponível (`503 Service Unavailable`)**:
Ocorre caso a API inicie, porém os artefatos `emotion_model.keras` e/ou `tokenizer.json` não existam ou não foram computados durante a etapa de treinamento.
```json
{
  "detail": "Serviço Indisponível: O modelo de IA não foi carregado. Certifique-se de executar o treinamento primeiro."
}
```
