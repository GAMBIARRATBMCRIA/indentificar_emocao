import os
import sys
import numpy as np

# Adicionar a pasta raiz ao sys.path para poder importar a pasta model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config
from model.preprocess import preprocess_texts, load_tokenizer

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.thresholds = {
            'anger': 0.20,
            'disgust': 0.15,
            'fear': 0.15,
            'joy': 0.20,
            'sadness': 0.20,
            'surprise': 0.15
        }
        self.load_artifacts()

    def load_artifacts(self):
        """Carrega o modelo Keras e o Tokenizer, se existirem."""
        try:
            if os.path.exists(config.MODEL_PATH) and os.path.exists(config.TOKENIZER_PATH):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
                self.tokenizer = load_tokenizer(config.TOKENIZER_PATH)
                self.is_loaded = True
                print("Modelos carregados com sucesso!")
            else:
                print("Artefatos não encontrados. Modo de predição desativado.")
        except Exception as e:
            print(f"Erro ao carregar os artefatos: {str(e)}")

    def predict(self, text: str) -> dict:
        """Recebe um texto, pré-processa, faz a inferência e retorna o resultado formatado."""
        if not self.is_loaded:
            raise RuntimeError("O modelo ainda não foi treinado ou os artefatos não foram encontrados.")

        # O pré-processamento espera uma lista de textos
        X = preprocess_texts([text], self.tokenizer)
        
        # Predição
        y_pred_probs = self.model.predict(X)[0] # Pega o primeiro item do batch
        
        # Montar o dicionário de probabilidades
        prob_dict = {
            emotion: float(prob) 
            for emotion, prob in zip(config.EMOTION_LABELS, y_pred_probs)
        }
        
        # Filtrar as emoções que ultrapassaram o threshold
        emocoes_detectadas = []
        max_emotion = None
        max_prob = -1.0
        
        for emotion, prob in prob_dict.items():
            if prob > max_prob:
                max_prob = prob
                max_emotion = emotion
                
            if prob >= self.thresholds.get(emotion, 0.20):
                emocoes_detectadas.append({"emocao": emotion, "confianca": prob})
        
        # Fallback: Se nenhuma emoção passar do threshold, retornamos a que tem a maior probabilidade
        # em vez de retornar "Neutro".
        if not emocoes_detectadas and max_emotion is not None:
            emocoes_detectadas.append({"emocao": max_emotion, "confianca": max_prob})
            
        # Ordenar as detectadas por confiança (decrescente)
        emocoes_detectadas.sort(key=lambda x: x["confianca"], reverse=True)
        
        return {
            "emocoes_detectadas": emocoes_detectadas,
            "probabilidades": prob_dict
        }

# Instância Singleton
predictor = EmotionPredictor()
