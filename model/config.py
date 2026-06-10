import os

# Caminhos absolutos/relativos baseados na pasta model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.json")

# Garantir que os diretórios existam
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Parâmetros de Treinamento e Modelo
VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 128
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE_LSTM = 0.4
DENSE_UNITS = 32
DROPOUT_RATE_DENSE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32

# Labels (Classes na ordem correspondente aos índices 0-5)
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)

# Mapa de emoções PT-BR para uso no frontend/logs
EMOTION_MAP_PTBR = {
    'anger': 'Raiva',
    'disgust': 'Nojo',
    'fear': 'Medo',
    'joy': 'Alegria',
    'sadness': 'Tristeza',
    'surprise': 'Surpresa'
}
