import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
try:
    from model import config
except ImportError:
    import config

def clean_text(text: str) -> str:
    """
    Limpa o texto convertendo para minúsculas e removendo caracteres especiais,
    mas preservando os acentos do português brasileiro.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remover pontuação e números, preservar letras (incluindo acentuadas) e espaços
    # [^a-záàâãéêíóôõúüç\s] significa: substitua o que NÃO for letra minúscula (com/sem acento) ou espaço
    text = re.sub(r'[^a-záàâãéêíóôõúüç\s]', ' ', text)
    
    # 3. Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def fit_tokenizer(texts: list[str]) -> Tokenizer:
    """
    Treina o tokenizer com os textos fornecidos.
    """
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def save_tokenizer(tokenizer: Tokenizer, filepath: str):
    """
    Salva o tokenizer em formato JSON.
    """
    tokenizer_json = tokenizer.to_json()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

def load_tokenizer(filepath: str) -> Tokenizer:
    """
    Carrega o tokenizer de um arquivo JSON.
    """
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(filepath, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)

def preprocess_texts(texts: list[str], tokenizer: Tokenizer) -> list:
    """
    Pipeline completo: limpa, tokeniza e faz padding em uma lista de textos.
    """
    cleaned_texts = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=config.MAX_SEQ_LENGTH, padding='post', truncating='post')
    return padded
