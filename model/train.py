import os
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config
from preprocess import clean_text, fit_tokenizer, preprocess_texts, save_tokenizer

def convert_multilabel_to_single(example):
    """
    Converte as anotações multi-label do BRIGHTER para single-label via argmax.
    O dataset tem as labels ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'].
    """
    scores = [
        example['anger'],
        example['disgust'],
        example['fear'],
        example['joy'],
        example['sadness'],
        example['surprise']
    ]
    example['label'] = np.argmax(scores)
    return example

def build_model(vocab_size, max_seq_length, embedding_dim, lstm_units, dense_units, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
        Bidirectional(LSTM(units=lstm_units, return_sequences=False)),
        Dropout(config.DROPOUT_RATE_LSTM),
        Dense(units=dense_units, activation='relu'),
        Dropout(config.DROPOUT_RATE_DENSE),
        Dense(units=num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("1. Baixando dataset BRIGHTER (ptbr)...")
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "ptbr")
    
    print("2. Convertendo labels para single-label (argmax)...")
    dataset = dataset.map(convert_multilabel_to_single)
    
    train_texts = dataset['train']['text']
    train_labels = np.array(dataset['train']['label'])
    
    val_texts = dataset['dev']['text']
    val_labels = np.array(dataset['dev']['label'])
    
    test_texts = dataset['test']['text']
    test_labels = np.array(dataset['test']['label'])
    
    print("3. Limpando os textos e treinando o tokenizer...")
    cleaned_train_texts = [clean_text(t) for t in train_texts]
    tokenizer = fit_tokenizer(cleaned_train_texts)
    
    # Salvar o tokenizer
    save_tokenizer(tokenizer, config.TOKENIZER_PATH)
    print(f"Tokenizer salvo em {config.TOKENIZER_PATH}")
    
    # Atualizar o vocab size real (usar o min entre o definido e o tamanho do word_index)
    actual_vocab_size = min(config.VOCAB_SIZE, len(tokenizer.word_index) + 1)
    
    print("4. Tokenizando e aplicando padding...")
    X_train = preprocess_texts(train_texts, tokenizer)
    X_val = preprocess_texts(val_texts, tokenizer)
    # X_test = preprocess_texts(test_texts, tokenizer) # A avaliação completa será feita no evaluate.py
    
    y_train = train_labels
    y_val = val_labels
    
    print("5. Construindo o modelo...")
    model = build_model(
        vocab_size=actual_vocab_size,
        max_seq_length=config.MAX_SEQ_LENGTH,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        num_classes=config.NUM_CLASSES
    )
    model.summary()
    
    print("6. Treinando o modelo...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=config.MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )
    
    print(f"Treinamento concluído. Modelo salvo em {config.MODEL_PATH}")

if __name__ == "__main__":
    main()
