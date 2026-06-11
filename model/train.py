import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config
from preprocess import clean_text

EMOTION_NAMES = [
    "anger", "disgust", "fear", "joy", "sadness", "surprise"
]

def convert_to_multilabel_array(example):
    labels = [
        example["anger"],
        example["disgust"],
        example["fear"],
        example["joy"],
        example["sadness"],
        example["surprise"]
    ]
    example["label"] = labels
    return example

def show_dataset_statistics(y_train):
    print("\nDistribuição das emoções:\n")
    totals = y_train.sum(axis=0)
    for emotion, total in zip(EMOTION_NAMES, totals):
        print(f"{emotion:10s}: {int(total)}")
    print("\nTotal de exemplos:", len(y_train))

def calculate_class_weights(y_train):
    totals = y_train.sum(axis=0)
    weights = len(y_train) / (len(totals) * np.maximum(totals, 1))
    print("\nPesos calculados:")
    for emotion, weight in zip(EMOTION_NAMES, weights):
        print(f"{emotion:10s}: {weight:.4f}")
    return weights

def weighted_binary_crossentropy(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # BERT output layer uses linear activation (logits). We must apply sigmoid.
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss_pos = y_true * tf.math.log(y_pred) * class_weights
        loss_neg = (1 - y_true) * tf.math.log(1 - y_pred)
        loss = -(loss_pos + loss_neg)
        return tf.reduce_mean(loss)
    return loss

def create_tf_dataset(texts, labels, tokenizer, batch_size, shuffle=False):
    cleaned_texts = [clean_text(text) for text in texts]
    encodings = tokenizer(
        cleaned_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=config.MAX_SEQ_LENGTH, 
        return_tensors='tf'
    )
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    print("1. Baixando dataset...")
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "ptbr")

    print("2. Convertendo labels...")
    dataset = dataset.map(convert_to_multilabel_array)

    train_texts = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])
    val_texts = dataset["dev"]["text"]
    val_labels = np.array(dataset["dev"]["label"])

    show_dataset_statistics(train_labels)
    class_weights = calculate_class_weights(train_labels)

    print("\n3. Carregando Tokenizer do HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    
    # Salvar tokenizer para uso posterior
    tokenizer.save_pretrained(config.HF_MODEL_DIR)

    print("\n4. Tokenizando datasets...")
    train_dataset = create_tf_dataset(train_texts, train_labels, tokenizer, config.BATCH_SIZE, shuffle=True)
    val_dataset = create_tf_dataset(val_texts, val_labels, tokenizer, config.BATCH_SIZE, shuffle=False)

    print("\n5. Carregando modelo BERTimbau...")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=config.NUM_CLASSES,
        problem_type="multi_label_classification"
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(class_weights),
        metrics=[
            "binary_accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    print("\n6. Treinando...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTreinamento concluído. Salvando modelo...")
    model.save_pretrained(config.HF_MODEL_DIR)
    print(f"Modelo salvo em: {config.HF_MODEL_DIR}")

if __name__ == "__main__":
    main()