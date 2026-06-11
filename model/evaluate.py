import os
import numpy as np
import tensorflow as tf

from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

import config
from preprocess import (
    preprocess_texts,
    load_tokenizer
)


def convert_to_multilabel_array(example):
    example["label"] = [
        example["anger"],
        example["disgust"],
        example["fear"],
        example["joy"],
        example["sadness"],
        example["surprise"]
    ]
    return example


def save_confusion_matrix(y_true, y_pred):

    cm = multilabel_confusion_matrix(
        y_true,
        y_pred
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes = axes.ravel()

    for i, (matrix, label) in enumerate(
        zip(cm, config.EMOTION_LABELS)
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False
        )

        axes[i].set_title(
            f"Classe: {label}"
        )

        axes[i].set_ylabel(
            "Verdadeiro"
        )

        axes[i].set_xlabel(
            "Predito"
        )

    plt.tight_layout()

    output_path = os.path.join(
        config.OUTPUTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nMatriz salva em:\n{output_path}"
    )


def evaluate_threshold(
    threshold,
    y_true,
    y_pred_probs
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"THRESHOLD = {threshold}"
    )

    print(
        "=" * 70
    )

    y_pred = (
        y_pred_probs > threshold
    ).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=config.EMOTION_LABELS,
        zero_division=0
    )

    print(report)

    return y_pred, report


def main():

    print(
        "1. Carregando modelo e tokenizer..."
    )

    if (
        not os.path.exists(config.MODEL_PATH)
        or not os.path.exists(
            config.TOKENIZER_PATH
        )
    ):
        print(
            "Erro: execute train.py primeiro."
        )
        return

    model = tf.keras.models.load_model(
        config.MODEL_PATH,
        compile=False
    )

    tokenizer = load_tokenizer(
        config.TOKENIZER_PATH
    )

    print(
        "2. Carregando dataset..."
    )

    dataset = load_dataset(
        "brighter-dataset/BRIGHTER-emotion-categories",
        "ptbr"
    )

    dataset = dataset.map(
        convert_to_multilabel_array
    )

    test_texts = dataset["test"]["text"]

    y_true = np.array(
        dataset["test"]["label"]
    )

    print(
        "3. Pré-processando..."
    )

    X_test = preprocess_texts(
        test_texts,
        tokenizer
    )

    print(
        "4. Realizando predições..."
    )

    y_pred_probs = model.predict(
        X_test,
        verbose=1
    )

    print("\nEstatísticas:")

    print(
        f"Min:  {y_pred_probs.min():.4f}"
    )

    print(
        f"Max:  {y_pred_probs.max():.4f}"
    )

    print(
        f"Mean: {y_pred_probs.mean():.4f}"
    )

    print(
        "\nPrimeiras probabilidades:"
    )

    print(y_pred_probs[:5])

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.40,
        0.50
    ]

    best_report = None
    best_preds = None

    for threshold in thresholds:

        preds, report = evaluate_threshold(
            threshold,
            y_true,
            y_pred_probs
        )

        if threshold == 0.30:
            best_report = report
            best_preds = preds

    report_path = os.path.join(
        config.OUTPUTS_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(best_report)

    print(
        f"\nRelatório salvo em:\n{report_path}"
    )

    save_confusion_matrix(
        y_true,
        best_preds
    )

    print(
        "\nAvaliação concluída."
    )


if __name__ == "__main__":
    main()