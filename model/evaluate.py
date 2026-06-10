import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
from preprocess import preprocess_texts, load_tokenizer

def convert_multilabel_to_single(example):
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

def main():
    print("1. Carregando modelo e tokenizer...")
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.TOKENIZER_PATH):
        print(f"Erro: Artefatos não encontrados. Execute o train.py primeiro.")
        return
        
    model = tf.keras.models.load_model(config.MODEL_PATH)
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    
    print("2. Baixando/Carregando dataset de teste...")
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "ptbr")
    dataset = dataset.map(convert_multilabel_to_single)
    
    test_texts = dataset['test']['text']
    y_true = np.array(dataset['test']['label'])
    
    print("3. Pré-processando dados de teste...")
    X_test = preprocess_texts(test_texts, tokenizer)
    
    print("4. Realizando predições...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("5. Gerando relatórios de métricas...")
    report = classification_report(y_true, y_pred, target_names=config.EMOTION_LABELS)
    print("\nClassification Report:\n")
    print(report)
    
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Relatório salvo em {report_path}")
    
    print("6. Gerando e salvando matriz de confusão...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.EMOTION_LABELS, 
                yticklabels=config.EMOTION_LABELS)
    plt.title('Matriz de Confusão - Reconhecimento de Emoções')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusão salva em {cm_path}")

if __name__ == "__main__":
    main()
