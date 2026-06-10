# Projeto: Plataforma de Reconhecimento de Emoções em Texto utilizando Deep Learning

## Objetivo

Desenvolver uma aplicação web capaz de identificar automaticamente emoções predominantes em comentários textuais enviados pelo usuário.

O projeto deve atender aos requisitos da disciplina de Deep Learning e demonstrar todas as etapas de um experimento real:

* utilização de dataset público e replicável;
* preparação dos dados;
* treinamento de modelo de Deep Learning;
* validação do modelo;
* análise dos resultados;
* disponibilização de uma aplicação funcional.

---

# Problema

Uma plataforma deseja identificar automaticamente emoções presentes em comentários de usuários.

A aplicação deverá:

1. Receber um texto digitado pelo usuário;
2. Aplicar um modelo de Deep Learning treinado previamente;
3. Identificar a emoção predominante;
4. Exibir as probabilidades de todas as emoções possíveis.

---

# Dataset

## Dataset escolhido

Emotion Dataset

Fonte:

https://huggingface.co/datasets/dair-ai/emotion

Características:

* Dataset público;
* Gratuito;
* Amplamente utilizado em pesquisas acadêmicas;
* Possui textos rotulados com emoções.

Classes disponíveis:

* sadness
* joy
* love
* anger
* fear
* surprise

Exemplo:

Texto:
"I feel very happy today"

Classe:
joy

---

# Tecnologias Obrigatórias

## Frontend

Utilizar:

* HTML5
* CSS3
* JavaScript puro

Frameworks CSS opcionais:

* Bootstrap

Não utilizar frameworks SPA complexos (React, Angular ou Vue), salvo necessidade técnica justificada.

---

## Backend

Utilizar:

* Python 3.11+

Framework recomendado:

* Flask

Alternativa aceita:

* FastAPI

---

## Deep Learning

Utilizar:

* TensorFlow/Keras

Bibliotecas auxiliares:

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn

---

# Arquitetura do Sistema

Frontend HTML/CSS/JS
↓
API Python (Flask)
↓
Modelo Deep Learning
↓
Resposta JSON
↓
Exibição do resultado

---

# Funcionalidades da Aplicação

## Tela Principal

Possuir:

* campo de texto para inserção do comentário;
* botão "Analisar Emoção".

Exemplo:

---

Digite seu comentário:

[ textarea ]

## [ Analisar Emoção ]

---

## Resultado

Após análise, exibir:

* emoção predominante;
* nível de confiança;
* probabilidades de todas as classes.

Exemplo:

Emoção predominante:
JOY

Probabilidades:

Joy ............. 92%
Love ............ 3%
Surprise ........ 2%
Fear ............ 1%
Anger ........... 1%
Sadness ......... 1%

---

# Estrutura do Modelo

Arquitetura sugerida:

Entrada
↓
Embedding
↓
Bidirectional LSTM
↓
Dense
↓
Softmax

Motivo:

* adequada para classificação textual;
* simples de explicar academicamente;
* atende ao requisito de Deep Learning.

---

# Pré-processamento

Implementar:

1. Conversão para minúsculas;
2. Remoção de caracteres especiais;
3. Tokenização;
4. Padding das sequências;
5. Transformação texto → sequência numérica.

Utilizar:

* Tokenizer do Keras;
* pad_sequences.

---

# Treinamento

Divisão dos dados:

* Treino: 70%
* Validação: 15%
* Teste: 15%

Parâmetros iniciais:

* Epochs: 10
* Batch Size: 32
* Optimizer: Adam
* Loss: SparseCategoricalCrossentropy

Esses valores podem ser ajustados após testes.

---

# Métricas Obrigatórias

Calcular:

* Accuracy
* Precision
* Recall
* F1-score

Opcional:

* Matriz de Confusão
* Curvas de treinamento

---

# API

## Endpoint de Predição

POST /predict

Entrada:

{
"texto": "Estou muito feliz hoje"
}

Saída:

{
"emocao": "joy",
"confianca": 0.92,
"probabilidades": {
"joy": 0.92,
"love": 0.03,
"surprise": 0.02,
"fear": 0.01,
"anger": 0.01,
"sadness": 0.01
}
}

---

# Estrutura de Pastas

projeto/

├── backend/
│   ├── app.py
│   ├── train.py
│   ├── model/
│   ├── dataset/
│   └── tokenizer/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│
├── docs/
│
└── README.md

---

# Entregáveis

O projeto deve produzir:

1. Aplicação web funcional;
2. Modelo treinado;
3. Código-fonte completo;
4. Relatório técnico contendo:

   * contextualização;
   * objetivo;
   * descrição do dataset;
   * arquitetura do modelo;
   * treinamento;
   * métricas;
   * resultados;
   * análise crítica.

---

# Critérios de Qualidade

Os agentes devem priorizar:

1. Clareza do código;
2. Organização em camadas;
3. Facilidade de manutenção;
4. Reprodutibilidade do treinamento;
5. Documentação adequada;
6. Explicabilidade dos resultados.

O foco principal do projeto é Deep Learning aplicado a processamento de linguagem natural (NLP). A aplicação web existe para demonstrar o funcionamento do modelo treinado.
