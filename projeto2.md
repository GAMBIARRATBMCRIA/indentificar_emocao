
# Especificação de Projeto: Plataforma de Reconhecimento de Emoções em Texto (Deep Learning)

## 1. Objetivo e Escopo
Desenvolver uma aplicação web de ponta a ponta capaz de identificar automaticamente emoções predominantes em comentários textuais. O projeto deve demonstrar a esteira completa de um experimento de Deep Learning: do pré-processamento matemático dos dados à disponibilização de uma interface de usuário fluida que consuma o modelo treinado via API.

---

## 2. Tecnologias Obrigatórias e Isolamento
Para evitar alucinações e conflitos de contexto, os agentes devem respeitar estritamente a divisão tecnológica e estrutural abaixo.

* **Frontend (Apresentação Visual):**
  * HTML5, CSS3 e JavaScript puro (ES6+).
  * Foco em uma arquitetura limpa, manipulação nativa do DOM (Fetch API) e componentização visual responsiva sem frameworks SPA complexos.
* **Backend (Servidor API):**
  * Python 3.11+ utilizando `FastAPI` (ou `Flask`, se estritamente necessário para compatibilidade).
* **Deep Learning (Treinamento e Modelagem):**
  * `TensorFlow/Keras` para a rede neural.
  * `pandas`, `numpy`, `scikit-learn` para manipulação vetorial e avaliação de métricas.

---

## 3. Dataset e Pré-processamento
* **Dataset Base:** Emotion Dataset (CARER - Hugging Face).
* **Classes (6 emoções):** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.
* **Pipeline de Pré-processamento Exigido:**
  1. Conversão para minúsculas e remoção de caracteres especiais.
  2. Tokenização (mapeamento de texto para sequência numérica).
  3. Padding das sequências para garantir a padronização das matrizes de entrada.

---

## 4. Estrutura do Modelo e Hiperparâmetros
O agente focado em dados deve implementar a seguinte arquitetura sequencial para garantir o correto cálculo probabilístico:

1. **Entrada**
2. **Embedding:** Representação vetorial densa das palavras.
3. **Bidirectional LSTM:** Para capturar dependências sequenciais passadas e futuras.
4. **Dense:** Com regularização (Dropout) se necessário.
5. **Softmax (Saída):** Camada final com 6 neurônios para gerar uma distribuição de probabilidades que some exatamente 100%.

**Regras Matemáticas de Treinamento (Hard-coded):**
* **Divisão dos Dados:** 70% Treino | 15% Validação | 15% Teste
* **Epochs:** 10
* **Batch Size:** 32
* **Optimizer:** Adam
* **Loss Function:** `SparseCategoricalCrossentropy` (ou `CategoricalCrossentropy`, dependendo do enconding das labels).

**Métricas Exigidas:** Accuracy, Precision, Recall, F1-score. (Opcional: Matriz de Confusão).

---

## 5. Fluxo da Interface de Usuário (Mockup e Requisitos)
O agente de frontend deve criar uma interface amigável e semântica, utilizando cores dinâmicas para as barras de progresso (ex: amarelo para alegria, azul para tristeza).

**Estrutura da Tela Principal:**
```text
[ Cabeçalho do Projeto ]

Digite seu comentário:
[ textarea longo e estilizado ]

[ Botão: Analisar Emoção ]

```

**Comportamento do Resultado (Renderização Dinâmica):**
Após a análise, a tela deve injetar os resultados abaixo, preferencialmente usando barras de preenchimento CSS proporcionais:

```text
Emoção predominante: JOY (92%)

Probabilidades Detalhadas:
Joy ............. [██████████████████  ] 92%
Love ............ [█                   ] 3%
Surprise ........ [                    ] 2%
Fear ............ [                    ] 1%
Anger ........... [                    ] 1%
Sadness ......... [                    ] 1%

```

---

## 6. Contrato de API (Endpoint)

O backend deve expor um endpoint para receber a requisição do frontend.

* **Rota:** `POST /predict`
* **Entrada (JSON):**
```json
{ "texto": "Estou muito feliz hoje" }

```


* **Saída Esperada (JSON):**
```json
{
  "emocao_predominante": "joy",
  "confianca": 0.92,
  "probabilidades": {
    "joy": 0.92, "love": 0.03, "surprise": 0.02,
    "fear": 0.01, "anger": 0.01, "sadness": 0.01
  }
}

```

## 7. Estrutura de Pastas Obrigatória

O código não deve ser misturado. Siga esta árvore de arquivos:

```text
projeto_emocoes/
├── model/
│   ├── train.py              # Isolado: carrega dados, treina, avalia e salva o modelo.
│   ├── model.keras           # Modelo treinado salvo (gerado após o treino).
│   └── tokenizer.json        # Vocabulário salvo.
├── backend/
│   ├── main.py               # Servidor FastAPI. Apenas carrega o modelo salvo e serve a API.
│   └── requirements.txt      
├── frontend/
│   ├── index.html            
│   ├── style.css             
│   └── app.js                # Lógica assíncrona para chamar o backend e montar o DOM.
└── README.md                 # Relatório final acadêmico do projeto.


