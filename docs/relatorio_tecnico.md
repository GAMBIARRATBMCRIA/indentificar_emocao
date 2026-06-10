# Relatório Técnico: Reconhecimento de Emoções em PT-BR usando Deep Learning

## 1. Contextualização e Objetivo
Este projeto tem como objetivo desenvolver uma plataforma web capaz de identificar automaticamente as emoções predominantes em comentários e textos escritos em Português Brasileiro (PT-BR). A aplicação visa demonstrar todo o ciclo de vida de um modelo de Machine Learning, desde a ingestão do dataset até o deploy via API REST.

## 2. Descrição do Dataset
Para o treinamento, foi utilizado o dataset **BRIGHTER** (`brighter-dataset/BRIGHTER-emotion-categories`, configuração `ptbr`), disponível publicamente no HuggingFace. 
Diferente de abordagens comuns que traduzem datasets estrangeiros (como o `dair-ai/emotion`), o BRIGHTER possui anotações nativas de brasileiros feitas por humanos (crowdsourcing). 
A taxonomia adotada possui 6 classes básicas de emoção baseadas em Ekman: `anger` (Raiva), `disgust` (Nojo), `fear` (Medo), `joy` (Alegria), `sadness` (Tristeza) e `surprise` (Surpresa).

No pré-processamento original, os dados multi-label (com escores para diversas emoções) foram convertidos para classificação single-label utilizando a função `argmax()`.

## 3. Pré-processamento
O pipeline de pré-processamento construído no arquivo `model/preprocess.py` é sensível às nuances do português:
- **Limpeza Textual**: Utilização de expressões regulares para manter apenas letras alfabéticas. Diferente do padrão inglês, o regex `[^a-záàâãéêíóôõúüç\s]` foi desenhado especificamente para preservar os acentos essenciais da língua portuguesa.
- **Tokenização e Padding**: O texto processado é transformado em sequências inteiras com `vocab_size` fixado em 20.000 tokens (ou menos, se o dataset não alcançar). Fez-se o padding limitando as sentenças ao tamanho de 128 (padrão usual para frases curtas da internet).

## 4. Arquitetura do Modelo
O classificador de textos usa uma arquitetura sequencial (Keras) simples e efetiva:
1. **Embedding**: Camada de incorporação convertendo os vetores de inteiros discretos para representações contínuas densas (`dim=128`).
2. **Bidirectional LSTM**: Uma rede neural recorrente de longo prazo computando `units=64` em dois sentidos espaciais. Isso permite extrair contexto tanto prececendo a palavra alvo, como palavras futuras na sentença.
3. **Dropout Layers**: Taxas de `0.4` e `0.3` foram adicionadas nas camadas intermediárias para penalizar o sobreajuste (overfitting).
4. **Dense e Softmax**: As saídas chegam numa camada densa de tamanho 32 com `ReLU`, e a última camada distribui probabilidade entre 6 neurônios através da ativação `Softmax`.

## 5. Treinamento e Métricas
O modelo foi compilado utilizando o otimizador Adam (learning rate = 0.001) e o avaliador de perda `SparseCategoricalCrossentropy`. 
Utilizamos `EarlyStopping` (com monitoração da validação e paciência de 3 épocas) para otimizar os 10 ciclos de épocas iniciais, restaurando os melhores pesos atingidos durante as passadas de treinamento.

**Avaliação**: Os resultados brutos das métricas (Accuracy, Precision, Recall e F1-Score) são gravados de modo determinístico durante o comando de avaliação `python evaluate.py`, em conjunto com as matrizes de confusão que mapeiam estatisticamente os falsos-positivos e falsos-negativos para as 6 categorias.

## 6. Análise Crítica do Sistema
- **Separação de Preocupações**: A arquitetura do sistema foi isolada em 3 grandes diretórios: `model` (que é dependente do Tensorflow), `backend` (que utiliza FastAPI e Pydantic para validação robusta) e `frontend`. Isso garante a escalabilidade.
- **Fallbacks (Tolerância a Falhas)**: A API contém uma implementação Singleton robusta, onde no caso de os arquivos `.keras` ou `.json` estarem ausentes, a plataforma continua acessível mas negando previsões através do status `HTTP 503`, prevenindo acidentes fatais em produção.
- **Limitações**: Frases sarcásticas continuam sendo um desafio inerente ao PLN. Além disso, a capacidade semântica para textos complexos poderia ser ampliada com a troca dos Embeddings LSTM pelo ecossistema de Transformers (como BERTimbau).
