# Migração de Arquitetura: Do Zero para o BERTimbau (Transfer Learning)

Chegamos ao teto do que uma LSTM clássica pode fazer com 2.000 frases. O modelo é cego para palavras que não estavam no dataset original. A solução definitiva na indústria para isso é usar **Transfer Learning** (Aprendizado por Transferência).

A nova arquitetura proposta utilizará o **BERTimbau** (um modelo da família BERT pré-treinado especificamente em gigabytes de textos em Português do Brasil pela NeuralMind). 
Como o BERTimbau já leu quase toda a internet em português, ele já sabe que "repugnante" e "nojento" são a mesma coisa, mesmo que a palavra "repugnante" nunca tenha aparecido no seu dataset! Nós só precisaremos treinar a última camada dele para "ligar os pontos" entre as frases e as suas 6 emoções.

## Open Questions

> [!CAUTION]
> **Atenção ao Hardware!** Treinar um modelo BERT no Windows apenas com CPU (sem placa de vídeo configurada via WSL2 ou DirectML) pode levar **muitas horas** em vez de minutos. 
> Você está disposto a deixar o PC rodando por horas/deixar a ventoinha do PC gritar, ou prefere que eu adapte o código para você rodar no **Google Colab** de forma gratuita com uma GPU?

## Proposed Changes

A migração envolverá a troca do pacote puro do Keras pela biblioteca `transformers` da Hugging Face.

### Dependências
#### [MODIFY] [requirements.txt](file:///c:/Users/User/Desktop/projeto_ia/requirements.txt)
- Adicionar o pacote `transformers` e `tf-keras`.

### Configurações
#### [MODIFY] [model/config.py](file:///c:/Users/User/Desktop/projeto_ia/model/config.py)
- Adicionar `MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"`.
- Reduzir o `BATCH_SIZE` para 8 ou 16 (o BERT consome muita memória RAM).

### O Coração do Sistema
#### [MODIFY] [model/train.py](file:///c:/Users/User/Desktop/projeto_ia/model/train.py)
- Remover todo o fluxo do `Tokenizer` antigo do Keras.
- Instanciar o `AutoTokenizer.from_pretrained()` do HuggingFace.
- Trocar o modelo Keras sequencial pelo `TFAutoModelForSequenceClassification` configurado para 6 rótulos e com problema do tipo `multi_label_classification`.
- Manter a nossa `weighted_binary_crossentropy` que funcionou tão bem.

#### [MODIFY] [model/evaluate.py](file:///c:/Users/User/Desktop/projeto_ia/model/evaluate.py)
- Atualizar a forma de carregamento dos artefatos salvos (usar a API do HuggingFace em vez do Keras tradicional) para realizar as matrizes de confusão.

#### [MODIFY] [backend/predictor.py](file:///c:/Users/User/Desktop/projeto_ia/backend/predictor.py)
- Refatorar a classe para carregar o modelo BERT na memória da API FastAPI.

---

## Verification Plan

### Teste de Sanidade (CPU)
Rodar apenas 1 ou 2 épocas (epochs) no seu computador para garantir que o código de treinamento não quebra e que a loss começa a cair.

### Avaliação Completa
Rodar o `evaluate.py` após o treinamento (seja no seu PC ou no Colab) e verificar as matrizes de confusão. A expectativa é que o F1-Score do "Nojo" e "Surpresa" suba radicalmente sem destruir a precisão.
