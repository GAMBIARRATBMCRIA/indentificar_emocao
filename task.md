# Implementação: Migração para o BERTimbau

- [ ] Atualizar `requirements.txt` com `transformers` e `tf-keras`.
- [ ] Atualizar `model/config.py` com `MODEL_CHECKPOINT`.
- [ ] Refatorar `model/train.py` para usar `AutoTokenizer` e `TFAutoModelForSequenceClassification`.
- [ ] Refatorar `model/evaluate.py` para carregar o modelo Hugging Face.
- [ ] Refatorar `backend/predictor.py` para servir o novo modelo BERT.
- [ ] Opcional: Atualizar `setup.bat` para lidar com a instalação limpa se necessário (não faremos agora, basta rodar o pip install).
