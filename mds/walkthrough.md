# Walkthrough — Plataforma SenseAI 🚀

O projeto de Deep Learning para **Reconhecimento de Emoções em Texto (PT-BR)** foi totalmente implementado, da coleta e processamento de dados ao visual final na web!

## O que foi construído

1. **Pipeline de Machine Learning (`model/`)**:
   - Criação de scripts organizados: `config.py`, `preprocess.py`, `train.py` e `evaluate.py`.
   - Utilizamos o dataset nativo PT-BR **BRIGHTER** via biblioteca do HuggingFace, convertendo dados multi-label em single-label.
   - O pipeline aplica uma limpeza que respeita acentuações da língua portuguesa, tokeniza via Keras e prepara tensores para a LSTM Bidirecional.

2. **Backend e API (`backend/`)**:
   - Construída inteiramente em **FastAPI** para alta performance e validações estritas de schema (`schemas.py`).
   - Ponto fortíssimo: a API possui mecanismo **graceful fallback**. Caso você decida treinar o modelo amanhã, a API não irá quebrar hoje. Ela aceita as requisições de Front-End e as responde com um Error 503 explicativo (`"O modelo ainda precisa ser treinado."`).

3. **Frontend Vanilla com Glassmorphism (`frontend/`)**:
   - Dispensa de SPA! Usamos HTML5/CSS3/JS limpos.
   - Design espetacular de "fundo fosco", cores semânticas vibrantes (Alegria = Amarelo, Raiva = Vermelho, etc.), e micro-animações nas barras de progresso (Width Transitions) que surpreendem e dão responsividade instantânea à tela.

4. **Documentação Acadêmica (`docs/` e `README.md`)**:
   - Relatório descritivo com contexto de escolhas arquiteturais.
   - API Reference providenciando contratos JSON estritos.

## Como Executar
O `README.md` localizado na raiz do projeto (e também exibido ao lado) possui as quatro etapas necessárias para ligar toda a máquina (Instalação, Treinamento, Uvicorn Server, e Live Server Frontend). 

Siga-o de perto quando estiver no seu ambiente local (ou na máquina onde for treinar o `emotion_model.keras`).

---
✨ **Tudo pronto! Seu projeto já pode ser treinado e exibido.**
