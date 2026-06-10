# SenseAI — Plataforma de Reconhecimento de Emoções em Texto

SenseAI é uma aplicação web completa (Deep Learning + FastAPI + Vanilla JS) projetada para identificar a emoção predominante em textos escritos em **Português Brasileiro (PT-BR)**. 

O sistema utiliza um modelo baseado em **Bidirectional LSTM**, treinado sobre o dataset público **BRIGHTER** (nativamente em PT-BR), capaz de classificar 6 tipos de emoções: Alegria, Tristeza, Raiva, Medo, Surpresa e Nojo.

---

## 🚀 Como Executar o Projeto Passo a Passo

Siga o procedimento abaixo para preparar o ambiente, treinar o modelo e iniciar a aplicação.

### 1. Preparando o Ambiente

Certifique-se de ter o Python 3.11 ou superior instalado. Recomenda-se utilizar um ambiente virtual:

```bash
# Criação do ambiente virtual
python -m venv venv

# Ativando o ambiente virtual
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instalando todas as dependências do projeto
pip install -r requirements.txt
```

### 2. Fase de Treinamento (Machine Learning)

O modelo precisa ser treinado localmente para gerar os pesos (artefatos).

```bash
# Entre na pasta model
cd model

# Inicie o treinamento (Isso baixará o dataset e iniciará as épocas no TensorFlow)
python train.py

# Após o treinamento concluir, avalie as métricas (gera matriz de confusão e relatório)
python evaluate.py

# Volte para a raiz do projeto
cd ..
```
*Nota: Após o treino, os arquivos `emotion_model.keras` e `tokenizer.json` serão gerados dentro de `model/artifacts/`.*

### 3. Iniciando o Backend API

O backend é construído em FastAPI e expõe o modelo via uma API REST.

```bash
# Entre na pasta backend
cd backend

# Inicie o servidor via Uvicorn na porta 8000
uvicorn main:app --reload
```
A API estará rodando em `http://localhost:8000`. 
Você pode visualizar a documentação interativa no navegador acessando: `http://localhost:8000/docs`.

### 4. Iniciando o Frontend (Interface Web)

O frontend não necessita de build steps (como Webpack ou Vite). É feito inteiramente com Vanilla HTML/CSS/JS.

- Abra a pasta `frontend` no VSCode e utilize a extensão **Live Server**.
- Ou simplesmente dê um duplo clique no arquivo `frontend/index.html` para abri-lo no seu navegador.
- Certifique-se de que a API do backend (Passo 3) esteja rodando, para que a comunicação da interface com o modelo funcione.

---

## 📁 Estrutura do Repositório

- `model/`: Contém os scripts do pipeline de Deep Learning (`train.py`, `evaluate.py`, `preprocess.py`).
- `backend/`: Código-fonte da API em FastAPI (`main.py`, `predictor.py`, etc).
- `frontend/`: Interface web contendo a apresentação visual (`index.html`, estilos e scripts dinâmicos).
- `docs/`: Documentação aprofundada (Relatório Técnico e Referência da API).

Para maiores detalhes sobre a implementação e decisões arquiteturais, consulte o [Relatório Técnico](docs/relatorio_tecnico.md).
