
# 🧠 Obesity FastCheck

**Obesity FastCheck** é uma aplicação interativa para prever o nível de obesidade com base em dados de hábitos e estilo de vida. O sistema utiliza um modelo de machine learning treinado com dados públicos, uma API REST com Flask e uma interface amigável em Streamlit.

---

## 🚀 Visão Geral

Este projeto tem como objetivo fornecer uma **análise rápida e educativa** sobre fatores relacionados à obesidade. Ele entrega:

- 🎯 Predição de nível de obesidade com base em dados individuais
- 📊 Indicadores interpretáveis como IMC, estilo de vida e hábitos alimentares
- 🔄 Comunicação entre frontend (Streamlit) e backend (Flask API)

---

## 🧱 Estrutura do Projeto

```bash
obesity-predict-model/
│
├── api/               # API Flask para expor o modelo via HTTP
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── streamlit/         # Interface de usuário (frontend)
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── train/             # Script de treinamento do modelo
│   ├── train.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── shared/            # Código compartilhado (ex: engenharia de features, utils)
│   ├── utils.py
│   └── Dockerfile     # (opcional) imagem base de dependências comuns
│
├── Obesity.csv        # Base de dados original
├── docker-compose.yml # Orquestração dos serviços (API + UI)
├── start.ps1          # Script PowerShell para iniciar os containers (Windows)
└── README.md

```

---

## 🧠 Indicadores Calculados

| Indicador             | Interpretação                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| **IMC**               | Classifica o peso de acordo com a altura. Ideal: entre 18.5 e 24.9            |
| **LifestyleScore**    | De 0 a 4. Quanto maior, melhor: considera não fumar, controle calórico, etc. |
| **HealthyMealRatio**  | Proporção de refeições com vegetais. Ideal > 0.4                              |
| **ActivityBalance**   | Diferença entre tempo ativo e tempo em telas. Positivo é desejável            |
| **TransportType**     | Classificação do transporte: *active*, *neutral*, *sedentary*                 |

---

## ⚙️ Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/obesity-fastcheck.git
cd obesity-fastcheck
```

### 2. Treinar o modelo

```bash
cd train
python train_model.py
```

O modelo será salvo em `shared/model.pkl`.

### 3. Rodar com Docker Compose

```bash
docker-compose up --build
```

- A API estará disponível em: `http://localhost:5000/predict`
- A interface Streamlit estará em: `http://localhost:8501`

---

## 🌐 Deploy no Render

- Crie **dois serviços Web**: um para `api/` (Flask) e outro para `streamlit/`
- Certifique-se que o app Streamlit consome a URL da API correta (`https://api-obesity.onrender.com/predict`)

---

## 🧪 Exemplo de Chamada à API

```bash
curl -X POST http://localhost:5000/predict      -H "Content-Type: application/json"      -d '{"Age": 25, "Height": 1.75, "Weight": 80, ...}'
```

---

## 📌 Observações

- Esta aplicação **não substitui diagnóstico médico**.
- O modelo foi treinado com base em dados simulados e deve ser usado com fins educativos e preventivos.

---

## 👨‍💻 Autor

Desenvolvido por **Luis Rufino**, Analista de Dados.  
Este projeto integra conceitos de machine learning, análise de dados e deploy de aplicações interativas.

