# api/Dockerfile

FROM python:3.9-slim

# Define o diretório de trabalho DENTRO do container
WORKDIR /app

# Copia o 'requirements.txt' da pasta 'api' (o contexto atual do build)
COPY api/requirements.txt /app/
# Instala os pacotes
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta 'shared' (que está fora da pasta 'api') para dentro do container
COPY ../shared /app/shared

# Copia TODO o resto da pasta 'api' (app.py, obesity_model.pkl) para dentro do container
COPY . /app/

# Inicia o servidor Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT api:app

