FROM python:3.9-slim

# Define diretório de trabalho
WORKDIR /app

# Copia e instala dependências primeiro (para cache otimizado)
COPY api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do diretório api
COPY api/ /app/
COPY shared /utils/

# Usa a porta dinâmica definida pelo Render
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]