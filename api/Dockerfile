# api/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]

# FROM python:3.9-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# RUN if [ ! -f "app.py" ]; then echo "Arquivo app.py não encontrado!" && exit 1; fi
# CMD ["python", "app.py"]