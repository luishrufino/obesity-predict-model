FROM python:3.9-slim

WORKDIR /app

COPY api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
