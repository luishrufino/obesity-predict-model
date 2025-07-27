FROM python:3.9-slim

WORKDIR /app

COPY streamlit/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit/ /app/

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0