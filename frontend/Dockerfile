FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PORT 8501

# Entry point to dynamically bind to the Render-provided $PORT
CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT"]