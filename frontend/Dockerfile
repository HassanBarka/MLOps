FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY ./frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./frontend .

EXPOSE 8501

# Command to run the Streamlit app, dynamically binding the port
CMD ["streamlit", "run", "--server.port", "$PORT", "main.py"]