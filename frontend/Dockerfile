FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Command to run the Streamlit app, dynamically binding the port
CMD ["streamlit", "run", "--server.port", "$PORT", "main.py"]