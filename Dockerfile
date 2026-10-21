FROM python:3.11-slim

RUN useradd -m -u 1000 app
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app . .
USER app
EXPOSE 7860
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxMessageSize=16", "--server.maxUploadSize=5", "--server.enableWebsocketCompression=true"]
