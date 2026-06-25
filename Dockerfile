# Container for the Streamlit app.
#   docker build -t india-equity-research .
#   docker run --rm -p 8501:8501 india-equity-research
# For the research mentor, pass LLM config: -e LLM_MODEL=... -e NVIDIA_NIM_API_KEY=...
# (A local Ollama on the host is reachable from the container via host.docker.internal.)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# WHY: bind 0.0.0.0 so the port is reachable from outside the container; headless so it does
# not try to open a browser; telemetry off. Env mirrors .streamlit/config.toml for clarity.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if b'ok' in urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).read() else 1)"

CMD ["streamlit", "run", "app.py"]
