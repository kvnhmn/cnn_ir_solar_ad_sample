FROM python:3.11-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=models/irnet.pt \
    DEVICE=cpu

COPY --from=builder /opt/venv /opt/venv

COPY serving/ ./serving/
COPY core/ ./core/
COPY model/ ./model/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]