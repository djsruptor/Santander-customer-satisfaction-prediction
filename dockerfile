FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .python-version uv.lock ./
RUN uv sync --locked
ENV PATH="/app/.venv/bin:$PATH"

COPY examples/ ./examples/
COPY predict.py ./
COPY models/best_model.pkl ./models/best_model.pkl
COPY src/ ./src/

EXPOSE 9696

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
