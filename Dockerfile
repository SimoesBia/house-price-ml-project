FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py .

# O modelo é montado via volume (-v ./models:/app/models) ou gerado com `python main.py train`.
RUN useradd --create-home appuser && mkdir -p models data reports && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "main.py", "serve"]
