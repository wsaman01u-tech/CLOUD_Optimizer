FROM python:3.11-slim

WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend API, models, frontend dashboard, and results
COPY api/ api/
COPY agents/ agents/
COPY simulator/ simulator/
COPY data/results/ data/results/
COPY outputs/ outputs/
COPY frontend/ frontend/

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

