FROM python:3.13-slim

# Install WeasyPrint system dependencies + tini for signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libpq-dev \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --no-create-home --uid 1000 app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory with proper permissions
RUN mkdir -p /app/output && chown -R app:app /app/output

# Switch to non-root user
USER app

EXPOSE 5050

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5050/api/health')" || exit 1

# Use tini for proper signal handling (graceful shutdown)
ENTRYPOINT ["tini", "--"]

# Gunicorn with 2 workers, 4 threads each for pipeline concurrency
CMD ["gunicorn", "wsgi:application", \
     "--bind", "0.0.0.0:5050", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "300", \
     "--keep-alive", "65", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
