# Multi-Database Analytics Platform - Production Dockerfile
# Optimized for AI/ML workloads with Streamlit

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML and PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose Streamlit port
EXPOSE 8501

# Health check for container monitoring
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "ai_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]