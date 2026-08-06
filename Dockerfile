# Production Dockerfile for AutoEDA Pro Django Platform
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=autoeda.settings

WORKDIR /app

# Install system dependencies for graphics and compilation libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy entire repository source into container
COPY . /app/

# Set Python path to include /app and /app/django_app
ENV PYTHONPATH="/app:/app/django_app:${PYTHONPATH}"

# Collect static files for WhiteNoise production serving
RUN python django_app/manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Launch Gunicorn using thread-isolated configuration
CMD ["gunicorn", "-c", "gunicorn.conf.py", "autoeda.wsgi:application"]
