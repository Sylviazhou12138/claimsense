FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install packages - gradio 5.x + google-genai are compatible on Python 3.11
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
