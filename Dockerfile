# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy model files (if they exist)
COPY *.joblib ./

# Set Python path
ENV PYTHONPATH=/app/src

# Run prediction script for verification
CMD ["python", "src/predict.py"] 