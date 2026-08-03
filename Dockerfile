# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /code

# Install system dependencies needed for OpenCV, MediaPipe, and Audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    glib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies (using CPU torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application files
COPY . /code

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Run Uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]