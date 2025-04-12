# Use official Python image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (ffmpeg for audio conversion)
RUN apt-get update && apt-get install -y ffmpeg

# Copy all files from backend folder into container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download model and label encoder from Google Drive
RUN python prestart.py

# Start the Flask app using Gunicorn on the dynamic port set by Railway
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 app:app