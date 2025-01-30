FROM python:3.8-slim

# Install system dependencies needed by pandas, numpy, and scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libatlas3-base \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app source files into the container
COPY ./src /app/src
COPY app.py /app/

# Expose the port for the application
EXPOSE 2020

# Use Waitress to serve the app
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:2020", "app:app"]
