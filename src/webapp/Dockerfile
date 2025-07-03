FROM python:3.12-slim

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy in your project files from the build context (the repo root)
COPY pyproject.toml .
COPY src/ src/

# Upgrade pip and install the project with the [webapp] extra
RUN pip install --upgrade pip
RUN pip install .[webapp]

# Switch into the webapp folder
WORKDIR /app/src/webapp

# Expose port 80 for HTTP
EXPOSE 80

# Start uvicorn in HTTP mode on port 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]