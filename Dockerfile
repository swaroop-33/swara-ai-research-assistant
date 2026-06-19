# Dockerfile
# ==============================================================================
# Use official lightweight Python 3.11 image based on Debian Bookworm
FROM python:3.11-slim

# Set environment variables to optimize Python performance inside the container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    SWARA_API_BASE=http://127.0.0.1:8000/api/v1 \
    ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a secure, non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory inside home to ensure full write permissions
WORKDIR /home/user/app

# Copy dependency configuration first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .

# Install dependencies into user space
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY --chown=user:user . .

# Ensure start.sh has executable permissions
RUN chmod +x start.sh

# Explicitly create and set permissions for directories where runtime files are written
RUN mkdir -p uploads chroma_db logs data screenshots

# Expose port 7860 (Hugging Face Spaces routes public traffic to this port by default)
EXPOSE 7860

# We will run our application using start.sh (which will be generated in the next step)
CMD ["/bin/sh", "start.sh"]
