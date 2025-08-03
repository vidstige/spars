FROM python:3.11-slim
# Dockerfile to run benchmarks

# Install build tools and scikit-sparse dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install scikit-sparse (for benchmarking comparison)
RUN pip install scikit-sparse

ENV PYTHONUNBUFFERED=1

# Generate matrices
COPY bindings/python/benchmarks/generate_data.py bindings/python/benchmarks/
RUN pip install numpy
RUN python bindings/python/benchmarks/generate_data.py

# Copy files
COPY src src
COPY include include
COPY lib lib
COPY bindings/python bindings/python

# Install Python package (edit if you prefer non-editable)
RUN pip install bindings/python

# Run benchmark
CMD ["python", "bindings/python/benchmarks/cholesky.py"]

