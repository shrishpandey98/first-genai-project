FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for pandas/numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire workspace to handle relative data paths if needed
# (In a real app, only the source code and data would be copied)
COPY . .

# Install dependencies
# We install the package in editable mode to ensure the relative imports work
RUN pip install --no-cache-dir -e .

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GROQ_API_KEY=""

# Run the FastAPI server
# We use the module path to run the API
CMD ["python", "-m", "restaurant_reco_phase5.api"]
