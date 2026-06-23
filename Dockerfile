# Container image for Research Radar.
# Serves the ADK agents over HTTP - suitable for Google Cloud Run.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Use the AI Studio (Gemini API) backend, not Vertex AI.
# Provide GOOGLE_API_KEY at runtime (e.g. a Cloud Run secret) - never bake it in.
ENV GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Cloud Run injects $PORT (defaults to 8080 locally). ADK serves the agents in /app.
EXPOSE 8080
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT:-8080}"]
