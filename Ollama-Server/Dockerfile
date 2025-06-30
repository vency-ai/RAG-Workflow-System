FROM ubuntu:22.04

# Install curl + jq + Ollama
RUN apt-get update && apt-get install -y curl jq gnupg \
 && curl -fsSL https://ollama.com/install.sh | sh

# Set path to Ollama
ENV PATH="/root/.ollama/bin:$PATH"

# Copy model config
WORKDIR /app
COPY models.json .

# Expose Ollama port
EXPOSE 11434

# Entry script that starts Ollama and pulls models
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
