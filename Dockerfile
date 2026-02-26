# 1. Imagem Base: Começamos com um Python leve (Slim)
FROM python:3.11-slim

# 2. Pasta de Trabalho dentro do container
WORKDIR /app

# 3. Copiar os requisitos primeiro (para aproveitar cache)
# Crie um arquivo requirements.txt com: fastapi, uvicorn, openai, structlog, prometheus-fastapi-instrumentator
COPY requirements.txt .

# 4. Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o código da API
COPY main5-1.py main.py

# 6. Expor a porta 8000
EXPOSE 8000

# 7. Comando para ligar o servidor quando o container inicializar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]