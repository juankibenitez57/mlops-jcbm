# Imagen base oficial de Python, versión slim para reducir tamaño
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# curl necesario para el HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias ANTES que el código fuente
# Así la capa de pip se cachea aunque cambie el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Puerto que expone el contenedor (documentación, no mapeo real)
EXPOSE 8000

# Health check: espera hasta 30s al arranque y reintenta cada 10s
# El contenedor pasa a "healthy" solo cuando el modelo está cargado (HTTP 200)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Bind en 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
