FROM python:3.11-slim

WORKDIR /app

# Dependencias antes que el código fuente (optimización de caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# URL del backend — se sobreescribe en docker-compose.yml con el nombre de servicio
ENV BACKEND_URL=http://localhost:8000

# Gradio arranca en 0.0.0.0 para ser accesible desde fuera del contenedor
CMD ["python", "gradio_app.py"]
