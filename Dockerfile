FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Eliminar la carpeta mlruns antes de instalar dependencias
RUN rm -rf mlruns

EXPOSE 8501

CMD ["streamlit", "run", "src/infrastructure/web/streamlit_app.py"]