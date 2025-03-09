# Evaluación Automática de Exámenes

Esta aplicación web permite procesar imágenes de exámenes y evaluar automáticamente las respuestas usando OCR (Reconocimiento Óptico de Caracteres).

## Requisitos del Sistema

- Python 3.11 o superior
- Tesseract OCR instalado en el sistema

## Configuración Local

1. Instalar Tesseract OCR:
   - Windows: Descargar el instalador desde [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Asegurarse de instalar el idioma español durante la instalación

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar la variable de entorno:
   - Crear un archivo `.env` con la ruta a Tesseract:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

4. Ejecutar la aplicación:
```bash
python app.py
```

## Formato de Exámenes

Las imágenes de exámenes deben seguir este formato:
- Preguntas numeradas que terminen en "?"
- Respuestas marcadas con uno de estos símbolos:
  - • (punto)
  - * (asterisco)
  - X o ✖ (equis)
  - O o ○ (círculo)
  - ✸ (estrella)
  - ★ (estrella llena)
  - ✔ (check)

## Despliegue en Render

1. Crear una cuenta en [Render](https://render.com)
2. Conectar tu repositorio de GitHub
3. Crear un nuevo Web Service
4. Configurar las variables de entorno:
   - `TESSERACT_PATH=/usr/bin/tesseract`
   - `PYTHON_VERSION=3.11.0`

## Notas Importantes

- La calidad de la imagen es crucial para el reconocimiento de texto
- Todas las preguntas deben tener una respuesta marcada
- La aplicación validará que no haya preguntas sin respuesta
