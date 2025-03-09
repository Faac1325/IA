from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
import re
import os
import openai
import json
from dotenv import load_dotenv

app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    print("ERROR: No se encontró la API key de OpenAI")

# Configurar la ruta de Tesseract desde la variable de entorno
tesseract_path = os.getenv('TESSERACT_PATH')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def cargar_preguntas():
    """Carga las preguntas y respuestas desde el archivo JSON."""
    try:
        with open('preguntas.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data['preguntas']
    except Exception as e:
        print(f"Error al cargar preguntas: {str(e)}")
        return {}

# Cargar preguntas y respuestas desde el archivo JSON
RESPUESTAS_CORRECTAS = cargar_preguntas()

def preprocess_image(image):
    """Preprocesa la imagen para mejorar el reconocimiento de texto."""
    # Convertir la imagen PIL a formato OpenCV
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para mejorar el contraste
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Reducir ruido
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Convertir de nuevo a formato PIL
    return Image.fromarray(denoised)

def extract_questions_and_answers(text):
    """Extrae preguntas y respuestas marcadas del texto."""
    lines = text.split('\n')
    current_question = None
    questions_answers = {}
    valid_markers = ['•', '*', 'X', '✖', 'O', '○', '✸', '★', '✔']
    empty_questions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es una pregunta (comienza con número y termina en ?)
        if re.match(r'^\d+\..*\?$', line):
            current_question = re.sub(r'^\d+\.\s*', '', line)
            questions_answers[current_question] = None
            empty_questions.append(current_question)  # Agregar pregunta a la lista de vacías
        # Detectar si es una respuesta marcada
        elif current_question and any(line.startswith(marker) for marker in valid_markers):
            # Eliminar el marcador y espacios
            answer = re.sub(r'^[' + ''.join(valid_markers) + r']\s*', '', line)
            # Eliminar la letra de la opción si existe (e.g., "A) ", "B) ")
            answer = re.sub(r'^[A-D]\)\s*', '', answer)
            questions_answers[current_question] = answer.strip()
            empty_questions.remove(current_question)  # Remover pregunta de la lista de vacías
    
    if empty_questions:
        raise ValueError(f"Las siguientes preguntas no tienen respuesta marcada: {', '.join(empty_questions)}")
            
    return questions_answers

def verificar_respuesta_con_ia(pregunta, respuesta_estudiante):
    """Verifica la respuesta usando GPT para determinar si es correcta."""
    try:
        print(f"\nVerificando respuesta con IA:")
        print(f"Pregunta: {pregunta}")
        print(f"Respuesta: {respuesta_estudiante}")
        
        prompt = f"""
        Actúa como un profesor experto evaluando un examen de computación básica.
        
        Pregunta: {pregunta}
        Respuesta del estudiante: {respuesta_estudiante}
        
        Evalúa si la respuesta es correcta y responde EXACTAMENTE en este formato JSON:
        {{
            "es_correcta": true/false,
            "respuesta_correcta": "la respuesta más precisa y completa",
            "explicacion": "explicación breve de por qué es correcta o incorrecta"
        }}
        
        La respuesta debe ser técnicamente correcta y precisa.
        """
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            n=1,
            stop=None
        )
        
        # Extraer y parsear la respuesta JSON
        respuesta_texto = response.choices[0].text.strip()
        print(f"Respuesta de OpenAI: {respuesta_texto}")
        
        # Intentar parsear el JSON
        try:
            resultado = json.loads(respuesta_texto)
            return resultado
        except json.JSONDecodeError as e:
            print(f"Error al parsear JSON: {str(e)}")
            print(f"Texto recibido: {respuesta_texto}")
            return {
                "es_correcta": False,
                "respuesta_correcta": "Error en el formato de respuesta",
                "explicacion": "No se pudo procesar la respuesta del sistema"
            }
            
    except Exception as e:
        print(f"Error al verificar con IA: {str(e)}")
        return {
            "es_correcta": False,
            "respuesta_correcta": "No se pudo verificar",
            "explicacion": f"Error en la verificación: {str(e)}"
        }

def evaluate_answers(student_answers):
    """Evalúa las respuestas del estudiante usando IA."""
    total_questions = len(student_answers)
    correct_answers = 0
    results = []
    
    print("\nEvaluando respuestas:")
    print(f"Total de preguntas: {total_questions}")
    print("Respuestas del estudiante:", student_answers)
    
    for question, student_answer in student_answers.items():
        print(f"\nEvaluando pregunta: {question}")
        # Verificar cada respuesta con IA
        verification = verificar_respuesta_con_ia(question, student_answer)
        
        is_correct = verification["es_correcta"]
        correct_answers += 1 if is_correct else 0
        
        results.append({
            'question': question,
            'student_answer': student_answer,
            'correct_answer': verification["respuesta_correcta"],
            'is_correct': is_correct,
            'explanation': verification["explicacion"]
        })
    
    grade = (correct_answers / total_questions) * 10 if total_questions > 0 else 0
    
    final_result = {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'grade': round(grade, 2),
        'details': results
    }
    
    print("\nResultado final:", json.dumps(final_result, indent=2, ensure_ascii=False))
    return final_result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'uploaded_image' not in request.files:
                return render_template('index.html', error="Debe seleccionar una imagen para subir.")
            
            file = request.files['uploaded_image']
            if file.filename == '':
                return render_template('index.html', error="No se ha seleccionado ningún archivo.")
            
            print("\n==== INICIO DE PROCESAMIENTO DE IMAGEN ====")
            print(f"Archivo recibido: {file.filename}")
            
            # Leer y preprocesar la imagen
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
            # Extraer texto de la imagen procesada
            extracted_text = pytesseract.image_to_string(processed_image, lang='spa')
            print("\n1. TEXTO EXTRAÍDO DE LA IMAGEN:")
            print("--------------------------------")
            print(extracted_text)
            print("--------------------------------")
            
            if not extracted_text.strip():
                print("ERROR: No se pudo extraer texto de la imagen")
                return render_template('index.html', error="No se pudo extraer texto de la imagen. Intente con una imagen más clara.")

            # Extraer preguntas y respuestas del texto
            try:
                student_answers = extract_questions_and_answers(extracted_text)
            except ValueError as e:
                return render_template('index.html', error=str(e))
                
            print("\n2. RESPUESTAS DETECTADAS:")
            print("--------------------------------")
            print(student_answers)
            print("--------------------------------")
            
            if not student_answers:
                return render_template('index.html', error="No se detectaron preguntas y respuestas en el formato correcto.")
            
            # Evaluar las respuestas
            evaluation = evaluate_answers(student_answers)
            print("\n3. RESULTADO DE LA EVALUACIÓN:")
            print("--------------------------------")
            print(evaluation)
            print("--------------------------------")
            print("\n==== FIN DE PROCESAMIENTO ====\n")
            
            return render_template('results.html', result=evaluation)

        except Exception as e:
            print(f"\nERROR EN EL PROCESAMIENTO: {str(e)}")
            return render_template('index.html', error=f"Error al procesar la imagen: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)