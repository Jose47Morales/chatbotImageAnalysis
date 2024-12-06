# Chatbot de analisis y combinación de colores

![image](https://github.com/user-attachments/assets/f700bb94-bb64-4299-9c46-486efc891410)

## Objetivo del proyecto
El programa es una aplicación Flask que permite:
1. Subir una imagen y extraer sus colores dominantes mediante clustering (K-Means).
2. Interactuar con un chatbot configurado como diseñador gráfico que responde preguntas relacionadas con los colores extraídos.

---

## Secciones del Código

### 1. **Configuración Inicial**

#### Importación de librerías
```python
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename 
from PIL import Image
import cv2
import numpy as np
import openai
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
```

Estas bibliotecas permiten manejar:
* Flask: Crear el servidor web.
* PIL y OpenCV: Procesar imágenes.
* LangChain y OpenAI: Construir el chatbot.
* dotenv: Cargar variables sensibles desde `.env`.

#### Configuración del servidor Flask
```python
# Inicializamos flask
app = Flask(__name__)

# Configuramos
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)
```

Se define la carpeta donde se almacenarán las imágenes subidas (`uploads`). Si no existe, se crea automáticamente.

#### Configuración de OpenAI
```python
# Configuración de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
```

Se carga la API Key de OpenAI desde el archivo `.env`.

---

### 2. **Variables Globales**
```python
extracted_colors = []
chat_memory = []
```

* `extracted_colors`: Lista para almacenar los colores dominantes extraídos.
* `chat_memory`: Historial de la conversación con el chatbot.

---

### 3. **Rutas del servidor**

#### Ruta principal(`/`)
```python
# Ruta de la main page
@app.route('/')
def index():
    return render_template('index.html')
```

Cargará la página principal, que contiene el formulario para subir imágenes y el chatbot.

#### Ruta para Subir y Analizar Imágenes (`/analyze`)
```python
# Ruta para procesar imágenes
@app.route('/analyze', methods = ['POST'])
def analyze_image():
    global extracted_colors
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Procesamos la imagen
    extracted_colors = extract_colors(filepath)
    global chat_memory
    chat_memory = []

    return jsonify({"colors": extracted_colors})
```

1. Verifica que se haya enviado una imagen.
2. Guarda la imagen en el servidor.
3. Llama a la función `extract_colors` para obtener los colores dominantes.
4. Devuelve los colores extraídos en formato JSON

---

### 4. **Procesamiento de Imágenes**

#### Función para ectraer colores dominantes
```python
# Función para extraer colores dominantes
def extract_colors(image_path, k = 3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    pixels = image.reshape(-1, 3)

    # Clustering con K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Convertimos a formato HEX
    hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]
    return hex_colors
```

#### 1. Lectura y procesamiento:

* Convierte la imagen a formato RGB y reduce su tamaño para optimizar el cálculo.

#### 2. Clustering:

* Usa K-Means para agrupar los píxeles en `k` clusters.

#### 3. Conversión:
* Convierte los colores RGB de los clusters al formato hexadecimal.

---

### 5. **Chatbot con colores**

#### Función `consultas`
```python
def consultas(colores, pregunta, memoria=[]):
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(
        [f"Color {i+1}: {color}" for i, color in enumerate(colores)],
        embeddings
    )

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    llm = ChatOpenAI(model='gpt-4', temperature=0.7)

    contexto = """
    Eres un diseñador gráfico profesional especializado en análisis y combinación de colores. 
    Proporciona respuestas detalladas y creativas sobre los colores que se te proporcionen. Habla como un humano experto y
    no menciones que eres un modelo de inteligencia artificial. Si te preguntan por una preferencia u opinión, basate en principios de diseño y responde.
    Si te hablan de una imagen, haz de cuenta que te hablan de los colores en tu contexto.
    """
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    respuesta = crc({'question': f"{contexto} {pregunta}", 'chat_history': memoria})
    memoria.append((pregunta, respuesta['answer']))

    return respuesta, memoria
```

#### 1. Crear Embeddings:
* Convierte los colores extraídos en representaciones vectoriales

#### 2. Configurar el Chatbot:
* Usa `LangChain` para crear un chatbot que responde preguntas basadas en los colores proporcionados.

### 3. Historial de Chat:
* Actualiza la memoria del chat con cada pregunta y respuesta.

---

### 6. **Ruta para el Chatbot**
```python
# Ruta para el chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    global extracted_colors, chat_memory
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "No se recibió nigún mensaje"}), 400
        
        if not extracted_colors:
            return jsonify({"error": "No hay colores dominantes disponibles. Por favor, analiza una imagen primero."}), 400

        respuesta, chat_memory = consultas(extracted_colors, user_message, chat_memory)

        return jsonify({'reply': respuesta['answer']})
    except Exception as e:
        print("Error en OpenAI API:", str(e))
        return jsonify({'error': str(e)}), 500
```

1. Recibe un mensaje del usuario.
2. Verifica si ya se han extraído colores de la imagen subida.
3. Llama a la función `consultas` para obtener una respuesta basada en los colores extraídos.
4. Devuelve la respuesta en formato JSON.

---

### 7. **Ejecución del Servidor**
```python
if __name__ == '__main__':
    app.run(debug = True)
```

Inicia la aplicación Flask en modo de depuración.

---

## Casos de uso

1. Extraer colores de una imagen:
* Subir una imagen y recibir los colores dominantes en formato hexadecimal.

2. Asesoramiento de Diseño:
* Hacer preguntas como:
  * "¿Cómo combinarías estos colroes en un diseño?"
  * "¿Cuál es el más adecuado para un logo profesional?"
