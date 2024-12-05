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

# Cargamos las variables de entorno
load_dotenv()

# Inicializamos flask
app = Flask(__name__)

# Configuramos
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

# Configuración de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

extracted_colors = []
chat_memory = []

# Ruta de la main page
@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug = True)