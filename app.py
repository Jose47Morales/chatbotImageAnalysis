from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename 
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

# Configuramos
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

# Ruta de la main page
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar imágenes
@app.route('/analyze', methods = ['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Procesamos la imagen
    colors = extract_colors(filepath)
    return jsonify({"colors": colors})

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

if __name__ == '__main__':
    app.run(debug = True)