# Chatbot de análisis y combinación de colores

![image](https://github.com/user-attachments/assets/f700bb94-bb64-4299-9c46-486efc891410)

Aplicación web en Flask que extrae los colores dominantes de una imagen con K-Means y permite conversar con un chatbot, configurado como diseñador gráfico, sobre cómo usar y combinar esos colores. Proyecto del diplomado de Machine Learning.

El clustering de colores es implementación propia (OpenCV + scikit-learn). El chatbot no es un modelo entrenado por mí: consume GPT-4 a través de la API de OpenAI, orquestado con LangChain.

## Tecnologías

- **Backend:** Python, Flask
- **Procesamiento de imágenes:** OpenCV, Pillow
- **Clustering:** scikit-learn (K-Means)
- **Chatbot:** LangChain, FAISS, API de OpenAI (GPT-4)
- **Frontend:** HTML y CSS (plantilla Jinja2)

## Cómo ejecutarlo

**Requisitos:** Python 3.12 (probado con esta versión) y, solo para el chatbot, una clave de la API de OpenAI.

1. Clona el repositorio y entra en la carpeta:
```bash
git clone https://github.com/Jose47Morales/chatbotImageAnalysis.git
cd chatbotImageAnalysis
```
2. Crea y activa un entorno virtual:
```bash
python -m venv venv
source venv/Scripts/activate # Git Bash en Windows
```
PowerShell: venv\Scripts\Activate.ps1

macOS / Linux: source venv/bin/activate

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```
4. Configura la clave de OpenAI: copia `.env.example` a `.env` y pega tu clave en `OPENAI_API_KEY=`. El archivo `.env` no se sube al repositorio.
5. Inicia la aplicación y abre `http://127.0.0.1:5000`:
```bash
python app.py
```

**Qué esperar**
- Subir una imagen devuelve sus colores dominantes en hexadecimal (k=3 por defecto). Esta parte funciona sin clave de OpenAI.
- El chatbot necesita la clave y usa el modelo `gpt-4`, definido en `app.py`.

**Notas de compatibilidad**
- Las versiones de `requirements.txt` son de finales de 2024. No se ha probado con Python 3.14.
- Las importaciones de LangChain cambian entre versiones; por eso las versiones están fijadas.

## Evaluación de K-Means: ¿cuántos colores extraer?

La app usa k=3 de forma fija. Para comprobar si ese valor tiene sentido, probé K-Means con k de 2 a 10 sobre seis imágenes de distinto tipo, usando el método del codo (inercia) y el coeficiente de silueta. El análisis está en [`notebooks/Analisis.ipynb`](notebooks/Analisis.ipynb).

### Método

- **Preprocesamiento:** lectura con OpenCV, conversión a RGB y reducción a 150x150 píxeles (22.500 píxeles por imagen).
- **Clustering:** `KMeans` de scikit-learn con `random_state=42` y `n_init=10`, para k = 2…10.
- **Silueta:** `silhouette_score` con `sample_size=1000` (submuestreo para que no sea lento).
- **Filtrado de fondo (solo en la acuarela):** primero se descartaron los píxeles casi blancos (todos los canales > 235); después, un filtro HSV de papel (saturación < 30 y brillo > 200).

### Resultados

| Imagen | Tipo | k con silueta máxima | Silueta máx. | ¿Codo claro? |
|---|---|---|---|---|
| Zorro | Logo, 3 colores planos | 3 | 0.978 | Sí, en k=3 |
| Starbucks | Logo, 2 colores planos | 2 (k=3 casi empata: 0,947) | 0,949 | Difuso |
| Acuarela | Ilustración con gradientes | 2 | 0,675 | No |
| Paisaje | Ilustración vectorial | 2 | 0,674 | No |
| Supermercado | Fotografía a color | 2 | 0,599 | No |
| Cascada | Fotografía en blanco y negro | 2 | 0,646 | No |

### Hallazgos

1. **La silueta recupera el número real de colores solo en logos de color plano.** En el zorro, la inercia cae un 86 % de k=2 a k=3 y un 41 % de k=3 a k=4. En Starbucks (verde y blanco) gana k=2, y los clusters extra capturan tonos de transición de los bordes suavizados, no colores de diseño.
2. **En ilustraciones y fotografías, k=2 gana siempre** y separa zonas claras de oscuras (o fondo de figura). La silueta máxima baja a 0,60-0,68 y el codo no muestra un quiebre claro. Con una semilla distinta en el supermercado, k=2 sigue ganando (0,590).
3. **Los acentos minoritarios no obtienen un cluster propio.** En la acuarela, los tonos verde-azulados son el 1,5 % de los píxeles (340 de 22.500) y no aparecen en la paleta de k=6. En el supermercado, un azul con el 4 % de los píxeles sí obtuvo un centro propio; con solo dos casos no se puede separar el efecto del área del de la distancia de color.
4. **Filtrar el fondo mejora la paleta pero no resuelve el punto anterior.** En la acuarela, los dos blancos ocupaban el 67 % de los píxeles de k=6; tras filtrarlos aparecen dos corales distintos, pero los acentos azul-verdosos siguen sin aparecer.
5. **En escala de grises, K-Means agrupa un solo eje (el brillo).** La paleta son grises puros (R=G=B) y no hay colores que separar. La inercia baja de forma suave, casi como 1/k² entre k=2 y k=4 (hipótesis consistente con una distribución continua de tonos; no se comprobó con el histograma).

### Limitaciones

- **Seis imágenes:** describe lo observado, no es una prueba estadística.
- **Una sola semilla** en casi todos los casos (solo el supermercado se repitió). Las diferencias de silueta menores de ~0,015 deben tratarse como ruido.
- Las proporciones (por ejemplo, el 1,5 %) dependen de los umbrales y rangos de tono que elegí.
- Las siluetas con y sin filtro de fondo **no son comparables** entre sí, porque cambian los datos.
- Las imágenes proceden de internet y no se incluyen en el repositorio.

### Consecuencias para la app

- k=3 fijo es razonable en logos de tres colores, pero en una imagen de dos colores la app entrega un color que no existe (un tono de transición), y el chatbot lo trata como parte del diseño.
- Elegir k automáticamente con la silueta sirve para logos, pero en el resto de imágenes devolvería k=2, que es poco útil como paleta. Para esos casos, k debería fijarse según lo que necesite el usuario (por ejemplo, 4-6).
- Una imagen en escala de grises solo produce grises; sería útil detectarla y avisar.
