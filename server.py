from flask import Flask, request, jsonify, send_file
import easyocr
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        # Verificar si se envió una imagen en la solicitud
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
        
        image = request.files['image']
        
        # Verificar si la imagen tiene una extensión válida
        if image.filename == '':
            return jsonify({'error': 'Nombre de archivo de imagen inválido'}), 400
        
        # Cargar EasyOCR
        reader = easyocr.Reader(['es'])
        
        # Cargar la imagen en formato cv2 y luego convertirla a numpy array
        image_cv2 = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        
        # Leer el texto de la imagen
        result = reader.readtext(image_np)
        
        # Realizar los cálculos en la imagen (dibujar cuadros alrededor del texto y agregar texto)
        for detection in result:
            polygon = detection[0]
            text = detection[1]
            
            # Convertir los puntos de la detección en una lista de coordenadas
            polygon_array = np.array(polygon, dtype=np.int32)
            
            # Dibujar un polígono alrededor del texto
            cv2.polylines(image_cv2, [polygon_array], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Posición del texto
            org = (polygon_array[0][0], polygon_array[0][1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 255, 0)
            line_type = 2
            cv2.putText(image_cv2, text, org, font, font_scale, font_color, line_type)
        
        # Convertir la imagen con cálculos a formato PIL
        image_pil = Image.fromarray(image_cv2)
        
        # Guardar la imagen en un objeto BytesIO
        image_buffer = BytesIO()
        image_pil.save(image_buffer, format="JPEG")
        image_buffer.seek(0)
        
        extracted_text = [detection[1] for detection in result]
        
        # Convertir los bytes de la imagen a base64
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        
        # Retornar la imagen y el texto extraído en la respuesta JSON
        return jsonify({'image': image_base64, 'extracted_text': extracted_text})
        
    except Exception as e:
        print("Excepción:", e)
        return jsonify({'error': 'Ocurrió un error en el servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True)
