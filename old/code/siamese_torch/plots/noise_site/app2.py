from flask import Flask, send_file, render_template, jsonify, make_response
from PIL import Image, ImageDraw
import io
import random
import os

app = Flask(__name__)

def noisetest(image):
    try:
        # Calculate evenly spaced sizes from 2 to 40
        sizes = [int(2 + (40 - 2) * i / 9) for i in range(10)]  # 10 sizes from 2 to 40
        print(sizes)
        
        result = {}
        for i, size in enumerate(sizes, 1):
            # Create a fresh copy for each image
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)
            
            # Calculate valid x range based on size
            x1 = random.randint(61, 169 - size)
            y1 = random.randint(105, 145 - size)
            bbox = (x1, y1, x1 + size, y1 + size)
            
            choice = random.choice(["rect", "circle"])
            if choice == "rect":
                draw.rectangle(bbox, fill=255)
            else: draw.ellipse(bbox, fill=255)
            
            # result
            image_key = f'image{i}'
            result[image_key] = {
                'image': img_copy,
                'coords': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x1 + size,
                    'y2': y1 + size
                }
            }
        
        return result
    except Exception as e:
        print(f"Error in noisetest: {str(e)}")
        raise

def send_image_file(image):
    try:
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        response = make_response(send_file(img_io, mimetype='image/png'))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response
    except Exception as e:
        print(f"Error in send_image_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/coordinates/<image_id>')
def get_coordinates(image_id):
    try:
        valid_ids = [f'image{i}' for i in range(1, 11)]
        if image_id not in valid_ids:
            return jsonify({'error': 'Invalid image ID'}), 400
            
        image = Image.open("im.png")
        results = noisetest(image)
        
        if image_id in results:
            return jsonify(results[image_id]['coords'])
        return jsonify({'error': 'Invalid image ID'}), 400
        
    except Exception as e:
        print(f"Error in get_coordinates: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/image<int:image_num>')
def get_image(image_num):
    try:
        if not 1 <= image_num <= 10:
            return jsonify({'error': 'Invalid image number'}), 400
            
        image = Image.open("imt1.png")
        results = noisetest(image)
        return send_image_file(results[f'image{image_num}']['image'])
    except Exception as e:
        print(f"Error in get_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)