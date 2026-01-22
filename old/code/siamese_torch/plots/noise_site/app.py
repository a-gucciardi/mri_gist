from flask import Flask, send_file, render_template, jsonify, make_response
from PIL import Image, ImageDraw
import io
import random
import os
# import utils as ut

app = Flask(__name__)

def noisetest(image):
    image1 = image.copy()
    image2 = image.copy()
    image3 = image.copy()
    draw1 = ImageDraw.Draw(image1)
    draw2 = ImageDraw.Draw(image2)
    draw3 = ImageDraw.Draw(image3)

    # big
    x1 = random.randint(61, 129)
    bbox1 = (x1, 105, x1 + 40, 145)
    # medium
    x2 = random.randint(61, 149)
    y2 = random.randint(106, 124)
    bbox2 = (x2, y2, x2 + 20, y2 + 20)
    # small
    x3 = random.randint(61, 159)
    y3 = random.randint(106, 134)
    bbox3 = (x3, y3, x3 + 10, y3 + 10)

    fill_color = 255
    draw1.rectangle(bbox1, fill=fill_color)
    draw2.rectangle(bbox2, fill=fill_color)
    draw3.rectangle(bbox3, fill=fill_color)

    return {
        'image1': {'image': image1, 'coords': {'x1': x1, 'y1': 105, 'x2': x1+40, 'y2': 145}},
        'image2': {'image': image2, 'coords': {'x1': x2, 'y1': y2, 'x2': x2+20, 'y2': y2+20}},
        'image3': {'image': image3, 'coords': {'x1': x3, 'y1': y3, 'x2': x3+10, 'y2': y2+10}}
    }

# ok range to draw is: x(61->169) y(105->145)
# size has to be in (2x2) -> (40x40)

def noisetest2(image, size = 0.5):
    # scale 0->1
    images, draws, boxes = [], [], []
    result = {}
    for i in range(10):
        images.append(image.copy())
        draws.append(ImageDraw.Draw(images[i]))

        size = random.randint(2, 50)
        print(size)

        x1 = random.randint(61, 169-size)
        y1 = random.randint(105, 145-size)
        bbox = (x1, y1, x1+size, y1+size)
        boxes.append(bbox)

        draws[i].rectangle(boxes[i], fill = 255)

        image_key = f'image{i+1}'
        result[image_key] = {
            'image': images[i], 
            'coords': {
                'x1':x1, 
                'y1':y1, 
                'x2': x1+size, 
                'y2': y1+size}
        }
    
    return result

@app.route('/coordinates/<image_id>')
def get_coordinates(image_id):
    try:
        if image_id not in ['image1', 'image2', 'image3']:
            return jsonify({'error': 'Invalid image ID'}), 400
            
        image = Image.open("im.png")
        results = noisetest(image)
        
        if image_id in results:
            return jsonify(results[image_id]['coords'])
        return jsonify({'error': 'Invalid image ID'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def send_image_file(image):
    try:
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        response = make_response(send_file(img_io, mimetype='image/png'))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image1')
def get_image1():
    try:
        image = Image.open("im.png")
        results = noisetest(image)
        return send_image_file(results['image1']['image'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image2')
def get_image2():
    try:
        image = Image.open("im.png")
        results = noisetest(image)
        return send_image_file(results['image2']['image'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image3')
def get_image3():
    try:
        image = Image.open("im.png")
        results = noisetest2(image)
        return send_image_file(results['image3']['image'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)