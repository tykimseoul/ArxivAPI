from flask import Flask, json, request
import requests
import re
import fitz
from pathlib import Path
from PIL import Image
import os
import numpy as np
import pytesseract

from model import Unet
from postprocess import regularize

app = Flask(__name__)
thumbnails_dir = Path("/tmp/thumbnails")
thumbnails_dir.mkdir(parents=True, exist_ok=True)


def load_mode():
    num_class = 3
    model = Unet(num_class, input_size=(256, 256, 1), deep_supervision=False)
    model.model.load_weights('./gan.hdf5')
    return model.model


model = load_mode()


def extract_cvf_thumbnail(url):
    code = re.match(r'https://openaccess.thecvf.com/(\w+)/html/([\w-]+).html', url).groups()
    pdf_link = 'https://openaccess.thecvf.com/{}/papers/{}.pdf'.format(code[0], code[1])
    if not check_thumbnail(code[1]):
        response = requests.get(pdf_link)
        store_thumbnail(response, code[1])
    return read_thumbnail(code[1]), pdf_link


def predict(image):
    result = model.predict(image, verbose=1)
    result = np.squeeze(result, axis=0)
    result = result * 255
    result, title_bbox, abstract_bbox = regularize(result)
    print(title_bbox, abstract_bbox)
    return title_bbox, abstract_bbox


def read_text(image, size, bbox):
    image = np.array(image)
    title_bbox = (np.multiply(np.array(bbox[0]), np.tile(size, (1, 2)))[0] / 256).astype(int)
    abstract_bbox = (np.multiply(np.array(bbox[1]), np.tile(size, (1, 2)))[0] / 256).astype(int)
    title_bbox = title_bbox + np.array([-1, -1, 1, 1]) * 4
    abstract_bbox = abstract_bbox + np.array([-1, -1, 1, 1]) * 4
    title_cropped = image[title_bbox[1]:title_bbox[3], title_bbox[0]:title_bbox[2]]
    abstract_cropped = image[abstract_bbox[1]:abstract_bbox[3], abstract_bbox[0]:abstract_bbox[2]]
    title = pytesseract.image_to_string(title_cropped)
    print(title)
    abstract = pytesseract.image_to_string(abstract_cropped)
    print(abstract)
    return title, abstract


def get_cover(response):
    with open('/tmp/downloaded.pdf', 'wb') as f:
        f.write(response.content)
        doc = fitz.Document('/tmp/downloaded.pdf')
        doc.select([0])
        doc.save('/tmp/cover.pdf')
        pix = doc[0].getPixmap(alpha=False, matrix=fitz.Matrix(2, 2))
        original = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
        resized = original.resize((256, 256), Image.NEAREST)
        resized = np.array(resized)
        resized = resized / 255
        resized = np.expand_dims(resized, 2)
        resized = np.expand_dims(resized, 0)
        return original, resized, (pix.width, pix.height)


@app.route('/', methods=['GET'])
def index():
    return 'hello world'


@app.route('/paper', methods=['GET'])
def get_paper_data():
    link = request.args.get('link')
    response = requests.get(link)
    original, resized, size = get_cover(response)
    bbox = predict(resized)
    title, abstract = read_text(original, size, bbox)
    return {'title': title, 'abstract': abstract}


def store_thumbnail(response, key):
    with open('/tmp/downloaded.pdf', 'wb') as f:
        f.write(response.content)
        doc = fitz.open('/tmp/downloaded.pdf')
        for i in range(len(doc)):
            images = doc.getPageImageList(i)
            if len(images) > 0:
                img = images[0]
                print(img)
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                try:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save('{}/{}.jpg'.format(str(thumbnails_dir), key), 'JPEG')
                except ValueError:
                    pix = fitz.Pixmap(pix, 0)
                    pix.writeImage('{}/{}.jpg'.format(str(thumbnails_dir), key))
                break


def read_thumbnail(key):
    return Image.open('{}/{}.jpg'.format(str(thumbnails_dir), key))


def check_thumbnail(key):
    return '{}.jpg'.format(key) in os.listdir(str(thumbnails_dir))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
