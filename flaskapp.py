from flask import Flask, json, request
import requests
import re
import fitz
from pathlib import Path
from PIL import Image
import os
import numpy as np

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


def get_cover(response):
    with open('/tmp/downloaded.pdf', 'wb') as f:
        f.write(response.content)
        doc = fitz.open('/tmp/downloaded.pdf')
        doc.select([0])
        doc.save('/tmp/cover.pdf')
        pix = doc[0].getPixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.resize((256, 256), Image.NEAREST).convert('L')
        img = np.array(img)
        img = img / 255
        img = np.expand_dims(img, 2)
        img = np.expand_dims(img, 0)
        return img


@app.route('/', methods=['GET'])
def index():
    return 'hello world'


@app.route('/paper', methods=['GET'])
def get_paper_data():
    link = request.args.get('link')
    response = requests.get(link)
    cover = get_cover(response)
    bbox = predict(cover)
    return json.dumps(bbox)


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
