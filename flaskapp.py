import json
import requests
import fitz
from pathlib import Path
from PIL import Image
import numpy as np
import pytesseract
import base64
from io import BytesIO

from model import Unet
from postprocess import regularize

thumbnails_dir = Path("/tmp/thumbnails")
thumbnails_dir.mkdir(parents=True, exist_ok=True)


def load_mode():
    num_class = 3
    model = Unet(num_class, input_size=(256, 256, 1), deep_supervision=False)
    model.model.load_weights('./gan.hdf5')
    return model.model


model = load_mode()


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
    abstract = pytesseract.image_to_string(abstract_cropped)
    return title, abstract


def get_cover(response):
    with open('/tmp/downloaded.pdf', 'wb') as f:
        f.write(response.content)
        doc = fitz.Document('/tmp/downloaded.pdf')
        thumbnail = get_thumbnail(doc)
        doc.select([0])
        doc.save('/tmp/cover.pdf')
        pix = doc[0].getPixmap(alpha=False, matrix=fitz.Matrix(2, 2))
        original = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
        resized = original.resize((256, 256), Image.NEAREST)
        resized = np.array(resized)
        resized = resized / 255
        resized = np.expand_dims(resized, 2)
        resized = np.expand_dims(resized, 0)
        return original, resized, (pix.width, pix.height), thumbnail


def get_paper_data(link):
    response = requests.get(link)
    original, resized, size, thumbnail = get_cover(response)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG")
    thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
    bbox = predict(resized)
    title, abstract = read_text(original, size, bbox)
    return {'title': title, 'abstract': abstract, 'thumbnail': thumbnail, 'pdfLink': link, 'bookmarked': False}


def get_thumbnail(doc):
    for i in range(len(doc)):
        images = doc.getPageImageList(i)
        if len(images) > 0:
            img = images[0]
            print(img)
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            try:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return img
            except ValueError:
                continue


def handler(event, context):
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
    print('handling..')
    event = json.loads(event['body'])
    result = get_paper_data(event['link'])
    print(result)
    response = {
        "statusCode": 200,
        "body": json.dumps(result)
    }
    return response
