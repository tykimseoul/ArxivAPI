import cv2
import numpy as np
import os
import re

title_color = [255, 0, 0]
abstract_color = [0, 0, 255]
background_color = [0, 0, 0]


def quantize(image):
    palette = np.array([title_color, abstract_color, background_color])
    distance = np.linalg.norm(image[:, :, None] - palette[None, None, :], axis=3)
    pal_img = np.argmin(distance, axis=2)
    rgb_img = palette[pal_img]
    return rgb_img


def mask_area(image, area):
    masked = cv2.inRange(image, tuple(area), tuple(area))
    masked = cv2.bitwise_and(image, image, mask=masked)
    return masked


def max_area(image):
    gray = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    areas = list(map(lambda c: cv2.contourArea(c), contours))
    max_contour = contours[np.argmax(areas)]

    return max_contour


def draw_contour(image, contour, color):
    x, y, w, h = cv2.boundingRect(contour)
    print(x, y, w, h, cv2.contourArea(contour))
    cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return x, y, w, h


def regularize(image):
    quantized = quantize(image)
    title_area = mask_area(quantized, title_color)
    abstract_area = mask_area(quantized, abstract_color)

    title_contour = max_area(title_area)
    abstract_contour = max_area(abstract_area)

    blank = np.zeros_like(image)
    title_bbox = draw_contour(blank, title_contour, title_color)
    abstract_bbox = draw_contour(blank, abstract_contour, abstract_color)

    return blank, title_bbox, abstract_bbox


if __name__ == "__main__":
    for file in sorted(os.listdir('./results/')):
        if not re.match(r'result_\d+.png', file):
            continue
        print(file)
        image = cv2.cvtColor(cv2.imread('./results/' + file), cv2.COLOR_BGR2RGB)
        regularized = regularize(image)
