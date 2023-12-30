import torch
from ultralytics import YOLO
import cv2

import tempfile
import itertools as IT
import os

import streamlit as st


def uniquify(path, sep=""):
    def name_sequence():
        count = IT.count()
        yield ""
        while True:
            yield "{s}{n:d}".format(s=sep, n=next(count))

    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


def detectImage(img, model):
    results = model(img)
    # print(results)

    if results:
        img = cv2.imread(img)
        for box in results[0].boxes:
            filename = results[0].names[int(box[0].cls[0].int())]
            x1, y1, x2, y2 = box[0].xyxy[0].int().tolist()
            cropped_img = img[y1:y2, x1:x2]
            cv2.imwrite(uniquify(f"./detected/{filename}.jpg"), cropped_img)
        return "Images cropped successfully."
    else:
        return "No object detected"


def main():
    model = YOLO("yolov8x.pt")
    img = r"bus.jpg"
    print(detectImage(img, model))

if __name__ == "__main__":
    main()
