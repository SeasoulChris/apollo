#!/usr/bin/env python

"""This script is just to demonstrate basic usage of opencv"""

import cv2
import numpy as np

def binary_to_picture(compressed_image_binary, generated_jpg_file):
  """Dump compressed image bytes with RGB format to photo."""
  with open(compressed_image_binary, "rb") as image_bin:
    data = image_bin.read()
  img = np.asarray(bytearray(data), dtype="uint8")
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  cv2.imwrite(generated_jpg_file, img)


if __name__ == '__main__':
    binary_to_picture('image-bin', 'image.jpg')

