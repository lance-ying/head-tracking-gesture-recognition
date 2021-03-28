import numpy as np
import sys
import os
import dlib
import glob
import cv2

# Useful reference:
# http://dlib.net/face_landmark_detection.py.html
# https://livecodestream.dev/post/detecting-face-features-with-python/
# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

class PremadeDetector():
	def __init__(self, predictor_filepath):
		self.predictor_filepath = predictor_filepath
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.predictor_filepath)

	def get_landmarks(self, img):
		bounding_boxes = self.detector(img, 1)
		bounding_box = bounding_boxes[0] # Only use the first detection
		shape = self.predictor(img, bounding_box)
		return self._shape_to_np(shape)

	def _shape_to_np(shape):
		coords = np.zeros((68, 2))
		for i in range(0, 68):
			coords[i] = (shape.part(i).x, shape.part(i).y)
		return coords

	def __call__(self, img):
		return self.get_landmarks(img)