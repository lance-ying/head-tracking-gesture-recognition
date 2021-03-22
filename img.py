import numpy as np
import math
import cv2

def image_to_orientation(landmarks):
	image_points = np.array([
		landmarks[30], # Nose tip
		landmarks[8],  # Chin
		landmarks[36], # Left eye left corner
		landmarks[45], # Right eye right corner
		landmarks[48], # Left mouth corner
		landmarks[54]  # Right mouth corner
	])
	img_shape = (450, 450)
	model_points = np.array([
		[0.0, 0.0, 0.0],
		[0.0, -330.0, -65.0],
		[-225.0, 170.0, -135.0],
		[225.0, 170.0, -135.0],
		[-150.0, -150.0, -125.0],
		[150.0, -150.0, -125.0]
	])
	focal_length = img_shape[1]
	center = (float(img_shape[1])/2.0, float(img_shape[0])/2.0)
	camera_matrix = np.array([
		[focal_length, 0, center[0]],
		[0, focal_length, center[1]],
		[0, 0, 1]
	])
	dist_coeffs = np.zeros((4, 1))
	(_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	p1 = tuple(landmarks[30].astype(int))
	p2 = tuple(nose_end_point2D.flatten().astype(int))
	return p1, p2