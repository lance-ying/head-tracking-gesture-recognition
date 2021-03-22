import numpy as np
import cv2

from network import load_from_checkpoint, torchify_image, detorchify_output
from img import image_to_orientation

mirror = True
model_fname = "checkpoints/first_colab_model.checkpoint"

cap = cv2.VideoCapture(0)
model = load_from_checkpoint(model_fname)

print("Press q to quit.")

def center_crop(img):
	return img[h//2 - 450//2:h//2 + 450//2, w//2 - 450//2:w//2 + 450//2]

while(True):
	# Grab a frame
	ret, frame = cap.read()

	if mirror:
		frame = np.flip(frame, axis=1)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Crop to 450x450 around the center
	h, w = gray.shape
	gray = center_crop(gray)

	landmarks = detorchify_output(model(torchify_image(gray))).reshape(-1,2)
	p1, p2 = image_to_orientation(landmarks)

	annotated = np.array(center_crop(frame))
	for i in range(len(landmarks)):
		x, y = landmarks[i]
		h = int(y)
		w = int(x)
		cv2.circle(annotated, tuple(landmarks[i]), 5, (0,0,255), -1)
	cv2.line(annotated, p1, p2, (255,0,0), 2)

	cv2.imshow('frame',cv2.resize(annotated, (800, 800)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
