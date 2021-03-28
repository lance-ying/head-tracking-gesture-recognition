import numpy as np
import cv2

from img import image_to_orientation
from premade_detector import PremadeDetector

mirror = True

cap = cv2.VideoCapture(0)
print("Press q to quit.")

detector = PremadeDetector("shape_predictor_68_face_landmarks.dat")

while(True):
	# Grab a frame
	ret, frame = cap.read()

	if mirror:
		frame = np.flip(frame, axis=1)
	landmarks, success = detector(frame)

	annotated = np.array(frame)
	if success:
		p1, p2 = image_to_orientation(landmarks.astype(float))
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
