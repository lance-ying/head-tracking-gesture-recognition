import numpy as np
import cv2

from network import load_from_checkpoint

mirror = True
model_fname = "checkpoints/first_colab_model.checkpoint"

cap = cv2.VideoCapture(0)
model = load_from_checkpoint(model_fname)

print("Press q to quit.")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	if mirror:
		frame = np.flip(frame, axis=1)

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
