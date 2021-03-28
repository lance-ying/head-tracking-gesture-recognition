import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

from img import image_to_orientation
from premade_detector import PremadeDetector

mirror = True

cap = cv2.VideoCapture(0)
print("Press q to quit.")
print("Press y to start recording a yes sequence, and then press y again to stop.")
print("Press n to start recording a no sequence, and then press n again to stop.")
print("Press o to start recording an other sequence, and then press o again to stop.")

detector = PremadeDetector("shape_predictor_68_face_landmarks.dat")

yes_seqs = []
no_seqs = []
other_seqs = []
current_seq = []
recording_yes = False
recording_no = False
recording_other = False

yes_seqs_fname = "data/gestures/yes_seqs.pkl"
no_seqs_fname = "data/gestures/no_seqs.pkl"
other_seqs_fname = "data/gestures/other_seqs.pkl"

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
		if recording_yes or recording_no:
			dx = p2[0]-p1[0]
			dy = p2[1]-p1[1]
			current_seq.append([dx, dy])

	cv2.imshow("frame",cv2.resize(annotated, (800, 800)))
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("y") and not recording_no and not recording_other:
		if recording_yes:
			print("Finished recording a yes sequence.")
			recording_yes = False
			yes_seqs.append(current_seq)
			# plt.plot(np.array(current_seq)[:,0], np.array(current_seq)[:,1])
			# plt.show()
			current_seq = []
		else:
			print("Recording a yes sequence.")
			recording_yes = True
	elif key == ord("n") and not recording_yes and not recording_other:
		if recording_no:
			print("Finished recording a no sequence.")
			recording_no = False
			no_seqs.append(current_seq)
			# plt.plot(np.array(current_seq)[:,0], np.array(current_seq)[:,1])
			# plt.show()
			current_seq = []
		else:
			print("Recording a no sequence.")
			recording_no = True
	elif key == ord("o") and not recording_yes and not recording_no:
		if recording_other:
			print("Finished recording an other sequence.")
			recording_other = False
			other_seqs.append(current_seq)
			# plt.plot(np.array(current_seq)[:,0], np.array(current_seq)[:,1])
			# plt.show()
			current_seq = []
		else:
			print("Recording an other sequence.")
			recording_other = True


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

with open(yes_seqs_fname, "wb") as f:
	pickle.dump(yes_seqs, f)
with open(no_seqs_fname, "wb") as f:
	pickle.dump(no_seqs, f)
with open(other_seqs_fname, "wb") as f:
	pickle.dump(other_seqs, f)