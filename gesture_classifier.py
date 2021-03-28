import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from time_series_similarity import M1, M2, M3, series_to_time_series

class KNNGestureClassifier():
	def __init__(self, yes_fname, no_fname, other_fname, metric, delta, eps, n_neighbors):
		self.metric = metric
		self.delta = delta
		self.eps = eps
		with open(yes_fname, "rb") as f:
			yes_seqs = pickle.load(f)
			yes_seqs.pop(3)
		with open(no_fname, "rb") as f:
			no_seqs = pickle.load(f)
			no_seqs.pop(-1)
		with open(other_fname, "rb") as f:
			other_seqs = pickle.load(f)
		self.all_seqs = yes_seqs + no_seqs + other_seqs
		labels = np.zeros(len(self.all_seqs))
		labels[:len(yes_seqs)] = 0
		labels[len(yes_seqs):len(yes_seqs)+len(no_seqs)] = 1
		labels[len(yes_seqs)+len(no_seqs):] = 2
		# 0 is yes, 1 is no, 2 is null/no gesture/other
		similarity_mat = np.array([[self.metric(series_to_time_series(np.array(seq1)), series_to_time_series(np.array(seq2)), self.delta, self.eps) for seq2 in self.all_seqs] for seq1 in self.all_seqs])
		kernel_mat = np.reciprocal(1 + similarity_mat)
		plt.imshow(kernel_mat)
		plt.show()
		self.clf = KNeighborsClassifier(metric="precomputed", weights="distance", n_neighbors=n_neighbors).fit(kernel_mat, labels)

	def predict(self, seq):
		test_sim = np.array([self.metric(series_to_time_series(np.array(seq)), series_to_time_series(np.array(train_seq)), self.delta, self.eps) for train_seq in self.all_seqs])
		print(test_sim)
		test_ker = np.reciprocal(1 + test_sim)
		return self.clf.predict([test_ker])

if __name__ == "__main__":
	yes_fname = "data/gestures/yes_seqs.pkl"
	no_fname = "data/gestures/no_seqs.pkl"
	other_fname = "data/gestures/other_seqs.pkl"
	metric = M3
	delta = 15
	eps = 10
	n_neighbors = 5
	clf = SVMGestureClassifier(yes_fname, no_fname, other_fname, metric, delta, eps, n_neighbors)

	import cv2
	from img import image_to_orientation
	from premade_detector import PremadeDetector
	detector = PremadeDetector("shape_predictor_68_face_landmarks.dat")
	cap = cv2.VideoCapture(0)
	seq_len = 40
	seq = []

	while True:
		ret, frame = cap.read()
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
			dx = p2[0]-p1[0]
			dy = p2[1]-p1[1]
			seq.append([dx, dy])

		cv2.imshow('frame',cv2.resize(annotated, (800, 800)))
		if len(seq) >= seq_len:
			print(clf.predict(seq))
			seq = []
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
