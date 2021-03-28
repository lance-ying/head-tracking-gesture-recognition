import numpy as np

def M1(A, B, delta, eps):
	l1 = len(A)
	l2 = len(B)
	return LCSS(A, B, delta, eps) / np.min([l1, l2])

def M2(A, B, delta, eps):
	l1 = len(A)
	l2 = len(B)
	return SLC(A, B, delta, eps) / np.min([l1, l2])

def M3(A, B, delta, eps):
	l1 = len(A)
	l2 = len(B)
	return SLC(A, B, delta, eps) / np.min([l1, l2]) / (1 + np.abs(l1-l2))

def LCSS(A, B, delta, eps):
	if len(A) == 0 or len(B) == 0:
		return 0
	else:
		arr = np.zeros((len(A)+1, len(B)+1))
		for i in range(1, len(A)+1):
			for j in range(1, len(B)+1):
				# Check time distance and real distance
				if abs(A[i-1][0]-B[j-1][0]) < delta and np.linalg.norm(A[i-1][1]-B[j-1][1]) < eps:
					arr[i][j] = arr[i-1][j-1] + 1
				else:
					arr[i][j] = np.max([arr[i-1][j], arr[i][j-1]])
		return arr[len(A), len(B)]

def SLC(A, B, delta, eps):
	if len(A) == 0 or len(B) == 0:
		return 0
	else:
		arr = np.zeros((len(A)+1, len(B)+1))
		for i in range(1, len(A)+1):
			for j in range(1, len(B)+1):
				# Check time distance and real distance
				if abs(A[i-1][0]-B[j-1][0]) < delta and np.linalg.norm(A[i-1][1]-B[j-1][1]) < eps:
					C = eps - np.linalg.norm(A[i-1][1]-B[j-1][1])
					const = np.min([1, 1-(C/eps)])
					arr[i][j] = arr[i-1][j-1] + const
				else:
					arr[i][j] = np.max([arr[i-1][j], arr[i][j-1]])
		return arr[len(A), len(B)]