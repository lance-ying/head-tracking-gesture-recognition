import pickle

def concatenate_seq_files(fname1, fname2, new_fname):
	new_arr = []
	with open(fname1, "rb") as f:
		arr1 = pickle.load(f)
	with open(fname2, "rb") as f:
		arr2 = pickle.load(f)
	for seq in arr1:
		new_arr.append(seq)
	for seq in arr2:
		new_arr.append(seq)
	with open(new_fname, "wb") as f:
		pickle.dump(new_arr, f)