import pickle

def concatenate_seq_files(fname1, fname2, new_fname):
	new_arr = []
	with open(fname1, "rb") as f:
		arr1 = pickle.load(f)
	with open(fname2, "rb") as f:
		arr2 = pickle.load(f)
	with open(new_fname, "wb") as f:
		pickle.dump(arr1 + arr2, f)