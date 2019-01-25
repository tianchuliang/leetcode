# merge sort
# nlogn
def merge(left,right):
	# A is shorter, B is longer
	res = []
	l = 0 
	r = 0 
	while l < len(left) and r < len(right):

		if left[l] <= right[r]:
			res.append(left[l])
			l += 1
		else:
			res.append(right[r])
			r += 1
		

	while l <len(left):
		res.append(left[l])
		l += 1 

	while r <len(right):
		res.append(right[r])
		r += 1

	return res

def mergesort(A):
	if len(A) == 0:
		return A 
	if len(A) == 1:
		return A

	return merge(mergesort(A[:int(len(A)/2)]),mergesort(A[int(len(A)/2):]))

