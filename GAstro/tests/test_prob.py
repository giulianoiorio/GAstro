from .. import prob as pr
import numpy as np
import math


def test_make_simplex():

	tollerance=0.1

	N=10000
	K=3
	A = np.random.uniform(0, 1, N * K).reshape(N, K)
	W=pr.make_simplex(A)
	mean_true=1/K
	std_true= np.sqrt((mean_true*(1-mean_true)) / (K+1) )
	assert np.sum(W)==N
	for i in range(K):
		assert math.isclose(np.mean(W[:,i]), mean_true, rel_tol=tollerance)
		assert math.isclose(np.std(W[:,i]),  std_true, rel_tol=tollerance)

	N=10000
	K=2
	A = np.random.uniform(0, 1, N * K).reshape(N, K)
	W=pr.make_simplex(A)
	mean_true=1/K
	std_true= np.sqrt((mean_true*(1-mean_true)) / (K+1) )
	assert np.sum(W)==N
	assert np.sum(W)==N
	for i in range(K):
		assert math.isclose(np.mean(W[:,i]), mean_true, rel_tol=tollerance)
		assert math.isclose(np.std(W[:,i]),  std_true, rel_tol=tollerance)


	N=10000
	K=10
	A = np.random.uniform(0, 1, N * K).reshape(N, K)
	W=pr.make_simplex(A)
	mean_true=1/K
	std_true= np.sqrt((mean_true*(1-mean_true)) / (K+1) )
	assert np.sum(W)==N
	assert np.sum(W)==N
	for i in range(K):
		assert math.isclose(np.mean(W[:,i]), mean_true, rel_tol=tollerance)
		assert math.isclose(np.std(W[:,i]),  std_true, rel_tol=tollerance)

