################################################
#
#
#
# Utils to generate random sample from distributions
#
#
#
#
################################################
import numpy as np

def Mutivariate(mean, covariance_matrix, Npersample=1, flatten=True):
	"""
	Function to generate a multivariate gaussian distribution starting from a list of mean vectors and covariance matrixes.
	If mean is just a monodimensional list, thefunction uses the numpy multivariate random. While in the other cases,
	the random sample is generated with the cholesky decomposition method, where.
	S = LxN +mena, where L is the cholesky decomposition of the covariance matrix, N are random sample generate by
	monodimensional gaussian distribution with mean=0 and std=1.
	:param mean:  Array of means of the multivariate distributions, (Nxndim)
	:param covariance_matrix:  Array of covariance matrixes (Nxndimxndim)
	:param Npersample:  Number of object to sample for each multivariate distribution in the list
	:param flatten:  Decide if the output matrix should be flattened or not
	:return: If flatten is true, return a np array with dimension ndimxNxNpersample,  otherwise NxndimxNpersample.
	Example:
		A = np.array([10,10])
		B = np.array([0,0])
		C = np.array([-10,-10])

		covA = np.array([[0.1,0],[0,0.1]])
		covB = np.array([[0.1,0],[0,0.1]])
		covC = np.array([[1,0.8],[0.8,1]])

		mean = np.vstack((A,B,C)) (dim 3x2)
		cov_list = np.dstack((covA,covB,covC)).T  (dim 3x2x2)

		x,y=Mutivariate(mean, cov_list, Npersample=10, flatten=True)
		(x and y contain all the sampled values)

		sample=Mutivariate(mean, cov_list, Npersample=10, flatten=False)
		x0,y0 = sample[0] 		(x0 and y0 contain all the sampled values from the first multivariate Gaussian in the list)
		x1,y1 = sample[1]    (x1 and y1 contain all the sampled values from the scond multivariate Gaussian in the list)



	"""

	if len(mean.shape)==1:
		Ninput=1
		ndim=mean.shape[0]
		if not covariance_matrix.shape == (ndim, ndim): raise ValueError('Dimension of covariance matrix %s are not consistent with the dimension of mean vector %s' % (str(covariance_matrix.shape), str(mean.shape)))

	else:

		Ninput =  mean.shape[0]
		ndim   = mean.shape[1]
		if not covariance_matrix.shape==(Ninput,ndim,ndim): raise ValueError('Dimension of covariance matrix %s are not consistent with the dimension of mean vector %s'%(str(covariance_matrix.shape),str(mean.shape)))


	if Ninput==1:

		if flatten: ret=np.random.multivariate_normal(mean, covariance_matrix, Npersample).T
		else: ret=np.random.multivariate_normal(mean, covariance_matrix, Npersample)

	else:

		#More than one extraction
		if Npersample>1:
			mean               = np.repeat(mean, Npersample, axis=0)
			covariance_matrix  = np.repeat(covariance_matrix, Npersample, axis=0)

		#Cholesky decomposition
		L=np.linalg.cholesky(covariance_matrix)
		V_norm = np.random.normal(0, 1, size=Ninput*Npersample * ndim).reshape(Ninput*Npersample, ndim, 1)

		random_sample=np.matmul(L, V_norm).reshape(Ninput*Npersample,ndim)+mean


		if flatten: ret=random_sample.T
		else: ret=random_sample.reshape(Ninput, Npersample, ndim).transpose((0,2,1))

	return ret

if __name__=='__main__':

	A = np.array([10,10])
	B = np.array([0,0])
	C = np.array([-10,-10])

	covA = np.array([[0.1,0],[0,0.1]])
	covB = np.array([[0.1,0],[0,0.1]])
	covC = np.array([[1,0.8],[0.8,1]])

	mean = np.vstack((A,B,C))
	mean_f = A

	print('Mean',mean.shape)


	cov_list = np.dstack((covA,covB,covC)).T
	cov_list_f = covA

	print('Cov',cov_list.shape)

	Vl, Vb=Mutivariate(mean_f, cov_list_f, Npersample=5)

	import matplotlib.pyplot as plt
	plt.scatter(Vl, Vb)

	A=Mutivariate(mean, cov_list, Npersample=10, flatten=False)
	print(A.shape)


	plt.show()
