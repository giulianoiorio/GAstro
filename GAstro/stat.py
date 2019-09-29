################################################
#
#
#
# Utils for statistical stuff
#
#
#
#
################################################
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.covariance import MinCovDet


def boostrap(arr, fstatistic=np.nanmean, N=1000):
	"""

	:param arr:  at least 1D array
	:param fstatistic: a user-defined function which takes a ND array of values (should support the argmunent axis), and outputs a single numerical statistic. This function will be called on the values in each bin. Empty bins will be represented by function([]), or NaN if this returns an error.
	:param N: number of sampling
	:return:
	"""

	arr = np.atleast_1d(arr)
	boot_random = np.random.choice(arr, size=(N, len(arr)), replace=True)
	stat_array = fstatistic(boot_random,axis=1)

	mean = np.mean(stat_array)
	std  = np.std(stat_array)

	return mean, std


def mad(arr,c=1.482602218505602,axis=None):
	"""
	Estimate the median and the std through the median deviation around median.
	:param arr: at least 1D array
	:param c: factor to pass from the definiton of mad to a gaussian std.
	:param axis: Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.
	:return: Median, MAD
	"""
	arr = np.atleast_1d(arr)
	med = np.nanmedian(arr,axis=axis)

	return med, c*np.nanmedian(np.abs(arr - med),axis=axis)

def calc_covariance(*args):
	"""
	Estimate the mean vector, the vector of std and the vector of correlation_coeff (sigma_xy/(sigma_x*sigma_y)) for a series of samples
	:param args: Samples (they need to have the same dimension).
	:return: mean, std, corr_coeff
	"""

	Nfeature = len(args)
	X    = np.vstack(args)
	Cov  = np.cov(X)
	Std  = np.sqrt(np.diag(Cov))
	Mean = np.mean(X,axis=1)

	rho = []
	for i in range(Nfeature):
		for j in range(i+1, Nfeature):
			rho.append(Cov[i,j]/np.sqrt(Cov[i,i]*Cov[j,j]))

	return Mean, Std, rho


def recursive_mean(value, old_mean, N):
	"""
	Estimate the  mean value using the mean of a sample of len N and a new value.
	:param value:   Array containing numbers whose the mean is deseride. If a is not an array, a conversion is attempted.
	:param old_mean: Array containing the value of the mean so far.
	:param N: Number of objects used to estimate the mean (including the value in  input)
	:return: The mean of the sample containing the old objects and the new one.
	"""

	new_mean = value/N + (N-1)/N * old_mean

	return new_mean



def recursive_std(value, old_std, N, dof=0):
	"""
	 where std^2 = (1/(N-dof))*(sum (xi - <x>)^2)
	:param value:
	:param old_std:
	:param N:
	:param dof:
	:return:
	"""
	l = N - dof
	A = (l - 1) / (l)
	B = (N - 1) / (l * N)

	var_a = A * old_std * old_std
	vec_diff = (value - old_mean)
	var_b = B * (vec_diff * vec_diff)

	return np.sqrt(var_a + var_b)

def recursive_cov(value, old_cov, old_mean, N, dof=1):
	"""
	where cov = (1/(N-dof))*(sum (Xi - <X>)(Xi-<X>)T)
	:param value:
	:param old_cov:
	:param old_mean:
	:param N:
	:param dof:
	:return:
	"""
	l = N - dof
	A = (l - 1) / (l)
	B = (N - 1) / (l * N)

	cov_a = A * old_cov
	vec_diff = value - old_mean
	cov_b = B * np.outer(vec_diff, vec_diff)

	return cov_a + cov_b

if __name__=='__main__':

	print(len(_key_list_obs))
