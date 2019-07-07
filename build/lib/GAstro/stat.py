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



if __name__=='__main__':

	print(len(_key_list_obs))
