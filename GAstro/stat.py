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

def mad(arr,c=1.482602218505602,axis=0):

	med = np.nanmedian(arr,axis=axis)

	return med, c*np.nanmedian(np.abs(arr - med),axis=axis)
