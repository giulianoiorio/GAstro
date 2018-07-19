################################################
#
#
#
# Utils to plot stuff
#
#
#
#
################################################
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_ellipse(mean,covar,ax=None):

	v, w = linalg.eigh(covar)
	v = 2*np.sqrt(2)*np.sqrt(v)
	u = w[0] / linalg.norm(w[0])
	angle=np.arctan(u[1]/u[0])*(180./np.pi)
	ell = mpl.patches.Ellipse(mean, v[0], v[1], 180+angle)

