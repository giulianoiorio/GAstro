################################################
#
#
#
# Utils to fit stuff
#
#
#
#
################################################
import numpy as np
from xdgmm import XDGMM

def GMM(ai, aj, ak, eai, eaj, eak, n_components=1, method="Bovy", mu=None, V=None, weights=None, fit=True):

	X = np.vstack([ai, aj, ak]).T
	Xerr = np.zeros(X.shape + X.shape[-1:])
	diag = np.arange(X.shape[-1])
	Xerr[:, diag, diag] = np.vstack([eai** 2, eaj ** 2, eak**2]).T

	xdgmm = XDGMM(n_components=n_components, method=method, mu=mu, V=V, weights=weights)

	if fit:
		xdgmm.fit(X, Xerr)

	LogL=xdgmm.logL(X, Xerr)

	return xdgmm, LogL

def velocity_ellipsoid(vi, vj, vk, evi, evj, evk):


	xdgmm, _ = GMM(vi, vj, vk, evi, evj, evk, n_components=1, method="Bovy", fit=True)

	mean = xdgmm.mu[0]
	cov  = xdgmm.V[0]


	covii, covjj, covkk = np.diag(cov)

	covij, covik=  cov[0,1:]
	covjk =  cov[1,2]

	alphaij= (0.5*np.arctan( (2*covij)/(covii-covjj) ))*(180./np.pi)
	alphaik= (0.5*np.arctan( (2*covik)/(covii-covkk) ))*(180./np.pi)
	alphajk= (0.5*np.arctan( (2*covjk)/(covjj-covkk) ))*(180./np.pi)


	return  mean, cov, np.array([alphaij,alphaik,alphajk]), xdgmm

