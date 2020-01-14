import numpy as np


class Extinction:


	def __init__(self,R0=3.1):
		"""
		Class to estimate the k factor for the reddening in the Gaia Band based on Eq.1 of Babusiaux+18 paper:
		E.g. G_0=G-Kg*ebv, Note that in Babusiaux Ax=kx*A0 where  A0=R0*ebv, the k estiamted with this class are already
		multiplited by R0.
		:param R0:  R factor so that A0=R0*EBV
		"""

		self.R0=R0

		#G band
		self._c1g =  0.9761
		self._c2g = -0.1704
		self._c3g =  0.0086
		self._c4g =  0.0011
		self._c5g = -0.0438
		self._c6g =  0.0013
		self._c7g =  0.0099
		self._cg_list = (self._c1g, self._c2g, self._c3g, self._c4g, self._c5g, self._c6g, self._c7g)

		#B band
		self._c1b = 1.1517
		self._c2b = -0.0871
		self._c3b = -0.0333
		self._c4b = 0.0173
		self._c5b = -0.0230
		self._c6b = 0.0006
		self._c7b = 0.0043
		self._cb_list = (self._c1b, self._c2b, self._c3b, self._c4b, self._c5b, self._c6b, self._c7b)

		#R band
		self._c1r = 0.6104
		self._c2r = -0.0170
		self._c3r = -0.0026
		self._c4r = -0.0017
		self._c5r = -0.0078
		self._c6r = 0.00005
		self._c7r = 0.0006
		self._cr_list = (self._c1r, self._c2r, self._c3r, self._c4r, self._c5r, self._c6r, self._c7r)

		#B-R correction
		#(B-R)_c = (B-Kb*A0) - (R-Kr*A0) = (B-R) - (Kb - Kr)*A0 = (B-R) -kbr*A0 -> Kbr = Kb-Kr, so c1br = c1b-c1r

		#BR
		self._c1br = self._c1b - self._c1r
		self._c2br = self._c2b - self._c2r
		self._c3br = self._c3b - self._c3r
		self._c4br = self._c4b - self._c4r
		self._c5br = self._c5b - self._c5r
		self._c6br = self._c6b - self._c6r
		self._c7br = self._c7b - self._c7r
		self._cbr_list = (self._c1br, self._c2br, self._c3br, self._c4br, self._c5br, self._c6br, self._c7br)

	def _k(self, band, bp_rp, ebv):
		"""
		Ax=K*R0*ebv
		:param band:
		:param bp_rp:
		:param ebv:
		:return:
		"""

		bp_rp = np.atleast_1d(bp_rp)
		ebv = np.atleast_1d(ebv)

		x=bp_rp
		A0=self.R0*ebv

		if band=='g':  c=self._cg_list
		elif band=='r':   c=self._cr_list
		elif band=='b':   c=self._cb_list
		elif band=='br':   c=self._cbr_list

		return c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*A0 + c[5]*A0*A0 + c[6]*x*A0

	def kg(self, bp_rp, ebv):

		return self.R0*self._k('g',bp_rp,ebv)

	def kb(self, bp_rp, ebv):

		return self.R0*self._k('b', bp_rp, ebv)

	def kr(self, bp_rp, ebv):

		return self.R0*self._k('r', bp_rp, ebv)

	def kbr(self, bp_rp, ebv):

		return self.R0*self._k('br', bp_rp, ebv)

	def kbr_iterative(self, bp_rp, ebv, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		bp_rp_obs=bp_rp
		bp_rp_old=np.copy(bp_rp_obs)
		_k=np.zeros_like(bp_rp_obs)
		idx_continue = np.ones_like(bp_rp_obs, dtype=np.bool)
		bp_rp_new = np.zeros_like(bp_rp_old)

		for i in range(Nmax):

			_k[idx_continue]=self.kbr(bp_rp_old[idx_continue], ebv[idx_continue])
			bp_rp_new[idx_continue] =  bp_rp_obs[idx_continue]-_k[idx_continue]*ebv[idx_continue]
			abs_toll = np.abs(bp_rp_new-bp_rp_old)
			rel_toll = abs_toll/bp_rp_old
			idx_continue =  (abs_toll>abs_tollerance) | (rel_toll>rel_tollerace)
			if np.sum(idx_continue)==0: break
			else: bp_rp_old=bp_rp_new

		return _k

	def kbr_iterative_error_sample(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		if bp_rp_error is None: bp_rp_s=np.repeat(bp_rp, Nerror)
		else:  bp_rp_s = np.random.normal(np.repeat(bp_rp,Nerror), np.repeat(bp_rp_error, Nerror))

		if ebv_error is None: ebv_s=np.repeat(ebv, Nerror)
		else: ebv_s = np.random.normal(np.repeat(ebv, Nerror), np.repeat(ebv_error, Nerror))

		kbr_s = self.kbr_iterative(bp_rp_s, ebv_s, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		return kbr_s, ebv_s

	def kbr_iterative_error(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):


		k_s,_ = self.kbr_iterative_error_sample(bp_rp=bp_rp, ebv=ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nerror, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		k_s = k_s.reshape(-1, Nerror).T

		k_mean, k_std = np.mean(k_s,axis=0), np.std(k_s,axis=0)


		return k_mean, k_std

	def Abr_iterative_error_sample(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		k_s, ebv_s = self.kbr_iterative_error_sample(bp_rp=bp_rp, ebv=ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nerror, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		A_s = k_s*ebv_s

		return A_s


	def kg_iterative(self, bp_rp, ebv, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		k_bp_rp_0 = self.kbr_iterative(bp_rp, ebv, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)
		bp_rp_0 = bp_rp - k_bp_rp_0*ebv

		return self.kg(bp_rp_0, ebv)

	def kg_iterative_error_sample(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		if bp_rp_error is None: bp_rp_s=np.repeat(bp_rp, Nerror)
		else:  bp_rp_s = np.random.normal(np.repeat(bp_rp,Nerror), np.repeat(bp_rp_error, Nerror))

		if ebv_error is None: ebv_s=np.repeat(ebv, Nerror)
		else: ebv_s = np.random.normal(np.repeat(ebv, Nerror), np.repeat(ebv_error, Nerror))

		k_s = self.kg_iterative(bp_rp_s, ebv_s, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		#bp_rp_0_s = self.kbr_iterative(bp_rp_s, ebv_s, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)
		#k_s = self.kg(bp_rp_0_s, ebv_s)

		#bp_rp_0_s = self.kbr_iterative(bp_rp_s, ebv_s, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)
		#k_s = self.kg(bp_rp_0_s, ebv_s)

		return k_s, ebv_s


	def kg_iterative_error(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):


		k_s,_ = self.kg_iterative_error_sample(bp_rp=bp_rp, ebv=ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nerror, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		k_s = k_s.reshape(-1, Nerror).T

		k_mean, k_std = np.mean(k_s,axis=0), np.std(k_s,axis=0)


		return k_mean, k_std

	def Ag_iterative_error_sample(self, bp_rp, ebv, bp_rp_error=None, ebv_error=None, Nerror=1000, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		k_s, ebv_s = self.kg_iterative_error_sample(bp_rp=bp_rp, ebv=ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nerror, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		A_s = k_s*ebv_s

		return A_s

#error on magnitude

#premission ZP from
ZP_g_VEGA_err=0.0017850023
ZP_bp_VEGA_err=0.0013918258
ZP_rp_VEGA_err=0.0019145719
#Gaia DR2 passbands and zero points used for the magnitudes published in Gaia DR2.
#from https://www.cosmos.esa.int/web/gaia/iow_20180316
#The magnitude inGaia is mag = -2.5*log10(Flux) + ZP
# som the error is dmag = sqrt(2.5*log10(e)*dFlux/Flux)**2 + ZP_error**2)

def mag_err(flux, flux_err, ZP_err):

	err_A = 2.5*np.log10(np.exp(1))*(flux_err/flux)
	err_B = ZP_err*ZP_err

	return np.sqrt(err_A*err_A + err_B*err_B)

def g_err(flux, flux_err):

	return mag_err(flux, flux_err, ZP_g_VEGA_err)

def bp_err(flux, flux_err):
	return mag_err(flux, flux_err, ZP_bp_VEGA_err)

def rp_err(flux, flux_err):
	return mag_err(flux, flux_err, ZP_rp_VEGA_err)

def bp_rp_err(flux_bp, flux_err_bp, flux_rp, flux_err_rp):

	_bp_err = bp_err(flux_bp, flux_err_bp)
	_rp_err = rp_err(flux_rp, flux_err_rp)

	return np.sqrt(_bp_err*_bp_err + _rp_err*_rp_err)

_ext_class_for_gc = Extinction()
def gc_sample_babusiaux(g,  ebv, bp_rp, g_error=None, bp_rp_error=None, ebv_error=None, Nsample=1000):
	"""
	Sample the unreddend g mag using the iterative formula on Eq.1 of Babusiaux+18 paper:
	:param g: Gaia DR2 measured g
	:param bp_rp:  GAIA DR2 bp_rp
	:param ebv: Reddening
	:param g_error: error on g
	:param bp_rp_error: error on bp_rp
	:param ebv_error: Error onebv (for Scheleghel is 0.16*ebv)
	:param Nsample: number of sample to draw per data
	:return: a 1D array withdimension length(g)*Nsample with gc.
	"""

	g = np.repeat(g, Nsample)
	if g_error is not None:
		g_error = np.repeat(g_error, Nsample)
		g = np.random.normal(g, g_error)

	Ag=_ext_class_for_gc.Ag_iterative_error_sample(bp_rp, ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nsample, Nmax=1000)
	gc = g-Ag

	return gc

def gc_sample_kg(g, ebv, kg=2.27, g_error=None,  kg_error=0.3, ebv_error=None, Nsample=1000):

	"""
	Sample the unreddened g mag as gc=g - kg*ebv

	:param g: Gaia DR2 measured g
	:param ebv: Reddening
	:param kg:  Kg reddening Factor (Ag=kg*ebv). It is used if bp_rp is None. The default value of 2.27 is from Iorio+19.
	:param g_error: error on g
	:param kg_error:  error on kg. The default value of 0.3 is estimated from the Hierachical bayesian fit using the AG from SOS.
	:param ebv_error: Error on ebv (for Scheleghel is 0.16*ebv)
	:param Nsample: number of sample to draw per data
	:return: a 1D array withdimension length(g)*Nsample with gc
	"""

	g = np.repeat(g, Nsample)
	if g_error is not None:
		g_error = np.repeat(g_error, Nsample)
		g = np.random.normal(g, g_error)

	kg = np.repeat(kg, Nsample)
	if kg_error is not None:
		kg_error = np.repeat(kg_error, Nsample)
		kg = np.random.normal(kg, kg_error)


	ebv = np.repeat(ebv, Nsample)
	if ebv_error is not None:
		ebv_error = np.repeat(ebv_error, Nsample)
		ebv = np.random.normal(ebv, ebv_error)

	gc = g - kg*ebv

	return gc




def gc_sample(g, ebv, bp_rp=None, kg=2.27, g_error=None, bp_rp_error=None, kg_error=0.3, ebv_error=0.15, Nsample=1000):
	"""
	Sample the unreddend g mag

	:param g: Gaia DR2 measured g
	:param ebv: Reddening
	:param bp_rp:  If not None GAIA DR2 bp_rp and the iterative estimate of formula on Eq.1 of Babusiaux+18 paper is used
	:param kg:  Kg reddening Factor (Ag=kg*ebv). It is used if bp_rp is None. The default value of 2.27 is from Iorio+19.
	:param g_error: error on g
	:param bp_rp_error: error on bp_rp
	:param kg_error:  error on kg. The default value of 0.3 is estimated from the Hierachical bayesian fit using the AG from SOS.
	:param ebv_error: Error on ebv (for Scheleghel is 0.16*ebv)
	:param Nsample: number of sample to draw per data
	:return: a 1D array withdimension length(g)*Nsample with gc
	"""


	if bp_rp is not None: gc = gc_sample_babusiaux(g=g, ebv=ebv, bp_rp=bp_rp, g_error=g_error, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nsample=Nsample)
	else: gc = gc_sample_kg(g=g, ebv=ebv, kg=kg, g_error=g_error,  kg_error=kg_error, ebv_error=ebv_error, Nsample=Nsample)

	return gc



#Correcting for rotating frame
#_omega_x   = −0.086
#_omega_y   = −0.114
#_omega_z   = -0.037
#_omega_err =  0.025

_omega_x = -0.086
_omega_y = -0.114
_omega_z = -0.037
_omega_err = 0.025

def pm_frame_correction(pmra, pmdec, ra, dec,  omega_x=_omega_x, omega_y=_omega_y, omega_z=_omega_z):
	"""
	Correct for the spuriuous pm due to the rotation of the frame of reference in gaia (only for G<12) (see known issue)
	:param pmra: mas/yr
	:param pmdec: mas/yr
	:param ra: radians
	:param dec:  radians
	:param omege_x: mas/yr
	:param omega_y:  mas/yr
	:param omega_z:  mas/yr
	:return: Corrected pmra and pmdec
	"""
	sra=np.sin(ra)
	cra=np.cos(ra)
	sdec=np.sin(dec)
	cdec=np.cos(dec)


	pmra_c  = pmra + omega_x*sdec*cra + omega_y*sdec*sra - omega_z*cdec
	pmdec_c = pmdec - omega_x*sra + omega_y*cra

	return pmra_c, pmdec_c


def sample_pm_frame_correction(pmra, pmdec, ra, dec,  Nsample=1000, degree=True, omega_x=_omega_x, omega_y=_omega_y, omega_z=_omega_z, domega_x=_omega_err, domega_y=_omega_err, domega_z=_omega_err):
	"""
	Sample the Correction for the spuriuous pm due to the rotation of the frame of reference in gaia (only for G<12) (see known issue)
	:param pmra: mas/yr
	:param pmdec: mas/yr
	:param ra: degree (if degree =True) or radians
	:param dec: degree (if degree =True) or radians
	:param Nsample: Number of elements to sample
	:param degree:  If true ra nd dec are in degree otherwise in radians.
	:param omega_x: mas/yr
	:param omega_y: mas/yr
	:param omega_z: mas/yr
	:param domega_x: mas/yr
	:param domega_y: mas/yr
	:param domega_z: mas/yr
	:return: Sampled corrected pmra, pmdec
	"""


	pmra=np.repeat(pmra, Nsample)
	pmdec=np.repeat(pmdec, Nsample)
	if degree:
		ra=np.radians(ra)
		dec=np.radians(dec)
	ra = np.repeat(ra, Nsample)
	dec = np.repeat(dec, Nsample)

	N_to_draw = len(pmra)

	omega_x_list = np.random.normal(omega_x, domega_x, N_to_draw)
	omega_y_list = np.random.normal(omega_y, domega_y, N_to_draw)
	omega_z_list = np.random.normal(omega_z, domega_z, N_to_draw)

	pmra_c, pmdec_c = pm_frame_correction(pmra, pmdec, ra, dec, omega_x=omega_x_list, omega_y=omega_y_list, omega_z=omega_z_list)

	return pmra_c, pmdec_c


#Total external and internal error
_k_error = 1.08
_sigma_s_gl13_pm = 0.032
_sigma_s_gu13_pm = 0.066
_sigma_s_gl13_parallax = 0.021
_sigma_s_gu13_parallax = 0.043

def tot_err(sigma_i, k, sigma_s):
	"""
	The errore reported in Gaia are just formal internal error, the finale errore should be
	sigma = sqrt(k*k*sigma_i*sigma_i + sigma_s*sigma_s) as discussed in https://www.cosmos.esa.int/documents/29201/1770596/Lindegren_GaiaDR2_Astrometry_extended.pdf/1ebddb25-f010-6437-cb14-0e360e2d9f09
	where sigma_s are the standard deviation of the systematic error
	:param sigma_i: internal Gaia uncertain
	:param k: factor to correct the internal uncertain to have the real standard deviation.
	:param sigma_s:  systematic error.
	:return:
	"""
	return np.sqrt(k*k*sigma_i*sigma_i + sigma_s*sigma_s)

def total_error_pm(pm_err, g):
	"""
	Estimate the total error on proper motion from the Gaia pm error estimate
	:param pm_err:  Gaia DR2 proper motion estimate
	:param g: Gaia DR2 G magnitude.
	:return:  The total pm error
	"""
	final_err = np.where(g<13, tot_err(pm_err, k=_k_error, sigma_s=_sigma_s_gl13_pm), tot_err(pm_err, k=_k_error, sigma_s=_sigma_s_gu13_pm))

	return final_err

if __name__=="__main__":


	ext = Extinction()
	ebv=np.array([0.02,0.2,0.8])
	bp_rp=np.array([0.7,0.7,0.7])

	kbr=ext.kbr(bp_rp,ebv)
	print(kbr)
	kbri=ext.kbr_iterative(bp_rp,ebv)
	print(kbri)

	print()

	kg=ext.kg(bp_rp,ebv)
	print(kg)
	kgi=ext.kg_iterative(bp_rp,ebv)
	print(kgi)
	kgi=ext.kg_iterative_error(bp_rp,ebv, bp_rp_error=0.1*bp_rp, ebv_error=np.array([0.15,0.15,0.15]))
	print(kgi)

	kgi=ext.kg_iterative_error_sample(bp_rp[0],ebv[0], bp_rp_error=0.1*bp_rp[0], ebv_error=np.array([0.15,]))
	print(kgi)

	flux_g = 43935.6612672334
	flux_g_err = 773.285765136811
	print(g_err(flux_g,flux_g_err))

	flux_b = 23446.7515446426
	flux_b_err = 1598.28661272363
	print(bp_err(flux_b,flux_b_err))


	g = 18.0858
	bp_rp = 1.40969
	ebv = 0.552063
	gc=gc_sample(g, ebv,bp_rp,  g_error=None, bp_rp_error=None, ebv_error=0.16*ebv, Nsample=100000)
	#print(gc)
	print(np.mean(gc),np.std(gc))


	flux_g=1905.08067455575
	flux_g_err=14.0150095217651
	flux_bp=805.81814324249
	flux_bp_err=24.6086486896657
	flux_rp=1715.23307621281
	flux_rp_err=47.7425642381516
	g_error=g_err(flux_g,flux_g_err)
	bp_rp_error=bp_rp_err(flux_bp,flux_bp_err, flux_rp,flux_rp_err)

	print(g_error)
	print(bp_rp_error)

	gc=gc_sample(g, ebv, bp_rp, g_error=g_error, bp_rp_error=bp_rp_error, ebv_error=0.16*ebv, Nsample=100000)
	#print(gc)
	print(np.mean(gc),np.std(gc))

	gc=gc_sample(g, ebv, kg=2.2, g_error=g_error, ebv_error=0.16*ebv, Nsample=100000)
	#print(gc)
	print(np.mean(gc),np.std(gc))
