################################################
#
#
#
# Utils for analyse RRL stars
#
#
#
#
################################################
import numpy as np
import pandas as pd
import os
from xdgmm import XDGMM
import galpy.util.bovy_coords as co
#import pycam.utils as ut

#Inner import
from .stat import mad, calc_covariance
from .gaia import gc_sample
from .transform import m_to_dist
#from .gaia import _ext_class_for_gc
from .gaia import Extinction
from . import utility as ut
from . import transform as tr
from . import constant as COST
from . import gaia
#from pycam.utils import xyz_to_m

data_file_path= os.path.abspath(os.path.dirname(__file__)) + "/data/"

_ext_class_for_gc = Extinction()


def _type_1_2_division(period, offset=-0.15):
	"""
	Line separating Type1, Type2 RRL in period/amp plane as defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 2)
	NB. for amp (peak_to_peak_g) from the Gaia DR2 RRL cats use the default offset=-0.15.
	:param period: rrl period [days]
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B)
	:return: the amplitude line separing type1 and type2 (type1 amp<return, type2 amp>return)
	"""

	_C0 = -2.477
	_C1 = -22.046
	_C2 = -30.876

	logP = np.log10(period)

	return _C0 + _C1*logP + _C2*logP*logP + offset

def _type12_cut(period, offset=-0.15):
	"""
	Extra cut on amp used in Belokurov+18 (MNRAS.477.1472B) to define Type1/Type2 RRL stars (Eq. 1C)
	NB. for amp (peak_to_peak_g) from the Gaia DR2 RRL cats use the default offset=-0.15.
	:param period: rrl period [days]
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B)
	:return:  the amplitude line used to define Type1/Type2  (amp>return the stars are not defined as Type1 or Type2)
	"""

	_A = 3.75
	_B = -3.5

	return _A + _B*period + offset

def classify_type12(period, amp, offset=-0.15, extra_cut=True, period_range=(0.43,0.9), amp_range=None):

	"""
	Taking the period and amplitude clasify the stars as Type1 or Type2 RRLs based on the definition in Belokurov+18 (MNRAS.477.1472B)
	NB. for amp (peak_to_peak_g) from the Gaia DR2 RRL cats use the default offset=-0.15.
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range: extra cut on period range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: extra cut on amp range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:return: A 1D array with classication as Type1 (1), Type2 (2) or unclassified (numpy NAN).
	"""


	idx_general = np.ones_like(period, dtype=bool)

	if extra_cut:                     idx_general*=amp<_type12_cut(period,offset)
	if period_range is not None:      idx_general*=(period>np.min(period_range))&(period<np.max(period_range))
	if amp_range is not None:         idx_general*=(amp>np.min(amp_range))&(period<np.max(amp_range))

	amp_separation = _type_1_2_division(period, offset)

	idx_type1 = idx_general & (amp<amp_separation)
	idx_type2 = idx_general & (amp>amp_separation)


	return_col = np.zeros_like(period)
	return_col[idx_type1] = 1
	return_col[idx_type2] = 2
	return_col[return_col==0] = np.nan

	return return_col

def is_type1(period, amp, offset=-0.15, extra_cut=True, period_range=(0.43,0.9), amp_range=None):

	"""
	Check what stars are Type1 RRL stars, following the definition in Belokurov+18 (MNRAS.477.1472B)
	NB. for amp (peak_to_peak_g) from the Gaia DR2 RRL cats use the default offset=-0.15.
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range: extra cut on period range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: extra cut on amp range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:return: A boolean array with the same len of period and amp with True where the stars can be classified as Type1 RRL
	"""


	col_type = classify_type12(period, amp, offset, extra_cut, period_range, amp_range)

	return col_type==1

def is_type2(period, amp, offset=-0.15, extra_cut=True, period_range=(0.43, 0.9), amp_range=None):
	"""
	Check what stars are Type2 RRL stars, following the definition in Belokurov+18 (MNRAS.477.1472B)
	NB. for amp (peak_to_peak_g) from the Gaia DR2 RRL cats use the default offset=-0.15.
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range: extra cut on period range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: extra cut on amp range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:return: A boolean array with the same len of period and amp with True where the stars can be classified as Type2 RRL
	"""

	col_type = classify_type12(period, amp, offset, extra_cut, period_range, amp_range)

	return col_type == 2

def is_HASP(period, amp, offset=-0.15, period_range=(0.43,0.48), amp_range=(0.9,1.45)):
	"""
	Check what stars can be considered HASP. based on the definition by Belokurov+18 (MNRAS.477.1472B) Eq.3.
	In general this can be used to check if the stars are in a given period-amplitude box.
	NB: remember that the the amp range is corrected for the offset (default value is -0.15), if the offset is not needed put it to 0.
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param period_range: period box to be used to select HASP RRL. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: amp box to be used to select HASP RRL. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:return: A boolean array with the same len of period and amp with True where the stars are in the given period-amplitude box.
	"""

	idx_general = np.ones_like(period, dtype=bool)
	if period_range is not None: idx_general*=(period>np.min(period_range))&(period<np.max(period_range))
	if amp_range is not None:  idx_general*=(amp>(np.min(amp)+offset))&(amp<(np.max(amp)+offset))

	return idx_general

def f_type1(period, amp, offset=-0.15, extra_cut=True, period_range=(0.43, 0.9), amp_range=None, consider_all=False):
	"""
	Estimate the fraction of Type1 RRL stars (Bases on the classification by Belokurov+18 (MNRAS.477.1472B) )
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range: extra cut on period range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: extra cut on amp range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param consider_all: if True, the fraction is N1/Ntot where Ntot is the len of the period array in input. If
	False the fraction is N1/(N1+N2) so we do not consider the number of objects that have not been classified.
	:return:  fraction of Type1 RRL stars.
	"""

	classification_array=classify_type12(period, amp, offset=offset, extra_cut=extra_cut, period_range=period_range, amp_range=amp_range)
	N1 = np.sum(classification_array == 1)
	if consider_all:
		frac = N1/len(period)
	else:
		N2 = np.sum(classification_array==2)
		Ntot = N1+N2
		frac = N1 / Ntot

	return frac

def f_type2(period, amp, offset=-0.15, extra_cut=True, period_range=(0.43, 0.9), amp_range=None, consider_all=False):
	"""
	Estimate the fraction of Type2 RRL stars (Bases on the classification by Belokurov+18 (MNRAS.477.1472B) )
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range: extra cut on period range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range: extra cut on amp range. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param consider_all: if True, the fraction is N2/Ntot where Ntot is the len of the period array in input. If
	False the fraction is N2/(N1+N2) so we do not consider the number of objects that have not been classified.
	:return:  fraction of Type1 RRL stars.
	"""

	classification_array = classify_type12(period, amp, offset=offset, extra_cut=extra_cut, period_range=period_range,
										   amp_range=amp_range)
	N2 = np.sum(classification_array == 2)
	if consider_all:
		frac = N2 / len(period)
	else:
		N1 = np.sum(classification_array == 1)
		Ntot = N1 + N2
		frac = N2 / Ntot

	return frac

def f_HASP(period, amp, offset=-0.15, extra_cut=True, period_range_HASP=(0.43,0.48), amp_range_HASP=(0.9,1.45), period_range_type12=(0.43, 0.9), amp_range_type12=None, consider_all=False):
	"""
	Estimate the fraction of HASP RRL stars (Bases on the classification by Belokurov+18 (MNRAS.477.1472B) )
	:param period: rrl period [days]
	:param amp:  rrl amplitude (peak_to_peak)
	:param offset: Offset of the amplitude (peak_to_peak) wrt to the function defined in Belokurov+18 (MNRAS.477.1472B) (Eq. 1C)
	:param extra_cut: if True, apply the extra_cut used in Belokurov+18 (MNRAS.477.1472B) to classify stars
	:param period_range_HASP: period box to be used to select HASP RRL. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range_HASP: amp box to be used to select HASP RRL. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param period_range_type12: extra cut on period range to be used to estimate type12 RRL. If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param amp_range_type12: extra cut on amp range to be used to estimate type12 RRL If not None, it need to be a tuple with two values. Only stars
	within these values will be taken into account for the classification.
	:param consider_all: if True, the fraction is NHASP/Ntot where Ntot is the len of the period array in input. If
	False the fraction is NHASP/(N1+N2) so we do not consider the number of objects that have not been classified.
	:return:  fraction of HASP RRL stars.
	"""

	NHASP = np.sum(is_HASP(period, amp, offset=offset, period_range=period_range_HASP, amp_range=amp_range_HASP))

	if consider_all:
		classification_array = classify_type12(period, amp, offset=offset, extra_cut=extra_cut, period_range=period_range_type12,
										   amp_range=amp_range_type12)
		Ntot = np.sum( (classification_array == 1) | (classification_array == 1) )
	else:
		Ntot=len(period)

	frac = NHASP/Ntot

	return frac

####
def _sample_p_phi31(N, p, phi31, p_err=0, phi31_err=0, corr_p_phi31=0):
	"""
	Auxilary funvtion for sample_p_phi31_RRab and sample_p_phi31_RRc.
	It samples using a random realisation from a multivariate Gaussian.
	:param N: Size of sample to generate
	:param p: Given period
	:param p_err: Given period error
	:param phi31: Given phi31_g
	:param phi31_err: Given phi31_g  error
	:param corr_p_phi31: Correlation between period and phi31
	:return: A numpy array with size (N,2), with the sampled period in the first column and sample phi31 in the second.
	"""

	covariance = p_err * phi31_err * corr_p_phi31
	cov_matrix = [[p_err * p_err, covariance], [covariance, phi31_err * phi31_err]]
	Xreturn = np.random.multivariate_normal(mean=[p, phi31], cov=cov_matrix, size=N)

	return Xreturn

def sample_p_phi31_RRab(N, p=None, p_err=0, phi31=None, phi31_err=0, corr_p_phi31=0, xdgmm_fit=None):
	"""
	Sample values of Period and Phi31 for RRab star. The method used depends on the value of p and phi31.
	Case 1: Both p and phi31 are given.
		The p and phi31 will be sampled from a multivariate normal using p_err, phi31_err and corr_p_phi31
		as standard deviations and correlation between values. (xdgmm_fit is not used)
	Case 2: Neither p nor phi31 are given.
		The values are sampled from the xdgmm_fit in input or if it is None from the standard 5 component gaussians
		fitted to the clean sample of RRab Gaia SOS stars.  (p_err, phi31_err,  corr_p_phi31 are not used)
	Case 3: p or phi31 are given.
		The values are sampled from the xdgmm_fit conditioned on  the value of the parameter given in input (p or phi31) taking into account
		also its error. (the error of the parameter not given in input and corr_p_phi31 are not used).

	:param N: Size of sample to generate
	:param p: Given period
	:param p_err: Given period error
	:param phi31: Given phi31_g
	:param phi31_err: Given phi31_g  error
	:param corr_p_phi31: Correlation between period and phi31
	:param xdgmm_fit: The extreme deconvolution model from the XDGMM module. If None the one fitted to the Gaia SOS RRab RRlyare is used.
	:return: A numpy array with size (2,N), with the sampled period in the first row and sample phi31 in the second row.
	"""


	if p_err is None: p_err=0
	if phi31_err is None: phi31_err=0

	#Option-0: Both p and phi31 (and their error) are provvided
	if (p is not None) and (phi31 is not None):
		Xreturn = _sample_p_phi31(N=N, p=p, phi31=phi31, p_err=p_err, phi31_err=phi31_err, corr_p_phi31=corr_p_phi31)
	else:

		if xdgmm_fit is None:
			path_to_file = data_file_path +  "period_phi31_fit/RRab_SOS_pf_phi31.xdgmm"
			xdgmm_ab = XDGMM(filename=path_to_file)
		else:
			xdgmm_ab=xdgmm_fit


		if p is None and phi31 is None:
			Xreturn = xdgmm_ab.sample(N)
		elif p is None:
			xdgmm_ab_cond=xdgmm_ab.condition(X_input =np.array([np.nan, phi31]), Xerr_input = np.array([p_err, phi31_err]))
			Xreturn=np.zeros(shape=(N,2))
			Xreturn[:,0] = xdgmm_ab_cond.sample(N)[:,0]
			Xreturn[:,1] = np.random.normal(phi31,phi31_err,N)
		elif phi31 is None:
			xdgmm_ab_cond=xdgmm_ab.condition(X_input =np.array([p, np.nan]), Xerr_input = np.array([p_err, phi31_err]))
			Xreturn=np.zeros(shape=(N,2))
			Xreturn[:,0] = np.random.normal(p,p_err,N)
			Xreturn[:,1] = xdgmm_ab_cond.sample(N)[:,0]
		else:
			raise ValueError("You should not be there from %s"%__file__)

	return Xreturn.T

def sample_p_phi31_RRc(N, p=None, p_err=0, phi31=None, phi31_err=0, corr_p_phi31=0, xdgmm_fit=None):
	"""
	Sample values of Period and Phi31 for RRc star. The method used depends on the value of p and phi31.
	Case 1: Both p and phi31 are given.
		The p and phi31 will be sampled from a multivariate normal using p_err, phi31_err and corr_p_phi31
		as standard deviations and correlation between values. (xdgmm_fit is not used)
	Case 2: Neither p nor phi31 are given.
		The values are sampled from the xdgmm_fit in input or if it is None from the standard 5 component gaussians
		fitted to the clean sample of RRc Gaia SOS stars.  (p_err, phi31_err,  corr_p_phi31 are not used)
	Case 3: p or phi31 are given.
		The values are sampled from the xdgmm_fit conditioned on  the value of the parameter given in input (p or phi31) taking into account
		also its error. (the error of the parameter not given in input and corr_p_phi31 are not used).

	:param N: Size of sample to generate
	:param p: Given period (p1_o)
	:param p_err: Given period error
	:param phi31: Given phi31_g
	:param phi31_err: Given phi31_g  error
	:param corr_p_phi31: Correlation between period and phi31
	:param xdgmm_fit: The extreme deconvolution model from the XDGMM module. If None the one fitted to the Gaia SOS RRab RRlyare is used.
	:return: A numpy array with size (2,N), with the sampled period in the first row and sample phi31 in the second row.
	"""

	if p_err is None: p_err=0
	if phi31_err is None: phi31_err=0

	# Option-0: Both p and phi31 (and their error) are provvided
	if (p is not None) and (phi31 is not None):
		Xreturn = _sample_p_phi31(N=N, p=p, phi31=phi31, p_err=p_err, phi31_err=phi31_err, corr_p_phi31=corr_p_phi31)
	else:

		if xdgmm_fit is None:
			path_to_file = data_file_path +  "period_phi31_fit/RRc_SOS_p1_phi31.xdgmm"
			xdgmm_ab = XDGMM(filename=path_to_file)
		else:
			xdgmm_ab = xdgmm_fit

		if p is None and phi31 is None:
			Xreturn = xdgmm_ab.sample(N)
		elif p is None:
			xdgmm_ab_cond = xdgmm_ab.condition(X_input=np.array([np.nan, phi31]),
											   Xerr_input=np.array([p_err, phi31_err]))
			Xreturn = np.zeros(shape=(N, 2))
			Xreturn[:, 0] = xdgmm_ab_cond.sample(N)[:, 0]
			Xreturn[:, 1] = np.random.normal(phi31, phi31_err, N)
		elif phi31 is None:
			xdgmm_ab_cond = xdgmm_ab.condition(X_input=np.array([p, np.nan]), Xerr_input=np.array([p_err, phi31_err]))
			Xreturn = np.zeros(shape=(N, 2))
			Xreturn[:, 0] = np.random.normal(p, p_err, N)
			Xreturn[:, 1] = xdgmm_ab_cond.sample(N)[:, 0]
		else:
			raise ValueError("You should not be there from %s" % __file__)

	return Xreturn.T

####Metallicity
def sample_metallicity(period, phi31, trace, period_err=None, phi31_err=None, Nsample=1000):
	"""
	Sample the metallicity from some linear relation of kind Fe/H=Fe/H_hat + slope_fe*Period + slope_phi31*phi31  with scatter Fe/H_scatter
	:param period: RRL period (scalar or 1D array, same length of phi31). If RR ab this is the fundamental period, otherwise it is the first overtone
	:param phi31: RRL  phi31 as estimated in Gaia (scalar of 1D array, same length of period).
	:param trace: pandas DataFrame or dictionary with keywords  Fe_hat, slope_p_fe, slope_phi31_fe, Fe_scatter. If it is None
				  the default trace will be used.
	:param period_err: Error on period (same length of period, it could be None).
	:param phi31_err: Error on phi31 (same length of phi31, it could be None).
	:param Nsample: Number of samplings per data.
	:return: a flattened array with sample metallicity with dimension length(period)*Nsample
	"""

	Ndata = len(np.atleast_1d(period))
	Ntrace = len(trace)
	period = np.repeat(period, Nsample)
	phi31 = np.repeat(phi31, Nsample)


	if period_err is not None:
		period_err = np.repeat(period_err, Nsample)
		period_err = np.where(np.isnan(period_err), 0, period_err)
		period = np.random.normal(period, period_err)
	if phi31_err is not None:
		phi31_err = np.repeat(phi31_err, Nsample)
		phi31_err = np.where(np.isnan(phi31_err), 0, phi31_err)
		phi31 = np.random.normal(phi31, phi31_err)

	idx_random_trace = np.random.choice(np.arange(Ntrace), replace=True, size=Nsample*Ndata)


	Fe_hat 		= trace['Fe_hat'][idx_random_trace]
	slope_fe 	= trace['slope_p_fe'][idx_random_trace]
	slope_phi31 = trace['slope_phi31_fe'][idx_random_trace]
	Fe_scatter  = trace['Fe_scatter'][idx_random_trace]



	lin_rel 	= Fe_hat + slope_fe*period + slope_phi31*phi31
	Fe_lin  	= np.random.normal(lin_rel, Fe_scatter)

	return Fe_lin

#RRab
metallicity_RRab_default_traces={"zinn":"trace_fe_pf_rel_zinn.csv","layden": "trace_fe_pf_rel_layden_mod.csv", "marsakov":"trace_fe_pf_rel_marsakov_mod.csv", "nemec":"trace_fe_pf_rel_nemec_mod.csv"}
def sample_metallicity_RRab(period, phi31, period_err=None, phi31_err=None, Nsample=1000, trace=None, default_trace="layden"):
	"""
	Sample the metallicity from some linear relation of kind Fe/H=Fe/H_hat + slope_fe*Period + slope_phi31*phi31.
	:param period: RRL ab period (scalar or 1D array, same length of phi31).
	:param phi31: RRL ab phi31 (scalar of 1D array, same length of period).
	:param period_err: Error on period (same length of period, it could be None).
	:param phi31_err: Error on phi31 (same length of phi31, it could be None).
	:param Nsample: Number of samplings per data.
	:param trace: Linear relation trace to use. It could be a path to a csv file, a pandas dataframe or None.
				  The keyword to be used are Fe_hat, slope_p_fe, slope_phi31_fe, Fe_scatter. If it is None
				  the default trace will be used.
	:param default_trace: name of the default trace to be used (layden, marsakov, nemec).
						  Layden and Marsakov are similar, Nemec tends to give more metal rich estimate.
	:return: a flattened array with sample metallicity with dimension length(period)*Nsample
	"""
	#Load trace
	if trace is not None and isinstance(trace, str): trace = pd.read_csv(trace)
	elif trace is not None and isinstance(trace,  pd.DataFrame): trace=trace
	elif trace is not None: raise ValueError("trace option should be a path to a csv datafile or a pandas Dataframe")
	else: trace = pd.read_csv(data_file_path +  "rrl_metallicity_fit_trace/%s"%metallicity_RRab_default_traces[default_trace].lower())

	Fe_lin = sample_metallicity(period=period, phi31=phi31, trace=trace, period_err=period_err, phi31_err=phi31_err, Nsample=Nsample)


	return Fe_lin

def metallicity_RRab(period, phi31, period_err=None, phi31_err=None, Nsample=1000, trace=None, default_trace="layden", flatten=False):
	"""
	Sample the metallicity from some linear relation of kind Fe/H=Fe/H_hat + slope_fe*Period + slope_phi31*phi31.
	:param period: RRL ab period (scalar or 1D array, same length of phi31).
	:param phi31: RRL ab phi31 (scalar of 1D array, same length of period).
	:param period_err: Error on period (same length of period, it could be None).
	:param phi31_err: Error on phi31 (same length of phi31, it could be None).
	:param Nsample: Number of samplings per data.
	:param trace: Linear relation trace to use. It could be a path to a csv file, a pandas dataframe or None.
				  The keyword to be used are Fe_hat, slope_p_fe, slope_phi31_fe, Fe_scatter. If it is None
				  the default trace will be used.
	:param default_trace: name of the default trace to be used (layden, marsakov, nemec).
						  Layden and Marsakov are similar, Nemec tends to give more metal rich estimate.
	:param flatten: If True, the first item to return is flatten (length length(period)*Nsample) othwerwise it is a (length(period), Nsample) 2D array
	:return: The first item is the dictionary with the statistics (mean, std, median, q16, q84 (~ 1sigma), q2, q97 (~ 2sigma), q02, q99 (~3 sigma).
			The second item is an array (1D or 2D, depending on the flatten parameter) containing all the sampled values.
	"""


	Fe_lin  	= sample_metallicity_RRab(period=period, phi31=phi31, period_err=period_err, phi31_err=phi31_err, Nsample=Nsample, trace=trace, default_trace=default_trace)

	ret_stat = {}
	_Fe_tmp  = Fe_lin.reshape(-1, Nsample)
	ret_stat["mean"] = np.mean(_Fe_tmp, axis=1)
	ret_stat["std"] = np.std(_Fe_tmp, axis=1)
	ret_stat["median"] = np.percentile(_Fe_tmp, q=50, axis=1)
	ret_stat["q16"] = np.percentile(_Fe_tmp, q=16, axis=1)
	ret_stat["q84"] = np.percentile(_Fe_tmp, q=84, axis=1)
	ret_stat["q97"] = np.percentile(_Fe_tmp, q=97.7, axis=1)
	ret_stat["q2"] = np.percentile(_Fe_tmp, q=2, axis=1)
	ret_stat["q99"] = np.percentile(_Fe_tmp, q=99.8, axis=1)
	ret_stat["q02"] = np.percentile(_Fe_tmp, q=0.2, axis=1)

	if flatten==False: Fe_lin=_Fe_tmp


	return ret_stat, Fe_lin

#RRc
metallicity_RRc_default_traces={"gc": "trace_fe_pf_rel_RRc_GC_mod.csv"}
def sample_metallicity_RRc(period, phi31, period_err=None, phi31_err=None, Nsample=1000, trace=None, default_trace="gc"):
	"""

	:param period: RRL c first overtone  period (scalar or 1D array, same length of phi31).
	:param phi31: RRL c phi31 as estimated in Gaia (scalar of 1D array, same length of period).
	:param period_err: Error on period (same length of period, it could be None).
	:param phi31_err: Error on phi31 (same length of phi31, it could be None).
	:param Nsample: Number of samplings per data.
	:param trace: Linear relation trace to use. It could be a path to a csv file, a pandas dataframe or None.
				  The keyword to be used are Fe_hat, slope_p_fe, slope_phi31_fe, Fe_scatter. If it is None
				  the default trace will be used.
	:param default_trace: name of the default trace to be used (only gc is available).
	:return: a flattened array with sample metallicity with dimension length(period)*Nsample
	"""

	# Load trace
	if trace is not None and isinstance(trace, str): trace = pd.read_csv(trace)
	elif trace is not None and isinstance(trace, pd.DataFrame): trace = trace
	elif trace is not None: raise ValueError("trace option should be a path to a csv datafile or a pandas Dataframe")
	else: trace = pd.read_csv( data_file_path + "rrl_metallicity_fit_trace/%s" % metallicity_RRc_default_traces[default_trace].lower())

	Fe_lin = sample_metallicity(period=period, phi31=phi31, trace=trace, period_err=period_err, phi31_err=phi31_err,Nsample=Nsample)

	return Fe_lin

def metallicity_RRc(period, phi31, period_err=None, phi31_err=None, Nsample=1000, trace=None, default_trace="gc", flatten=False):
	"""
	Sample the metallicity from some linear relation of kind Fe/H=Fe/H_hat + slope_fe*Period + slope_phi31*phi31.
	:param period: RRL c first overtone  period (scalar or 1D array, same length of phi31).
	:param phi31: RRL c phi31 as estimated in Gaia (scalar of 1D array, same length of period).
	:param period_err: Error on period (same length of period, it could be None).
	:param phi31_err: Error on phi31 (same length of phi31, it could be None).
	:param Nsample: Number of samplings per data.
	:param trace: Linear relation trace to use. It could be a path to a csv file, a pandas dataframe or None.
				  The keyword to be used are Fe_hat, slope_p_fe, slope_phi31_fe, Fe_scatter. If it is None
				  the default trace will be used.
	:param default_trace: name of the default trace to be used (only gc is available).
	:param flatten: If True, the first item to return is flatten (length length(period)*Nsample) othwerwise it is a (length(period), Nsample) 2D array
	:return: The first item is the dictionary with the statistics (mean, std, median, q16, q84 (~ 1sigma), q2, q97 (~ 2sigma), q02, q99 (~3 sigma).
			The second item is an array (1D or 2D, depending on the flatten parameter) containing all the sampled values.
	"""

	Fe_lin  	= sample_metallicity_RRc(period=period, phi31=phi31, period_err=period_err, phi31_err=phi31_err, Nsample=Nsample, trace=trace, default_trace=default_trace)

	ret_stat = {}
	_Fe_tmp  = Fe_lin.reshape(-1, Nsample)
	ret_stat["mean"] = np.mean(_Fe_tmp, axis=1)
	ret_stat["std"] = np.std(_Fe_tmp, axis=1)
	ret_stat["median"] = np.percentile(_Fe_tmp, q=50, axis=1)
	ret_stat["q16"] = np.percentile(_Fe_tmp, q=16, axis=1)
	ret_stat["q84"] = np.percentile(_Fe_tmp, q=84, axis=1)
	ret_stat["q97"] = np.percentile(_Fe_tmp, q=97.7, axis=1)
	ret_stat["q2"] = np.percentile(_Fe_tmp, q=2, axis=1)
	ret_stat["q99"] = np.percentile(_Fe_tmp, q=99.8, axis=1)
	ret_stat["q02"] = np.percentile(_Fe_tmp, q=0.2, axis=1)

	if flatten==False: Fe_lin=_Fe_tmp


	return ret_stat, Fe_lin


def sample_metallicity_SOS(metallicity, metallicity_err,type,N):

	if metallicity is None:
		if type.lower() == "rrab": path_to_file = data_file_path + "gaia_sos_distribution/RRab_SOS_metallicity.xdgmm"
		elif type.lower() == "rrc": path_to_file = data_file_path + "gaia_sos_distribution/RRc_SOS_metallicity.xdgmm"
		else: raise ValueError("type %s not allowed"%(type))

		xdgmm = XDGMM(filename=path_to_file)
		met = xdgmm.sample(N).flatten()
	elif metallicity_err is None:
		met=np.random.normal(metallicity,0.5,N)
	else:
		met = np.random.normal(metallicity, metallicity_err, N)

	return met


##Mg
#Mg-Fe relation from Muraveva+19  -> Mg = M_hat + _slope_fe*[Fe/H] +- _Mscatter
_Mhat           = 1.11
_Mhat_err       = 0.06
_slope_fe       = 0.33
_slope_fe_err   = 0.04
_Mscatter       = 0.17
_Mscatter_err   = 0.02
##
def sample_Mg_single(fe_list):
	"""
	Sample the Mg from a samplet of metallicity using the Muraveva+19 relation.
	The relation in Muraveva+19 is sampled (same length of fe_list) using the errors reported in the paper.
	:param fe_list: sample of Metallicity.
	:return: numpy array containing Sample of Mg  (same length of fe_list)
	"""
	fe_list=np.atleast_1d(fe_list)
	Nsample=len(fe_list)

	Mhat = np.random.normal(_Mhat, _Mhat_err, Nsample)
	slope_fe = np.random.normal(_slope_fe, _slope_fe_err, Nsample)
	Mscatter = np.random.normal(_Mscatter, _Mscatter_err, Nsample)

	lin_rel = Mhat + slope_fe * fe_list
	M = np.random.normal(lin_rel, Mscatter)


	return M

def sample_Mg(fe, fe_err=None, Nsample=1000):
	"""
	Sample the Mg from the metallicity using  the Muraveva+19 relation.
	:param fe: Metallicity (can be a scalar or a 1D numpy array).
	:param fe_err: Metallicity error (can be a scalar, a 1D numpy array or None)
	:param Nsample: Number of point to sample.
	:return: numpy array containing Sample of Mg (len(fe)*Nsample)
	"""

	fe = np.repeat(fe, Nsample)
	if fe_err is not None:
		fe_err = np.repeat(fe_err, Nsample)
		fe_err = np.where(np.isnan(fe_err), 0, fe_err)
		fe = np.random.normal(fe, fe_err)


	M=sample_Mg_single(fe)


	return M

def estimate_Mg(fe, fe_err=None, Nsample=1000, flatten=False):
	"""
	Sample the Mg from the metallicity using  the Muraveva+19 relation and return statistics
	:param fe: Metallicity (can be a scalar or a 1D numpy array).
	:param fe_err: Metallicity error (can be a scalar, a 1D numpy array or None)
	:param Nsample: Number of point to sample.
	:param flatten: If True, the first item to return is flatten (length length(period)*Nsample) othwerwise it is a (length(period), Nsample) 2D array
	:return: The first item is the dictionary with the statistics (mean, std, median, q16, q84 (~ 1sigma), q2, q97 (~ 2sigma), q02, q99 (~3 sigma).
			The second item is an array (1D or 2D, depending on the flatten parameter) containing all the sampled values.
	"""

	M = sample_Mg(fe=fe, fe_err=fe_err, Nsample=Nsample)
	print('M', M.shape)

	ret_stat = {}
	_M_tmp  = M.reshape(-1, Nsample)
	ret_stat["mean"] = np.mean(_M_tmp, axis=1)
	ret_stat["std"] = np.std(_M_tmp, axis=1)
	ret_stat["median"] = np.percentile(_M_tmp, q=50, axis=1)
	ret_stat["q16"] = np.percentile(_M_tmp, q=16, axis=1)
	ret_stat["q84"] = np.percentile(_M_tmp, q=84, axis=1)
	ret_stat["q97"] = np.percentile(_M_tmp, q=97.7, axis=1)
	ret_stat["q2"] = np.percentile(_M_tmp, q=2, axis=1)
	ret_stat["q99"] = np.percentile(_M_tmp, q=99.8, axis=1)
	ret_stat["q02"] = np.percentile(_M_tmp, q=0.2, axis=1)

	if flatten==False: M=_M_tmp


	return ret_stat, M


def sample_g_gaia_to_g_sos(g_gaia, Nsample=1000):
	"""

	:param g_gaia:
	:param Nsample:
	:return:
	"""

	g_gaia = np.repeat(g_gaia, Nsample)
	path_to_file = data_file_path + "ggaia_to_sos_correction/gsos_ggaia_diff.xdgmm"
	xdgmm = XDGMM(filename=path_to_file)
	delta_sample = xdgmm.sample(len(g_gaia))

	g_sos = g_gaia + delta_sample.flatten()

	return g_sos

def sample_Dsun_fromMg(g,ebv,Mg,Mg_error=None, bp_rp=None, g_error=None, ebv_error=None,bp_rp_error=None, kg=2.27, kg_error=0.3,sos_correction=False, return_all=False, Nsample=1000):

	if sos_correction:
		g_sample = sample_g_gaia_to_g_sos(g, Nsample=Nsample)
		if bp_rp is not None:
			Ag_sample = _ext_class_for_gc.Ag_iterative_error_sample(bp_rp, ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nsample, Nmax=1000)
		else:
			if ebv_error is None: ebv_error=0
			Ag_sample = np.random.normal(kg, kg_error, Nsample) * np.random.normal(ebv, ebv_error, Nsample)
		_gc_sample = g_sample - Ag_sample
	else:
		_gc_sample = gc_sample(g, ebv, bp_rp=bp_rp, kg=kg, g_error=g_error, bp_rp_error=bp_rp_error, kg_error=kg_error,ebv_error=ebv_error, Nsample=Nsample)

	if (Mg_error is not None):
		Mg_sample=np.random.normal(Mg,Mg_error,Nsample)
	else:
		Mg_sample=np.ones(Nsample)*Mg

	dist_sample=m_to_dist(_gc_sample, Mg_sample)

	if return_all:
		return dist_sample, _gc_sample, Mg_sample
	else:
		return dist_sample

def sample_Dsun_single_SOS(g, ebv, metallicity=None, bp_rp=None, g_error=None, ebv_error=None, metallicity_error=None, bp_rp_error=None, kg=2.27, kg_error=0.3,  type="RRab", sos_correction=False, return_all=False, Nsample=1000):

	#1 ad 2 (depend on type)
	met_sample=sample_metallicity_SOS(metallicity, metallicity_error, type, Nsample)

	#3-Mg
	Mg_sample=sample_Mg_single(met_sample)
	#print(np.mean(Mg_sample), np.std(Mg_sample))

	#5-Correct g for SOS (optional)
	if sos_correction:
		g_sample = sample_g_gaia_to_g_sos(g, Nsample=Nsample)
		if bp_rp is not None:
			Ag_sample = _ext_class_for_gc.Ag_iterative_error_sample(bp_rp, ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nsample, Nmax=1000)
		else:
			if ebv_error is None: ebv_error=0
			Ag_sample = np.random.normal(kg, kg_error, Nsample) * np.random.normal(ebv, ebv_error, Nsample)
		_gc_sample = g_sample - Ag_sample
	else:
		_gc_sample = gc_sample(g, ebv, bp_rp=bp_rp, kg=kg, g_error=g_error, bp_rp_error=bp_rp_error, kg_error=kg_error,ebv_error=ebv_error, Nsample=Nsample)

	#5b-Gc
	#_gc_sample = gc_sample(g, ebv, bp_rp=bp_rp, kg=kg, g_error=g_error, bp_rp_error=bp_rp_error, kg_error=kg_error, ebv_error=ebv_error,Nsample=Nsample)
	#print(np.mean(_gc_sample), np.std(_gc_sample))


	#Distance
	dist_sample=m_to_dist(_gc_sample, Mg_sample)
	#print(np.mean(dist_sample), np.std(dist_sample))

	if return_all:
		return dist_sample, met_sample, _gc_sample, Mg_sample
	else:
		return dist_sample


def sample_Dsun_single(g, ebv, period=None, phi31=None, bp_rp=None,  g_error=None, ebv_error=None, period_error=0, phi31_error=0, bp_rp_error=None, kg=2.27, kg_error=0.3,  type="RRab", default_trace="layden", sos_correction=False, return_all=False, Nsample=1000):
	"""
	Sample the distance from the Sun of a RRL given the observed g magnitude, the extinction, color, period and phi31 (optional).
	NB it accepts only scalar values.
	:param g:
	:param ebv:
	:param period:
	:param phi31:
	:param bp_rp:
	:param g_error:
	:param ebv_error:
	:param period_error:
	:param phi31_error:
	:param bp_rp_error:
	:param kg:
	:param kg_error:
	:param type:
	:param Nsample:
	:return:
	"""


	#1 ad 2 (depend on type)
	if type.lower() == "rrab":
		#1-Sample period and phi31
		period, phi31 = sample_p_phi31_RRab(Nsample, p=period, p_err=period_error, phi31=phi31, phi31_err=phi31_error)
		#2-Metallicity
		met_sample = sample_metallicity_RRab(period, phi31, period_err=0, phi31_err=0,Nsample=1, default_trace=default_trace) #Period and Phi31 are already sample
		#print(met_sample.shape)
		#print(np.mean(met_sample), np.std(met_sample))
	elif type.lower() == "rrc":
		#1-Sample period and phi31
		period, phi31 = sample_p_phi31_RRc(Nsample, p=period, p_err=period_error, phi31=phi31, phi31_err=phi31_error)
		#2-Metallicity
		met_sample = sample_metallicity_RRc(period, phi31, period_err=0, phi31_err=0,Nsample=1, default_trace="gc") #Period and Phi31 are already sample
		#print(met_sample.shape)
		#print(np.mean(met_sample), np.std(met_sample))

	#3-Mg
	Mg_sample=sample_Mg_single(met_sample)
	#print(np.mean(Mg_sample), np.std(Mg_sample))

	#5-Correct g for SOS (optional)
	if sos_correction:
		g_sample = sample_g_gaia_to_g_sos(g, Nsample=Nsample)
		if bp_rp is not None:
			Ag_sample = _ext_class_for_gc.Ag_iterative_error_sample(bp_rp, ebv, bp_rp_error=bp_rp_error, ebv_error=ebv_error, Nerror=Nsample, Nmax=1000)
		else:
			if ebv_error is None: ebv_error=0
			Ag_sample = np.random.normal(kg, kg_error, Nsample) * np.random.normal(ebv, ebv_error, Nsample)
		_gc_sample = g_sample - Ag_sample
	else:
		_gc_sample = gc_sample(g, ebv, bp_rp=bp_rp, kg=kg, g_error=g_error, bp_rp_error=bp_rp_error, kg_error=kg_error,ebv_error=ebv_error, Nsample=Nsample)

	#5b-Gc
	#_gc_sample = gc_sample(g, ebv, bp_rp=bp_rp, kg=kg, g_error=g_error, bp_rp_error=bp_rp_error, kg_error=kg_error, ebv_error=ebv_error,Nsample=Nsample)
	#print(np.mean(_gc_sample), np.std(_gc_sample))


	#Distance
	dist_sample=m_to_dist(_gc_sample, Mg_sample)
	#print(np.mean(dist_sample), np.std(dist_sample))

	if return_all:
		return dist_sample, met_sample, _gc_sample, Mg_sample
	else:
		return dist_sample

_str_plist="(id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc, distance, distance_error, internal_id)"
_str_kpc="kpc"
_str_kms="km/s"
def sample_obs_error_5D_rrl(property_list:_str_plist, Rsun:_str_kpc=8.2, Rsun_err:_str_kpc=None, U:_str_kms=11.1, V:_str_kms=12.24, W:_str_kms=7.25, U_err:_str_kms=None, V_err:_str_kms=None, W_err:_str_kms=None, Vlsr:_str_kms=235, Vlsr_err:_str_kms=None, Mgc:"mag"=None, Mgc_err:"mag"=None, N:"int"=1000, sos_correction:"sos_correction"=True, q=1.0, qinf=1.0, rq=10.0, p=1.0, alpha=0, beta=0, gamma=0, ax='zyx')->"array and dic with properties":
	"""
	NB: THE INPUT ARE ASSUME A GALACTIC RH system (Sun is a x=Rsun),
		BUT THE OUTPUT ARE IN GALACTIC LH system (I know is crazy).
	:param property_list: A tuple with the following properties (in this order):
		"(id: source_id of the star (can be None),
		 ra: degrees,
		 dec: degrees,
		 l: degrees,
		 b: degrees,
		 pmra: pmra proper motion (non corrected for solar motion) mas/yr,
		 pmdec: pmdec proper motion (non corrected for solar motion) mas/yr,
		 pmra_err: pmra error,
		 pmdec_err: pmdec error,
		 cov_pmra_pmdec: pmra-pmdec corr. coff (sigma_ra_dec/(sigma_ra*sigma_dec)),
		 g: G magnitude (can be None if distance is provided), not corrected for reddening
		 g_sos: G magnitude as estimated in SOS (can be None), not corrected for reddening
		 bp_rp: Gaia color,
		 ebv: ebv,
		 period: fundamental period (can be None).
		 period_1o: first overtone can be None,
		 period_error:
		 phi31: LC Phase (can be None),
		 phi31_error:
		 type: "rrab" or "rrc",
		 distance: Heliocentric distance in kpc (can be None if gc is provided)
		 distance_error:
		 internal_id: a user defined internal_id (can be None)".

	:param Rsun: Distance of the Sun from the Galactic centre.
	:param Rsun_err: error on Rsun.
	:param U: Solar motion (wrt LSR) toward the Galactic center
	(NB: here it is defined positive if it is toward the Galctice centre, but sample_obs_erro we used a left-hand system,
	in this system a motion toward the GC is negatie. However this converstion is automatically made).
	:param V: Solar proper motion (wrt LSR) along the direction of Galactic rotation.
	:param W: Solar proper motion (wrt LSR) along the normal to the Galactic plane (positive value is an upaward motion).
	:param U_err: error on U.
	:param V_err: error on V.
	:param W_err: error on W.
	:param Vlsr:  Circular motion of the LSR.
	:param Vlsr_err:  Error on Vlsr.
	:param Mgc: Absolute magnitude to estimate distance from gc.
	:param Mgc_err: error on Absolute magnitude.
	:param N: Number of MC samples to generate.
	:return: An array and a dictionary containing spatial and kinematic information obtained from the observables.
	"""

	_key_list_obs = ('x', 'y', 'z', 'x_err', 'y_err', 'z_err', 'p_x_y', 'p_x_z', 'p_y_z',
					 'Rcyl', 'phi', 'Rcyl_err', 'phi_err', 'p_Rcyl_phi', 'p_Rcyl_z', 'p_phi_z',
					 'r', 'theta', 'r_err', 'theta_err', 'p_r_theta', 'p_r_phi', 'p_theta_phi',
					 'pmra', 'pmdec', 'pmra_err', 'pmdec_err', 'p_pmra_pmdec',
					 'pmra_c', 'pmdec_c', 'pmra_c_err', 'pmdec_c_err', 'p_pmra_c_pmdec',
					 'pml', 'pmb', 'pml_err', 'pmb_err', 'p_pml_pmb',
					 'pml_c', 'pmb_c', 'pml_c_err', 'pmb_c_err', 'p_pml_c_pmb_c',
					 'Vl', 'Vb', 'Vl_err', 'Vb_err', 'p_Vl_Vb',
					 'Vl_c', 'Vb_c', 'Vl_c_err', 'Vb_c_err', 'p_Vl_c_Vb_c',
					 'dsun', 'Vtan_c', 'dsun_err', 'Vtan_c_err', 'p_dsun_Vtan_c',
					 'l', 'b', 'ra', 'dec', 'gc', 'gc_err', 'Mg', 'Mg_err', 'feh', 'feh_err',
					 'p_dsun_feh', 'p_Vl_c_feh', 'p_Vb_c_feh', 're', 're_err',  'type_rrl', 'source_id', 'id')



	_K = COST._K
	cts = tr.cartesian_to_spherical
	stc = tr.spherical_to_cartesian

	#TO BE SURE THAT EVERYTHING IS OK

	try:
		id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, g, g_sos, bp_rp,ebv, period, period_1o, period_error,  phi31, phi31_error, type, distance, distance_error, internal_id = property_list
		onesl   = np.ones(N) #list of ones
		ral     = onesl*ra
		decl    = onesl*dec
		ll      = onesl*l
		bl      = onesl*b
		ll = np.radians(ll)
		bl = np.radians(bl)
		type_input = type

		#1-Distance stuff
		#Check  if we have to use  distance (priority), Mg,  or gc and Mg
		if distance is not None and distance_error is not None:
			Dsunl  = np.random.normal(distance, distance_error,N)
			FeHl = gcl = Mgl = np.repeat(-999, N)
		elif distance is not None:
			Dsunl  = np.repeat(distance, N)
			FeHl = gcl = Mgl = np.repeat(-999, N)
		else:

			if type.lower()=="rrab": period_input, type_input = period, type
			elif type.lower()=="rrc": period_input, type_input = period_1o, type
			else: raise ValueError("Can be rrab or rrc in file %s, it is instead %s"%(__file__,type))

			if sos_correction:
				if g_sos is None: sos_correction_input, g_input = True, g
				else: sos_correction_input, g_input = False, g_sos
			else:
				sos_correction_input, g_input = False, g

			if Mgc is not None:
				Dsunl,gcl, Mgl = sample_Dsun_fromMg(g_input,ebv,Mgc,Mg_error=Mgc_err, bp_rp=bp_rp, g_error=None, ebv_error=0.16*ebv,bp_rp_error=None,
													kg=2.27, kg_error=0.3,sos_correction=sos_correction_input, return_all=True, Nsample=N)
				FeHl = np.repeat(-999, N)
			else:

				Dsunl, FeHl, gcl, Mgl = sample_Dsun_single(g_input, ebv, period=period_input, phi31=phi31, bp_rp=bp_rp, g_error=None, ebv_error=0.16*ebv,
									   period_error=period_error, phi31_error=phi31_error, bp_rp_error=None, kg=2.27, kg_error=0.3, type=type_input, default_trace="layden",
									   sos_correction=sos_correction_input, return_all=True, Nsample=N)

		#1b-
		#Rsun
		if Rsun_err is None: Rsunl = onesl*Rsun
		else: Rsunl = np.random.normal(Rsun, Rsun_err, N)

		# LHS COORD
		xsl    =  Dsunl * np.cos(ll) * np.cos(bl)
		yl     =  Dsunl * np.sin(ll) * np.cos(bl)
		zl     =  Dsunl * np.sin(bl)
		xl     =  Rsunl - xsl
		Rl     =  np.sqrt(xl * xl + yl * yl)
		phil   =  np.arctan2(yl, xl)
		rl     =  np.sqrt( Rl*Rl + zl*zl )
		thetal =  np.arcsin(zl/rl)
		rel	   =  tr.xyz_to_m(xl, yl, zl, q=q, qinf=qinf, rq=rq, p=p, alpha=alpha, beta=beta, gamma=gamma, ax=ax)

		#STOP HERE IF PMRA OR PMDEC ARE NONE
		if pmra is None or np.isnan(pmra) or pmdec is None or np.isnan(pmdec):

			# Estiamte Std, Cov
			_nan3= np.array([np.nan,]*3)
			_nan2= np.array([np.nan,]*2)
			Mean_cart, Std_cart, rho_cart = calc_covariance( xl, yl, zl)
			Mean_cyl, Std_cyl, rho_cyl = calc_covariance(Rl, np.degrees(phil), zl)
			Mean_sph, Std_sph, rho_sph = calc_covariance(rl, np.degrees(thetal), np.degrees(phil))
			Mean_sky, Std_sky, rho_sky = _nan2, _nan2, _nan2
			Mean_sky_corr, Std_sky_corr, rho_sky_corr = _nan2,_nan2,_nan2
			Mean_skyp, Std_skyp, rho_skyp = _nan2,_nan2,_nan2
			Mean_skyp_corr, Std_skyp_corr, rho_skyp_corr = _nan2,_nan2,_nan2
			Mean_sky_tan, Std_sky_tan, rho_Sky_tan = _nan2,_nan2,_nan2
			_, _, rho_Dsun_Vlc = _nan2,_nan2,_nan2
			_, _, rho_Dsun_Vbc = _nan2,_nan2,_nan2
			Mean_Dsun_feh, Std_Dsun_feh, rho_Dsun_feh = calc_covariance(Dsunl, FeHl)
			_, _, rho_vl_feh = _nan3,_nan3,_nan3

			# pmra pmdec corr
			Mean_skyeq_corr, Std_skyeq_corr, rho_skyeq_corr = _nan2,_nan2,_nan2

		else:


			#2a-Estiamte total error for proper motion
			if g is not None:
				pmra_err = gaia.total_error_pm(pmra_err, g)
				pmdec_err = gaia.total_error_pm(pmdec_err, g)


			#2b-Sample proper motion
			cov_pm               =  pmra_err*pmdec_err*cov_pmra_pmdec
			cov_matrix           =  [ [pmra_err**2, cov_pm],  [cov_pm, pmdec_err**2] ]
			pmral, pmdecl        =  np.random.multivariate_normal( [pmra, pmdec], cov_matrix, N).T

			#print("pmral",pmral[0], pmral.shape)

			#3-Correct proper motion  for internal rotation for bright star
			if g is not None and g<12:
				pmral, pmdecl = gaia.sample_pm_frame_correction(pmral, pmdecl, ral, decl, Nsample=1)

			#print("pmral",pmral[0], pmral.shape)

			#4-Sample proper motion l and b
			pmll, pmbl		     =  co.pmrapmdec_to_pmllpmbb(pmral, pmdecl, ral, decl, degree=True).T

			#5 Sample Galacitc properties
			#Vsun
			Vsunl = tr._make_Vsunl(U, V, W, U_err, V_err, W_err, Vlsr, Vlsr_err, N)


			#pml pmb corr
			Vl_nocorr =  _K*Dsunl*pmll
			Vb_nocorr =  _K*Dsunl*pmbl

			vxsl_corr, vysl_corr, vzsl_corr  = stc(np.zeros_like(Vl_nocorr), Vl_nocorr, Vb_nocorr, ll, bl, true_theta=False, degree=False)
			vxl_corr, vyl_corr, vzl_corr     = -(vxsl_corr + Vsunl[:,0]), vysl_corr + Vsunl[:, 1], vzsl_corr + Vsunl[:, 2]
			_, Vbl, Vll                      = cts(-vxl_corr, vyl_corr, vzl_corr, ll, bl, true_theta=False, degree=False)
			pmbl_corr, pmll_corr             = Vbl/(_K*Dsunl), Vll/(_K*Dsunl)
			#pmral_corr, pmdecl_corr			 = co.pmllpmbb_to_pmrapmdec(pmll=pmll_corr, pmbb=pmbl_corr, l=ll, b=bl, degree=True).T
			#A		 = co.pmllpmbb_to_pmrapmdec(pmll=pmll_corr, pmbb=pmbl_corr, l=ll, b=bl, degree=True).T
			Vtanl                            = np.sqrt(Vll*Vll + Vbl*Vbl)

			#Estiamte Std, Cov
			Mean_cart, Std_cart, rho_cart = calc_covariance( xl, yl, zl)
			Mean_cyl, Std_cyl, rho_cyl    = calc_covariance( Rl, np.degrees(phil), zl)
			Mean_sph, Std_sph, rho_sph    = calc_covariance( rl, np.degrees(thetal), np.degrees(phil))
			Mean_sky, Std_sky, rho_sky    = calc_covariance(pmll, pmbl)
			Mean_sky_corr, Std_sky_corr, rho_sky_corr = calc_covariance(pmll_corr, pmbl_corr)
			Mean_skyp, Std_skyp, rho_skyp = calc_covariance(Vl_nocorr, Vb_nocorr)
			Mean_skyp_corr, Std_skyp_corr, rho_skyp_corr = calc_covariance(Vll, Vbl)
			Mean_sky_tan, Std_sky_tan, rho_Sky_tan = calc_covariance(Dsunl, Vtanl)
			_,_, rho_Dsun_Vlc = calc_covariance(Dsunl, Vll)
			_,_, rho_Dsun_Vbc = calc_covariance(Dsunl, Vbl)
			Mean_Dsun_feh, Std_Dsun_feh, rho_Dsun_feh = calc_covariance(Dsunl, FeHl)
			_, _, rho_vl_feh = calc_covariance(FeHl, Vll, Vbl)

			#pmra pmdec corr
			pmral_corr, pmdecl_corr = co.pmllpmbb_to_pmrapmdec(pmll_corr,pmbl_corr, ll, bl, degree=False, epoch=2000.0).T
			Mean_skyeq_corr, Std_skyeq_corr, rho_skyeq_corr = calc_covariance(pmral_corr, pmdecl_corr)

		#out_array = np.zeros(65)
		out_array = np.zeros(76)
		#Cartesian
		out_array[0:3] = Mean_cart[:3] #x,y,z
		out_array[3:6] = Std_cart[:3] #err on x,y,z
		out_array[6:9] = rho_cart[:3] #cov on x,y,z
		#Cylindrical
		out_array[9:11] = Mean_cyl[:2] #R, phi
		out_array[11:13] = Std_cyl[:2] #err on Rphi
		out_array[13:16] = rho_cyl[:3]
		#Spherical
		out_array[16:18] = Mean_sph[:2] #r, theta
		out_array[18:20] = Std_sph[:2] #err on r, theta
		out_array[20:23] = rho_sph[:3]

		#PMRA
		out_array[23] = pmra
		out_array[24] = pmdec
		out_array[25] = pmra_err
		out_array[26] = pmdec_err
		out_array[27] = cov_pmra_pmdec
		out_array[28:30] = Mean_skyeq_corr #r, theta
		out_array[30:32] = Std_skyeq_corr #err on r, theta
		out_array[32] = rho_skyeq_corr[0]

		#PML
		out_array[33:35] = Mean_sky #r, theta
		out_array[35:37] = Std_sky #err on r, theta
		out_array[37] = rho_sky[0]
		out_array[38:40] = Mean_sky_corr #r, theta
		out_array[40:42] = Std_sky_corr #err on r, theta
		out_array[42] = rho_sky_corr[0]

		#V
		out_array[43:45] = Mean_skyp #r, theta
		out_array[45:47] = Std_skyp #err on r, theta
		out_array[47] = rho_skyp[0]
		out_array[48:50] = Mean_skyp_corr #r, theta
		out_array[50:52] = Std_skyp_corr #err on r, theta
		out_array[52] = rho_skyp_corr[0]
		out_array[53:55] = Mean_sky_tan #r, theta
		out_array[55:57] = Std_sky_tan#err on r, theta
		out_array[57] = rho_Sky_tan[0]

		#AUX
		out_array[58] 	 = l
		out_array[59] 	 = b
		out_array[60] 	 = ra
		out_array[61] 	 = dec
		out_array[62] 	 = np.nanmean(gcl)
		out_array[63] 	 = np.nanstd(gcl)
		out_array[64] 	 = np.nanmean(Mgl)
		out_array[65] 	 = np.nanstd(Mgl)
		out_array[66]	 = Mean_Dsun_feh[1]
		out_array[67]	 = Std_Dsun_feh[1]
		out_array[68]	 = rho_Dsun_feh[0]
		out_array[69:71] = rho_Dsun_feh[:2]
		out_array[71]    = np.mean(rel)
		out_array[72]    = np.std(rel)

		#CHECK ID
		if id is None and internal_id is None: id = internal_id = ut.create_long_index()
		elif internal_id is None: internal_id = id
		elif id is None: id = internal_id

		if type_input is None:  out_array[73] = np.nan
		elif type_input=="rrab": out_array[73]    = 0
		elif type_input=="rrc": out_array[73]    = 1

		out_array[74] 	 = int(id)
		out_array[75] 	 = internal_id

		out_array=np.where(np.isfinite(out_array),out_array,np.nan)

	except:
		out_array = np.zeros(76)
		out_array[:] = np.nan


	return out_array, dict(zip(_key_list_obs, out_array))

def sample_obs_error_5D_rrl_SOS(property_list:_str_plist, Rsun:_str_kpc=8.2, Rsun_err:_str_kpc=None, U:_str_kms=11.1, V:_str_kms=12.24, W:_str_kms=7.25, U_err:_str_kms=None, V_err:_str_kms=None, W_err:_str_kms=None, Vlsr:_str_kms=235, Vlsr_err:_str_kms=None, Mgc:"mag"=None, Mgc_err:"mag"=None, N:"int"=1000, sos_correction:"sos_correction"=True, q=1.0, qinf=1.0, rq=10.0, p=1.0, alpha=0, beta=0, gamma=0, ax='zyx')->"array and dic with properties":
	"""
	NB: THE INPUT ARE ASSUME A GALACTIC RH system (Sun is a x=Rsun),
		BUT THE OUTPUT ARE IN GALACTIC LH system (I know is crazy).
	:param property_list: A tuple with the following properties (in this order):
		"(id: source_id of the star (can be None),
		 ra: degrees,
		 dec: degrees,
		 l: degrees,
		 b: degrees,
		 pmra: pmra proper motion (non corrected for solar motion) mas/yr,
		 pmdec: pmdec proper motion (non corrected for solar motion) mas/yr,
		 pmra_err: pmra error,
		 pmdec_err: pmdec error,
		 cov_pmra_pmdec: pmra-pmdec corr. coff (sigma_ra_dec/(sigma_ra*sigma_dec)),
		 g: G magnitude (can be None if distance is provided), not corrected for reddening
		 g_sos: G magnitude as estimated in SOS (can be None), not corrected for reddening
		 bp_rp: Gaia color,
		 ebv: ebv,
		 metallicity: SOS metallicity,
		 metallicity_error: SOS metallicity error,
		 type: "rrab" or "rrc",
		 distance: Heliocentric distance in kpc (can be None if gc is provided)
		 distance_error:
		 internal_id: a user defined internal_id (can be None)".

	:param Rsun: Distance of the Sun from the Galactic centre.
	:param Rsun_err: error on Rsun.
	:param U: Solar motion (wrt LSR) toward the Galactic center
	(NB: here it is defined positive if it is toward the Galctice centre, but sample_obs_erro we used a left-hand system,
	in this system a motion toward the GC is negatie. However this converstion is automatically made).
	:param V: Solar proper motion (wrt LSR) along the direction of Galactic rotation.
	:param W: Solar proper motion (wrt LSR) along the normal to the Galactic plane (positive value is an upaward motion).
	:param U_err: error on U.
	:param V_err: error on V.
	:param W_err: error on W.
	:param Vlsr:  Circular motion of the LSR.
	:param Vlsr_err:  Error on Vlsr.
	:param Mgc: Absolute magnitude to estimate distance from gc.
	:param Mgc_err: error on Absolute magnitude.
	:param N: Number of MC samples to generate.
	:return: An array and a dictionary containing spatial and kinematic information obtained from the observables.
	"""

	_key_list_obs = ('x', 'y', 'z', 'x_err', 'y_err', 'z_err', 'p_x_y', 'p_x_z', 'p_y_z',
					 'Rcyl', 'phi', 'Rcyl_err', 'phi_err', 'p_Rcyl_phi', 'p_Rcyl_z', 'p_phi_z',
					 'r', 'theta', 'r_err', 'theta_err', 'p_r_theta', 'p_r_phi', 'p_theta_phi',
					 'pmra', 'pmdec', 'pmra_err', 'pmdec_err', 'p_pmra_pmdec',
					 'pmra_c', 'pmdec_c', 'pmra_c_err', 'pmdec_c_err', 'p_pmra_c_pmdec',
					 'pml', 'pmb', 'pml_err', 'pmb_err', 'p_pml_pmb',
					 'pml_c', 'pmb_c', 'pml_c_err', 'pmb_c_err', 'p_pml_c_pmb_c',
					 'Vl', 'Vb', 'Vl_err', 'Vb_err', 'p_Vl_Vb',
					 'Vl_c', 'Vb_c', 'Vl_c_err', 'Vb_c_err', 'p_Vl_c_Vb_c',
					 'dsun', 'Vtan_c', 'dsun_err', 'Vtan_c_err', 'p_dsun_Vtan_c',
					 'l', 'b', 'ra', 'dec', 'gc', 'gc_err', 'Mg', 'Mg_err', 'feh', 'feh_err',
					 'p_dsun_feh', 'p_Vl_c_feh', 'p_Vb_c_feh', 're', 're_err',  'type_rrl', 'source_id', 'id')



	_K = COST._K
	cts = tr.cartesian_to_spherical
	stc = tr.spherical_to_cartesian

	#TO BE SURE THAT EVERYTHING IS OK

	try:
		id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, g, g_sos, bp_rp, ebv, metallicity, metallicity_err,  type, distance, distance_error, internal_id = property_list
		onesl   = np.ones(N) #list of ones
		ral     = onesl*ra
		decl    = onesl*dec
		ll      = onesl*l
		bl      = onesl*b
		ll = np.radians(ll)
		bl = np.radians(bl)
		type_input = type

		print("WE")

		#1-Distance stuff
		#Check  if we have to use  distance (priority), Mg,  or gc and Mg
		if distance is not None and distance_error is not None:
			Dsunl  = np.random.normal(distance, distance_error,N)
			FeHl = gcl = Mgl = np.repeat(-999, N)
		elif distance is not None:
			Dsunl  = np.repeat(distance, N)
			FeHl = gcl = Mgl = np.repeat(-999, N)
		else:



			if sos_correction:
				if g_sos is None: sos_correction_input, g_input = True, g
				else: sos_correction_input, g_input = False, g_sos
			else:
				sos_correction_input, g_input = False, g

			if Mgc is not None:
				Dsunl,gcl, Mgl = sample_Dsun_fromMg(g_input,ebv,Mgc,Mg_error=Mgc_err, bp_rp=bp_rp, g_error=None, ebv_error=0.16*ebv,bp_rp_error=None,
													kg=2.27, kg_error=0.3,sos_correction=sos_correction_input, return_all=True, Nsample=N)
				FeHl = np.repeat(-999, N)
			else:

				Dsunl, FeHl, gcl, Mgl = sample_Dsun_single_SOS(g=g_input, ebv=ebv, metallicity=metallicity, bp_rp=bp_rp, g_error=None, ebv_error=0.16*ebv, metallicity_error=metallicity_err,
									   bp_rp_error=None, kg=2.27, kg_error=0.3,  type=type, sos_correction=sos_correction_input, return_all=True, Nsample=N)

		#1b-
		#Rsun
		if Rsun_err is None: Rsunl = onesl*Rsun
		else: Rsunl = np.random.normal(Rsun, Rsun_err, N)

		# LHS COORD
		xsl    =  Dsunl * np.cos(ll) * np.cos(bl)
		yl     =  Dsunl * np.sin(ll) * np.cos(bl)
		zl     =  Dsunl * np.sin(bl)
		xl     =  Rsunl - xsl
		Rl     =  np.sqrt(xl * xl + yl * yl)
		phil   =  np.arctan2(yl, xl)
		rl     =  np.sqrt( Rl*Rl + zl*zl )
		thetal =  np.arcsin(zl/rl)
		rel	   =  tr.xyz_to_m(xl, yl, zl, q=q, qinf=qinf, rq=rq, p=p, alpha=alpha, beta=beta, gamma=gamma, ax=ax)

		#STOP HERE IF PMRA OR PMDEC ARE NONE
		if pmra is None or np.isnan(pmra) or pmdec is None or np.isnan(pmdec):

			# Estiamte Std, Cov
			_nan3= np.array([np.nan,]*3)
			_nan2= np.array([np.nan,]*2)
			Mean_cart, Std_cart, rho_cart = calc_covariance( xl, yl, zl)
			Mean_cyl, Std_cyl, rho_cyl = calc_covariance(Rl, np.degrees(phil), zl)
			Mean_sph, Std_sph, rho_sph = calc_covariance(rl, np.degrees(thetal), np.degrees(phil))
			Mean_sky, Std_sky, rho_sky = _nan2, _nan2, _nan2
			Mean_sky_corr, Std_sky_corr, rho_sky_corr = _nan2,_nan2,_nan2
			Mean_skyp, Std_skyp, rho_skyp = _nan2,_nan2,_nan2
			Mean_skyp_corr, Std_skyp_corr, rho_skyp_corr = _nan2,_nan2,_nan2
			Mean_sky_tan, Std_sky_tan, rho_Sky_tan = _nan2,_nan2,_nan2
			_, _, rho_Dsun_Vlc = _nan2,_nan2,_nan2
			_, _, rho_Dsun_Vbc = _nan2,_nan2,_nan2
			Mean_Dsun_feh, Std_Dsun_feh, rho_Dsun_feh = calc_covariance(Dsunl, FeHl)
			_, _, rho_vl_feh = _nan3,_nan3,_nan3

			# pmra pmdec corr
			Mean_skyeq_corr, Std_skyeq_corr, rho_skyeq_corr = _nan2,_nan2,_nan2

		else:

			#2a-Estiamte total error for proper motion
			if g is not None:
				pmra_err = gaia.total_error_pm(pmra_err, g)
				pmdec_err = gaia.total_error_pm(pmdec_err, g)


			#2b-Sample proper motion
			cov_pm               =  pmra_err*pmdec_err*cov_pmra_pmdec
			cov_matrix           =  [ [pmra_err**2, cov_pm],  [cov_pm, pmdec_err**2] ]
			pmral, pmdecl        =  np.random.multivariate_normal( [pmra, pmdec], cov_matrix, N).T

			#print("pmral",pmral[0], pmral.shape)

			#3-Correct proper motion  for internal rotation for bright star
			if g<12:
				pmral, pmdecl = gaia.sample_pm_frame_correction(pmral, pmdecl, ral, decl, Nsample=1)

			#print("pmral",pmral[0], pmral.shape)

			#4-Sample proper motion l and b
			pmll, pmbl		     =  co.pmrapmdec_to_pmllpmbb(pmral, pmdecl, ral, decl, degree=True).T

			#5 Sample Galacitc properties
			#Vsun
			Vsunl = tr._make_Vsunl(U, V, W, U_err, V_err, W_err, Vlsr, Vlsr_err, N)


			#pml pmb corr
			Vl_nocorr =  _K*Dsunl*pmll
			Vb_nocorr =  _K*Dsunl*pmbl

			vxsl_corr, vysl_corr, vzsl_corr  = stc(np.zeros_like(Vl_nocorr), Vl_nocorr, Vb_nocorr, ll, bl, true_theta=False, degree=False)
			vxl_corr, vyl_corr, vzl_corr     = -(vxsl_corr + Vsunl[:,0]), vysl_corr + Vsunl[:, 1], vzsl_corr + Vsunl[:, 2]
			_, Vbl, Vll                      = cts(-vxl_corr, vyl_corr, vzl_corr, ll, bl, true_theta=False, degree=False)
			pmbl_corr, pmll_corr             = Vbl/(_K*Dsunl), Vll/(_K*Dsunl)
			#pmral_corr, pmdecl_corr			 = co.pmllpmbb_to_pmrapmdec(pmll=pmll_corr, pmbb=pmbl_corr, l=ll, b=bl, degree=True).T
			#A		 = co.pmllpmbb_to_pmrapmdec(pmll=pmll_corr, pmbb=pmbl_corr, l=ll, b=bl, degree=True).T
			Vtanl                            = np.sqrt(Vll*Vll + Vbl*Vbl)

			#Estiamte Std, Cov
			Mean_cart, Std_cart, rho_cart = calc_covariance( xl, yl, zl)
			Mean_cyl, Std_cyl, rho_cyl    = calc_covariance( Rl, np.degrees(phil), zl)
			Mean_sph, Std_sph, rho_sph    = calc_covariance( rl, np.degrees(thetal), np.degrees(phil))
			Mean_sky, Std_sky, rho_sky    = calc_covariance(pmll, pmbl)
			Mean_sky_corr, Std_sky_corr, rho_sky_corr = calc_covariance(pmll_corr, pmbl_corr)
			Mean_skyp, Std_skyp, rho_skyp = calc_covariance(Vl_nocorr, Vb_nocorr)
			Mean_skyp_corr, Std_skyp_corr, rho_skyp_corr = calc_covariance(Vll, Vbl)
			Mean_sky_tan, Std_sky_tan, rho_Sky_tan = calc_covariance(Dsunl, Vtanl)
			_,_, rho_Dsun_Vlc = calc_covariance(Dsunl, Vll)
			_,_, rho_Dsun_Vbc = calc_covariance(Dsunl, Vbl)
			Mean_Dsun_feh, Std_Dsun_feh, rho_Dsun_feh = calc_covariance(Dsunl, FeHl)
			_, _, rho_vl_feh = calc_covariance(FeHl, Vll, Vbl)

			#pmra pmdec corr
			pmral_corr, pmdecl_corr = co.pmllpmbb_to_pmrapmdec(pmll_corr,pmbl_corr, ll, bl, degree=False, epoch=2000.0).T
			Mean_skyeq_corr, Std_skyeq_corr, rho_skyeq_corr = calc_covariance(pmral_corr, pmdecl_corr)

		#out_array = np.zeros(65)
		out_array = np.zeros(76)
		#Cartesian
		out_array[0:3] = Mean_cart[:3] #x,y,z
		out_array[3:6] = Std_cart[:3] #err on x,y,z
		out_array[6:9] = rho_cart[:3] #cov on x,y,z
		#Cylindrical
		out_array[9:11] = Mean_cyl[:2] #R, phi
		out_array[11:13] = Std_cyl[:2] #err on Rphi
		out_array[13:16] = rho_cyl[:3]
		#Spherical
		out_array[16:18] = Mean_sph[:2] #r, theta
		out_array[18:20] = Std_sph[:2] #err on r, theta
		out_array[20:23] = rho_sph[:3]

		#PMRA
		out_array[23] = pmra
		out_array[24] = pmdec
		out_array[25] = pmra_err
		out_array[26] = pmdec_err
		out_array[27] = cov_pmra_pmdec
		out_array[28:30] = Mean_skyeq_corr #r, theta
		out_array[30:32] = Std_skyeq_corr #err on r, theta
		out_array[32] = rho_skyeq_corr[0]

		#PML
		out_array[33:35] = Mean_sky #r, theta
		out_array[35:37] = Std_sky #err on r, theta
		out_array[37] = rho_sky[0]
		out_array[38:40] = Mean_sky_corr #r, theta
		out_array[40:42] = Std_sky_corr #err on r, theta
		out_array[42] = rho_sky_corr[0]

		#V
		out_array[43:45] = Mean_skyp #r, theta
		out_array[45:47] = Std_skyp #err on r, theta
		out_array[47] = rho_skyp[0]
		out_array[48:50] = Mean_skyp_corr #r, theta
		out_array[50:52] = Std_skyp_corr #err on r, theta
		out_array[52] = rho_skyp_corr[0]
		out_array[53:55] = Mean_sky_tan #r, theta
		out_array[55:57] = Std_sky_tan#err on r, theta
		out_array[57] = rho_Sky_tan[0]

		#AUX
		out_array[58] 	 = l
		out_array[59] 	 = b
		out_array[60] 	 = ra
		out_array[61] 	 = dec
		out_array[62] 	 = np.nanmean(gcl)
		out_array[63] 	 = np.nanstd(gcl)
		out_array[64] 	 = np.nanmean(Mgl)
		out_array[65] 	 = np.nanstd(Mgl)
		out_array[66]	 = Mean_Dsun_feh[1]
		out_array[67]	 = Std_Dsun_feh[1]
		out_array[68]	 = rho_Dsun_feh[0]
		out_array[69:71] = rho_Dsun_feh[:2]
		out_array[71]    = np.mean(rel)
		out_array[72]    = np.std(rel)

		#CHECK ID
		if id is None and internal_id is None: id = internal_id = ut.create_long_index()
		elif internal_id is None: internal_id = id
		elif id is None: id = internal_id

		if type_input=="rrab": out_array[73]    = 0
		elif type_input=="rrc": out_array[73]    = 1

		out_array[74] 	 = int(id)
		out_array[75] 	 = internal_id

		out_array=np.where(np.isfinite(out_array),out_array,np.nan)

	except ValueError():
		out_array = np.zeros(76)
		out_array[:] = np.nan


	return out_array, dict(zip(_key_list_obs, out_array))


sample_obs_error_5D_key_list_obs_rrl = ('x', 'y', 'z', 'x_err', 'y_err', 'z_err', 'p_x_y', 'p_x_z', 'p_y_z',
					 'Rcyl', 'phi', 'Rcyl_err', 'phi_err', 'p_Rcyl_phi', 'p_Rcyl_z', 'p_phi_z',
					 'r', 'theta',  'r_err', 'theta_err',  'p_r_theta', 'p_r_phi', 'p_theta_phi',
					 'pmra', 'pmdec', 'pmra_err', 'pmdec_err', 'p_pmra_pmdec',
					 'pmra_c', 'pmdec_c', 'pmra_c_err', 'pmdec_c_err', 'p_pmra_c_pmdec',
					 'pml', 'pmb', 'pml_err', 'pmb_err', 'p_pml_pmb',
					 'pml_c', 'pmb_c', 'pml_c_err', 'pmb_c_err', 'p_pml_c_pmb_c',
					 'Vl', 'Vb', 'Vl_err', 'Vb_err', 'p_Vl_Vb',
					 'Vl_c', 'Vb_c', 'Vl_c_err', 'Vb_c_err', 'p_Vl_c_Vb_c',
					 'dsun', 'Vtan_c', 'dsun_err', 'Vtan_c_err', 'p_dsun_Vtan_c',
					 'l', 'b', 'ra', 'dec', 'gc', 'gc_err', 'Mg', 'Mg_err', 'feh', 'feh_err',
					 'p_dsun_feh', 'p_Vl_c_feh', 'p_Vb_c_feh', 're', 're_err', 'type_rrl', 'source_id', 'id')




def test_Mg():

	path_to_file = os.path.abspath(os.path.dirname(__file__)) + "/data/rrl_metallicity_fit_trace/trace_fe_pf_rel_layden_mod.csv"

	print(path_to_file)

	t=pd.read_csv(path_to_file)
	print(t)

	path_to_file = os.path.abspath(os.path.dirname(__file__)) + "/data/period_phi31_fit/RRab_SOS_pf_phi31.xdgmm"
	xdgmm_ab = XDGMM(filename=path_to_file)
	print(xdgmm_ab)

	path_to_file = os.path.abspath(os.path.dirname(__file__)) + "/data/period_phi31_fit/RRc_SOS_p1_phi31.xdgmm"
	xdgmm_c = XDGMM(filename=path_to_file)
	print(xdgmm_c)

	return t




def _test():

	p=[0.6,0.4]
	perr=[0.00,00]
	phi31=[2.5,1.5]
	phi31err=[0.0,0]
	Fe_stat, Fe =metallicity_RRab(period=p, phi31=phi31,Nsample=10000, period_err=perr, phi31_err=phi31err)
	print(Fe_stat)
	Fe_stat, Fe =metallicity_RRab(period=p, phi31=phi31,Nsample=10000, period_err=perr, phi31_err=phi31err, default_trace="nemec")
	print(Fe_stat)
	p=[0.3,0.2]
	perr=[0.00,00]
	phi31=[3.5,4.0]
	phi31err=[0.0,0]
	Fe_stat, Fe =metallicity_RRc(period=p, phi31=phi31,Nsample=10000, period_err=perr, phi31_err=phi31err)
	print(Fe_stat)


	fel=[-0.5,-1.5]
	fel_err=[0.3,0.3]
	Mg_stat, Mg = estimate_Mg(fe=fel, fe_err=fel_err,Nsample=10000)
	print(Mg_stat)
	print(Mg)

	p=[0.9,]
	perr=[0.000001,]
	phi31=[3,]
	phi31err=[0.2,]
	Fes = sample_metallicity_RRab(period=p, phi31=phi31,Nsample=100000, period_err=perr, phi31_err=phi31err)
	Mg_list=sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))


	p,phi31=sample_p_phi31_RRab(N=100000)
	Fes=sample_metallicity_RRab(period=p, phi31=phi31,Nsample=1)
	Mg_list = sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))

	p,phi31=sample_p_phi31_RRab(p=0.4,N=100000)
	Fes=sample_metallicity_RRab(period=p, phi31=phi31,Nsample=1)
	Mg_list = sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))
	print('We',np.mean(phi31), np.std(phi31))


	p=[0.4,]
	perr=[0.000000,]
	phi31=[2.5,]
	phi31err=[0.55,]
	Fes = sample_metallicity_RRab(period=p, phi31=phi31,Nsample=1000000, period_err=perr, phi31_err=phi31err)
	Mg_list=sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))

	print()
	p,phi31=sample_p_phi31_RRc(p=0.3,N=100000)
	Fes=sample_metallicity_RRc(period=p, phi31=phi31,Nsample=1)
	Mg_list = sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))
	print('We',np.mean(phi31), np.std(phi31))


	p=[0.3,]
	perr=[0.000000,]
	phi31=[3.5,]
	phi31err=[0.55,]
	Fes = sample_metallicity_RRc(period=p, phi31=phi31,Nsample=1000000, period_err=perr, phi31_err=phi31err)
	Mg_list=sample_Mg_single(Fes)
	print(np.mean(Fes), np.std(Fes))
	print(np.mean(Mg_list), np.std(Mg_list))

	return Fes, Mg_list


if __name__=="__main__":

	g = 18.0858
	bp_rp = 1.40969
	ebv = 0.552063

	print("Using color")
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, ebv_error=0.16*ebv,Nsample=10000)
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, ebv_error=0.16*ebv, type="RRc",Nsample=10000)

	print("Whitout color")
	sample_Dsun_single(g=g, ebv=ebv, ebv_error=0.16*ebv,Nsample=100000)
	sample_Dsun_single(g=g, ebv=ebv, ebv_error=0.16*ebv, type="RRc",Nsample=10000)

	print("With period")
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, period=0.6, ebv_error=0.16*ebv,Nsample=10000)
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, period=0.3, ebv_error=0.16*ebv, type="RRc",Nsample=10000)

	print("With period andphi31 ")
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, period=0.6, phi31=2.7, ebv_error=0.16*ebv,Nsample=10000)
	sample_Dsun_single(g=g, ebv=ebv, bp_rp=bp_rp, period=0.3, phi31=2.7, ebv_error=0.16*ebv, type="RRc",Nsample=10000)

#def _estimate_Mg( ):
