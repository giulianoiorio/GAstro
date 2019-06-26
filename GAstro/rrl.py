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

