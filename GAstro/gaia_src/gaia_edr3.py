import numpy as np

def correct_flux_excess_factor_edr3(bp_rp, phot_bp_rp_excess_factor):
	"""
	Calculate the corrected flux excess factor for the input Gaia EDR3 data (Eq. 6 from Riello+2020).
	This function is taken directly from https://github.com/agabrown/gaiaedr3-flux-excess-correction/blob/main/FluxExcessFactorCorrectionCode.ipynb

	:param bp_rp: float, numpy.ndarray
			The (BP-RP) colour listed in the Gaia EDR3 archive.
	:param phot_bp_rp_excess_factor: float, numpy.ndarray
		The flux excess factor listed in the Gaia EDR3 archive.
	:return: The corrected value for the flux excess factor, which is zero for "normal" stars.
	:example: phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
	"""



	if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
		bp_rp = np.float64(bp_rp)
		phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)

	if bp_rp.shape != phot_bp_rp_excess_factor.shape:
		raise ValueError('Function parameters must be of the same shape!')

	do_not_correct = np.isnan(bp_rp)
	bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
	greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
	redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)

	correction = np.zeros_like(bp_rp)
	correction[bluerange] = 1.154360 + 0.033772 * bp_rp[bluerange] + 0.032277 * np.power(bp_rp[bluerange], 2)
	correction[greenrange] = 1.162004 + 0.011464 * bp_rp[greenrange] + 0.049255 * np.power(bp_rp[greenrange], 2) \
							 - 0.005879 * np.power(bp_rp[greenrange], 3)
	correction[redrange] = 1.057572 + 0.140537 * bp_rp[redrange]

	return phot_bp_rp_excess_factor - correction