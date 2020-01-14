from GAstro.rrl import sample_obs_error_5D_rrl

"""
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
	 period_error
	 phi31: LC Phase (can be None),
	 phi31_error:
	 type: "rrab" or "rrc",
	 distance: Heliocentric distance in kpc (can be None if gc is provided)
	 distance_error:
	 internal_id: a user defined internal_id (can be None)".
"""

id=1635721458409799680
ra=174.330602556356
dec=23.9518689129458
l=78.07159936606098
b=35.8787594670742
pmra=1.66922636532664
pmdec=0.835775447555281
pmra_err=0.102864145829596
pmdec_err=0.0657665852617861
cov_pmra_pmdec=0.0312125
g=16.3507
g_sos=16.3286776890802
bp_rp=1.17203
ebv=0.573976
period=0.6
period_1o=None
period_error=0
phi31=None
phi31_error=None
type="rrab"
distance=None
distance_error=None
internal_id=None

plist=[id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, g, g_sos, bp_rp, ebv, period, period_1o, period_error,phi31,phi31_error, type, distance, distance_error, internal_id]

Rsun=8.13
Rsun_err=0.03
zsun=0
Vlsr=238
Vlsr_err=9
U=11.1
U_err=1.23
V=12.24
V_err=2.05
W=7.25
W_err=0.63
Mg=0.64
N=100000
nproc=4

_,d=sample_obs_error_5D_rrl(plist,Rsun=Rsun, Rsun_err=Rsun_err, U=U, V=V, W=W, U_err=U_err, V_err=V_err, W_err=W_err, Vlsr=Vlsr, Vlsr_err=Vlsr_err, N=N, sos_correction=True)
print(d)
