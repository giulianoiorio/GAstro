################################################
#
#
#
# Utils to transform different
#
#
#
#
#################################################
#FUNCTIONS
#m_to_dist(m,M)
#cylindrical_to_spherical(AR,Ap,Az,phi,theta)
#spherical_to_cartesian(Ar, Aphi, Atheta, phi, theta, true_theta=False, degree=True)
#cartesian_to_spherical(Ax, Ay, Az, phi, theta, true_theta=False, degree=True)
#cartesian_to_cylindrical(Ax, Ay, Az, phi, degree=True)
#XYZ_to_lbd(R,phi,z, xsun=8)
#lbd_to_XYZ(l,b,d,xsun=8)
#HRF_to_LRF(ra, dec, VL, raL, decL, muraL, mudecL, DL)
#################################################

import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import galpy.util.bovy_coords as co
import matplotlib as mpl
from multiprocessing import Pool
import time
from functools import partial
import multiprocessing as mp
import roteasy as rs


#INNER IMPORT
from .stat import mad, calc_covariance
from . import coordinates as gao
from . import constant as COST
from . import utility as ut
from .cutils import calc_m

#STUFF FOR BETTER PLOT
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams['contour.negative_linestyle'] = 'solid'
mpl.rcParams['axes.facecolor'] = 'white'
prop_cycle = mpl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def m_to_dist(m,M):
	"""
	Pass from observed magnitude m to heliocentric distance D given the absolute magnitude 
	M.
	:param m: observed magnitude
	:param dec:  absolute magnitude
	:return: Heliocentric distance
	"""
	
	distance_modulus = (m-M)
	power_argument   = distance_modulus/5 -2
	Dsun             = 10**power_argument

	return Dsun

def cartesian_to_sky(Ax,Ay,Az,l,b, degree=True):
	"""

	:param Ax:
	:param Ay:
	:param Az:
	:param l:
	:param b:
	:param degree:
	:return:
	"""

	if degree:
		l=np.radians(l)
		b=np.radians(b)

	cb = np.cos(b)
	sb = np.sin(b)
	cl = np.cos(l)
	sl = np.sin(l)

	Vb   = -Ax*cl*sb -Ay*sl*sb + Az*cb
	Vl   = -Ax*sl + Ay*cl
	Vlos = Ax*cl*cb + Ay*sl*cb + Az*sb

	return Vlos, Vl, Vb

def sky_to_cartesian(Alos, Al, Ab, l, b, degree=True):
	"""

	:param Alos:
	:param Al:
	:param Ab:
	:param l:
	:param b:
	:param degree:
	:return:
	"""

	if degree:
		l=np.radians(l)
		b=np.radians(b)

	cb = np.cos(b)
	sb = np.sin(b)
	cl = np.cos(l)
	sl = np.sin(l)

	Ax =  -Ab*cl*sb -Al*sl + Alos*cl*cb
	Ay = -Ab*sl*sb + Al*cl + Alos*sl*cb
	Az = Ab*cb + Alos*sb



	return Ax, Ay, Az

def sky_to_spherical(Alos, Al, Ab, l,b, phi, theta, true_theta=False, degree=True):
	"""

	:param Alos:
	:param Al:
	:param Ab:
	:param l:
	:param b:
	:param phi:
	:param theta:
	:param true_theta:
	:param degree:
	:return:
	"""

	Ax, Ay, Az = sky_to_cartesian(Alos, Al, Ab, l, b, degree=degree)
	Ar, Atheta, Aphi = cartesian_to_spherical(Ax, Ay, Az, phi, theta, true_theta=true_theta, degree=degree)

	return Ar, Atheta, Aphi


def cylindrical_to_spherical(AR, Az, Aphi, theta, true_theta=False, degree=True):

	if degree: theta = np.radians(theta)
	if true_theta==True:
		theta= np.pi/2. -  theta
		Stheta=-1
	else:
		Stheta=1

	cost = np.cos(theta)
	sint = np.sin(theta)

	Ar     =  AR*cost + Az*sint
	Atheta = -AR*sint + Az*cost

	return Ar, Atheta*Stheta, Aphi

def spherical_to_cylindrical(Ar, Atheta, Aphi, theta, true_theta=False, degree=True):

	if degree: theta = np.radians(theta)
	if true_theta==True:
		theta= np.pi/2. -  theta
		Stheta=-1
	else:
		Stheta=1

	cost = np.cos(theta)
	sint = np.sin(theta)

	AR = Ar*cost - Atheta*sint
	Az = Ar*sint + Atheta*cost

	return AR, Az*Stheta, Aphi

def cylindrical_to_spherical_old(AR,Ap,Az,phi,theta):

	st=np.sin(theta)
	ct=np.cos(theta)
	sp=np.sin(phi)
	cp=np.cos(phi)

	#From Cylindrical to Cartesian
	Ax = AR*cp-Ap*sp
	Ay = AR*sp+Ap*cp

	#From Cartesian to Spherical
	Ar = Ax*(st*cp)+Ay*(st*sp)+Az*(ct)
	At = Ax*(ct*cp)+Ay*(ct*sp)+Az*(-st)

	#Note that in cylindrical and spherical coordinates Aphi is the same
	#Ar along the 3D radial direction
	#At along the teta angle (zenital angle)
	#Ap along the phi angle (azimuthal angle)


	return Ar, At, Ap, Ax, Ay

def spherical_to_cartesian(Ar, Aphi, Atheta, phi, theta, true_theta=False, degree=True):
	"""
	Transform a vector from spherical to cartesian coordinate
	:param Ar: Vector component along the radial direction
	:param Aphi:  Vector component along the azimuthal direction
	:param Atheta: Vector component along the zenithal direction
	:param phi: azimuthal angle, i.e. phi=arctan(y/x) [degrees or rad]
	:param theta:  zenithal angle, if true_theta=True: theta=np.arccos(z/r), if true_theta=False: theta=np.arcsin(z/r) [degrees or rad]
	:param true_theta: see above
	:param degree: If true, phi and theta are expressed in degrees else in radians
	:return: x,y,z component of the Vector
	"""

	if degree: phi, theta = np.radians(phi), np.radians(theta)
	if true_theta==False:
		theta= np.pi/2. -  theta
		Atheta=-Atheta

	cost = np.cos(theta)
	sint = np.sin(theta)
	cosf = np.cos(phi)
	sinf = np.sin(phi)

	Ax = Ar * sint * cosf + Atheta * cost * cosf - Aphi * sinf
	Ay = Ar * sint * sinf + Atheta * cost * sinf + Aphi * cosf
	Az = Ar * cost        - Atheta * sint

	return Ax, Ay, Az

def cartesian_to_spherical(Ax, Ay, Az, phi, theta, true_theta=False, degree=True):
	"""
	Transform a vector from spherical to cartesian coordinate
	:param Ax: Vector component along x-axis
	:param Ay:  Vector component along y-axis
	:param Az: Vector component along z-axis
	:param phi: azimuthal angle, i.e. phi=arctan(y/x) [degrees or rad]
	:param theta:  zenithal angle, if true_theta=True: theta=np.arccos(z/r), if true_theta=False: theta=np.arcsin(z/r) [degrees or rad]
	:param true_theta: see above
	:param degree: If true, phi and theta are expressed in degrees else in radians
	:return: r, theta,phi component of the vector
	"""

	costheta=1
	if degree: 
		phi, theta = np.radians(phi), np.radians(theta)
	if true_theta==False:
		theta= np.pi/2. - theta
		costheta=-1
	

	cost = np.cos(theta)
	sint = np.sin(theta)
	cosf = np.cos(phi)
	sinf = np.sin(phi)
	
	Ar      =    Ax*sint*cosf + Ay*sint*sinf + Az*cost
	Atheta  =    Ax*cost*cosf + Ay*cost*sinf - Az*sint
	Aphi    =	-Ax*sinf      + Ay*cosf
	

	return Ar, Atheta*costheta, Aphi
	
def cartesian_to_cylindrical(Ax, Ay, Az, phi, degree=True):
	"""
	Transform a vector from spherical to cylindrical coordinate
	:param Ax: Vector component along x-axis
	:param Ay:  Vector component along y-axis
	:param Az: Vector component along z-axis
	:param phi: azimuthal angle, i.e. phi=arctan(y/x) [degrees or rad]
	:param degree: If true, phi and theta are expressed in degrees else in radians
	:return: R, phi, z component of the vector
	"""

	costheta=1
	if degree: phi = np.radians(phi) 

	
	cosf = np.cos(phi)
	sinf = np.sin(phi)

	AR      =    Ax*cosf + Ay*sinf 
	Aphi    =   -Ax*sinf + Ay*cosf 

	return AR, Aphi, Az

def XYZ_to_lbd(R,phi,z, xsun=8):
	"""
	Take the X,Y,Z coordinate in the Galactic reference and return the l,b distance from the sun assuming xsun as distance of the sun from the Galactic centre
	:param cord: X,Y,Z in galactic frame of reference, (left-hand X toward the sun. Y toward the sun rotation)
	:param xsun: position of the sun in kpc
	:return:
		l: Galactic longitude
		b: Galactic latitude
		Dsun:  distance from the sun
	"""
	cost=360./(2*np.pi)
	phirad=phi/cost

	x_s=xsun-R*np.cos(phirad)
	y_s=R*np.sin(phirad)
	z_s=z


	rad_s=np.sqrt(x_s*x_s+y_s*y_s+z_s*z_s)

	l=np.arctan2(y_s,x_s)*cost
	l[l<0]+=360
	b=np.arcsin(z_s/rad_s)*cost

	return l,b,rad_s

def lbd_to_XYZ(l,b,d,xsun=8):
	"""
	Pass from l,b and d (heliocentric distance) to the Rectulangar (left-handed) Galactic frame of 
	:param l: Galactic longitude [degree]
	:param b: Galactic latitude [degree]
	:param d: heliocentric distance [kpc]
	:param xsun: position of the sun in kpc
	:return:
		x: Galactic x (positive toward the Sun).
		y: Galactic y (positive toward the Galactic rotational motion). 
		z: Galactic z positive toward the North Cap.
	"""
	l=np.radians(l)
	b=np.radians(b)
	z_s=d*np.sin(b)
	R_s=d*np.cos(b)
	x_s=R_s*np.cos(l)
	y_s=R_s*np.sin(l)
	
	z_g=z_s
	y_g=y_s
	x_g=xsun-x_s

	return x_g, y_g, z_g

def xyz_to_m(x,y,z,q=1.0,qinf=1.0,rq=10.0,p=1.0,alpha=0,beta=0,gamma=0,ax='zyx'):
	"""
	Return the m-value of an ellipsoid from the observ magnitude and galactic coordinate.
	if q=1 and p=1, the ellipsoid is indeed a sphere and m=r
	:param x: Galactic x (lhf, sun is a x~8)
	:param y: Galactic y
	:param z: Galactic z
	:param q: Flattening along the z-direction, q=1 no flattening.
	:param p: Flattening along the y-direction, p=1 no flattening.
	:return: the m-value for an ellipsoid m^2=x^2+(y/p)^2+(z/q)^2.
	"""

	x = np.asarray(x, dtype=np.float64)
	y = np.asarray(y, dtype=np.float64)
	z = np.asarray(z, dtype=np.float64)


	i=np.abs(alpha)+np.abs(beta)+np.abs(gamma)
	if i!=0:
		cord=rs.rotate_frame(cord=np.array([x,y,z]).T, angles=(alpha,beta,gamma), axes=ax, reference='lh' )
		x=cord[:,0]
		y=cord[:,1]
		z=cord[:,2]

	if q==qinf:
		y=y/p
		z=z/q
		m=np.sqrt(x*x+y*y+z*z)
	else:
		m=np.array(calc_m(x,y,z, q, qinf, rq, p))

	return m


def HRF_to_LRF(ra, dec, VL, raL, decL, muraL, mudecL, DL):
	"""
	Pass from an Heliocentric frame of referece velocity to a Local (e.g. Dwarf) frame of reference velocity (Appendix A, Walker08)
	:param ra: ra of the stars [degree]
	:param dec: dec of the stars  [degree]
	:param VL: systemic velocity of the LRF [km/s]
	:param raL: ra of the centre of  LRF [degree]
	:param decL: dec of the centre of  LRF [degree]
	:param muraL: mura of the centre of  LRF [mas/yr]
	:param mudecL: mudec of the centre of  LRF [mas/yr]
	:param DL: distance of the LRF wrt the Sun [kpc]
	:return: vrel(ra, dec) (see Walker08)
	"""
	
	ra=np.radians(ra)
	dec=np.radians(dec)
	raL=np.radians(raL)
	decL=np.radians(decL)
	cdL=np.cos(decL)
	crL=np.cos(raL)
	sdL=np.sin(decL)
	srL=np.sin(raL)
	cd=np.cos(dec)
	cr=np.cos(ra)
	sd=np.sin(dec)
	sr=np.sin(ra)
	
	K=4.74 #From kpc*mas/yr to km/s
	VraL= K*DL*muraL
	VdecL=  K*DL*mudecL
	
	
	Bx = cd*sr
	By = -cd*cr
	Bz = sd
	
	Adotx = VL*cdL*srL + VraL*cdL*crL -VdecL*sdL*srL
	Adoty = -VL*cdL*crL + VdecL*sdL*crL + VraL*cdL*srL
	Adotz = VL*sdL + VdecL*cdL
	
	return Bx*Adotx + By*Adoty + Bz*Adotz

#####################
#Sample from obs
_str_plist="(id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc, distance, distance_error, internal_id)"
_str_kpc="kpc"
_str_kms="km/s"
def _make_Vsunl(U:_str_kms=11.1, V:"km/s"=12.24, W:"km/s"=7.25, U_err:"km/s"=None, V_err:"km/s"=None, W_err:"km/s"=None, Vlsr:"km/s"=235, Vlsr_err:"km/s"=None, N:"int"=1000)->"Nx3 km/s":
	"""
	Auxilary function for sample_obs_error. It generates a MC sample of U,V,W solar motions.
	:param U: Solar motion (wrt LSR) toward the Galactic center
	(NB: here it is defined positive if it is toward the Galctice centre, but sample_obs_erro we used a left-hand system,
	in this system a motion toward the GC is negatie. However this converstion is automatically made in sample_obs).
	:param V: Solar proper motion (wrt LSR) along the direction of Galactic rotation.
	:param W: Solar proper motion (wrt LSR) along the normal to the Galactic plane (positive value is an upaward motion).
	:param U_err: Error on U.
	:param V_err: Error on V.
	:param W_err: Error on W.
	:param Vlsr:  Circular motion of the LSR.
	:param Vlsr_err:  Error on Vlsr.
	:param N:  Number of sample to extract.
	:return:  a Nx3 Array with a realisation of (U,V,W) in each row.
	"""
	onesl   = np.ones(N) #list of ones
	#Vsun
	if Vlsr_err is None: Vlsrl=onesl*Vlsr
	else: Vlsrl=np.random.normal(Vlsr, Vlsr_err)
	if U_err is None: Ul=onesl*U
	else: Ul=np.random.normal(U, U_err)
	if V_err is None: Vl=onesl*V
	else: Vl=np.random.normal(V, V_err)
	if W_err is None: Wl=onesl*W
	else: Wl=np.random.normal(W, W_err)
	Vsunl = np.vstack((Ul, Vl+Vlsrl, Wl)).T

	return  Vsunl

#TODO: I don't like this implementation with property_list. It is nice for parallelisation but it not very user-friendly. Maybe we have to create a class od create more high-level user wrapper.
def sample_obs_error_5D(property_list:_str_plist, Mg:"mag"=0.64, Mg_err:"mag"=0.24, Rsun:_str_kpc=8.2, Rsun_err:_str_kpc=None, U:_str_kms=11.1, V:_str_kms=12.24, W:_str_kms=7.25, U_err:_str_kms=None, V_err:_str_kms=None, W_err:_str_kms=None, Vlsr:_str_kms=235, Vlsr_err:_str_kms=None, N:"int"=1000)->"array and dic with properties":
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
		 gc: G magnitude corrected for extinction (can be None if distance is provided),
		 distance: Heliocentric distance in kpc (can be None if gc is provided)
		 distance_error:
		 internal_id: a user defined internal_id (can be None)".
	:param Mg: Absolute magnitude to estimate distance from gc.
	:param Mg_err: error on Absolute magnitude.
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
					 'l', 'b', 'ra', 'dec', 'gc', 'source_id', 'id')

	_K = COST._K
	cts = cartesian_to_spherical
	stc = spherical_to_cartesian

	id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc, distance, distance_error, internal_id = property_list
	onesl   = np.ones(N) #list of ones
	ral     = onesl*ra
	decl    = onesl*dec
	ll      = onesl*l
	bl      = onesl*b
	ll = np.radians(ll)
	bl = np.radians(bl)

	cov_pm               =  pmra_err*pmdec_err*cov_pmra_pmdec
	cov_matrix           =  [ [pmra_err**2, cov_pm],  [cov_pm, pmdec_err**2] ]
	pmral, pmdecl        =  np.random.multivariate_normal( [pmra, pmdec], cov_matrix, N).T
	pmll, pmbl		     =  co.pmrapmdec_to_pmllpmbb(pmral, pmdecl, ral, decl, degree=True).T

	#Check  if we have to use distance (priority) or gc and Mg
	if distance is not None and distance_error is not None:
		Dsunl  = np.random.normal(distance, distance_error,N)
	elif distance is not None:
		Dsunl  = np.repeat(distance, N)
	elif gc is not None:
		Mgl = np.random.normal(Mg, Mg_err, N)
		Dsunl = m_to_dist(gc, Mgl)
	else:
		raise ValueError("distance and gc cannot be both None")
	#

	#Rsun
	if Rsun_err is None: Rsunl = onesl*Rsun
	else: Rsunl = np.random.normal(Rsun, Rsun_err, N)


	#Vsun
	Vsunl = _make_Vsunl(U, V, W, U_err, V_err, W_err, Vlsr, Vlsr_err, N)

	# LHS COORD
	xsl    =  Dsunl * np.cos(ll) * np.cos(bl)
	yl     =  Dsunl * np.sin(ll) * np.cos(bl)
	zl     =  Dsunl * np.sin(bl)
	xl     =  Rsunl - xsl
	Rl     =  np.sqrt(xl * xl + yl * yl)
	phil   =  np.arctan2(yl, xl)
	rl     =  np.sqrt( Rl*Rl + zl*zl )
	thetal =  np.arcsin(zl/rl)


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

	#pmra pmdec corr
	pmral_corr, pmdecl_corr = co.pmllpmbb_to_pmrapmdec(pmll_corr,pmbl_corr, ll, bl, degree=False, epoch=2000.0).T
	Mean_skyeq_corr, Std_skyeq_corr, rho_skyeq_corr = calc_covariance(pmral_corr, pmdecl_corr)

	out_array = np.zeros(65)
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
	out_array[62] 	 = gc


	#CHECK ID
	if id is None and internal_id is None: id = internal_id = ut.create_long_index()
	elif internal_id is None: internal_id = id
	elif id is None: id = internal_id


	out_array[63] 	 = int(id)
	out_array[64] 	 = internal_id



	return out_array, dict(zip(_key_list_obs, out_array))

sample_obs_error_5D_key_list_obs = ('x', 'y', 'z', 'x_err', 'y_err', 'z_err', 'p_x_y', 'p_x_z', 'p_y_z',
					 'Rcyl', 'phi', 'Rcyl_err', 'phi_err', 'p_Rcyl_phi', 'p_Rcyl_z', 'p_phi_z',
					 'r', 'theta', 'r_err', 'theta_err', 'p_r_theta', 'p_r_phi', 'p_theta_phi',
					 'pmra', 'pmdec', 'pmra_err', 'pmdec_err', 'p_pmra_pmdec',
					 'pmra_c', 'pmdec_c', 'pmra_c_err', 'pmdec_c_err', 'p_pmra_c_pmdec',
					 'pml', 'pmb', 'pml_err', 'pmb_err', 'p_pml_pmb',
					 'pml_c', 'pmb_c', 'pml_c_err', 'pmb_c_err', 'p_pml_c_pmb_c',
					 'Vl', 'Vb', 'Vl_err', 'Vb_err', 'p_Vl_Vb',
					 'Vl_c', 'Vb_c', 'Vl_c_err', 'Vb_c_err', 'p_Vl_c_Vb_c',
					 'dsun', 'Vtan_c', 'dsun_err', 'Vtan_c_err', 'p_dsun_Vtan_c',
					 'l', 'b', 'ra', 'dec', 'gc', 'source_id', 'id')

#####################



#####
	
	
if __name__=="__main__":

	print(_make_Vsunl.__annotations__)
	print(sample_obs_error.__annotations__)
	print(ut.create_long_index())

