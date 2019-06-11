import numpy as np
import gala.coordinates as gala
import astropy.coordinates as coord
from astropy import units as u
import roteasy as rt

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


def pmradec_solar_correction(ra, dec, dist, pmra, pmdec,vsun=(11.1, 12.2, 7.25),vlsr=235, vrad=0):

	vsun=np.array(vsun)
	vsun[1]=vsun[1]+vlsr


	c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=0*u.km/u.s)

	vsun = coord.CartesianDifferential(vsun*u.km/u.s)
	gc_frame = coord.Galactocentric(galcen_v_sun=vsun, z_sun=0*u.kpc)
	ccorr=gala.reflex_correct(c,gc_frame)

	return ccorr.pm_ra_cosdec.value, ccorr.pm_dec.value

def pmlb_solar_correction(l, b, dist, pml, pmb, vsun=(11.1, 12.2, 7.25), vlsr=235, vrad=0):

	vsun=np.array(vsun)
	vsun[1]=vsun[1]+vlsr


	c = coord.SkyCoord(l=l*u.deg, b=b*u.deg, distance=dist*u.kpc, pm_l_cosb=pml*u.mas/u.yr, pm_b=pmb*u.mas/u.yr, radial_velocity=0*u.km/u.s, frame='galactic')
	vsun = coord.CartesianDifferential(vsun*u.km/u.s)
	gc_frame = coord.Galactocentric(galcen_v_sun=vsun, z_sun=0*u.kpc)
	ccorr=gala.reflex_correct(c,gc_frame)

	return ccorr.pm_l_cosb.value, ccorr.pm_b.value


def pmradec_to_pmlb(ra, dec, dist, pmra, pmdec, vrad=0, vsun=(11.1, 12.2, 7.25),vlsr=235, solar_correction=False):

	if solar_correction:

		pmra, pmdec = pmradec_solar_correction(ra, dec, dist, pmra, pmdec,vsun=vsun,vlsr=vlsr, vrad=vrad)

	c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=0*u.km/u.s)

	cG = c.galactic


	return cG.pm_l_cosb.value, cG.pm_b.value

def lb_to_radec(l,b):

	c = coord.SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
	cradec=c.icrs

	return cradec.ra.value, cradec.dec.value

def radec_to_sag(ra,dec):

	#Belokurov2014
	ra=np.radians(ra)
	dec=np.radians(dec)
	ca=np.cos(ra)
	sa=np.sin(ra)
	cd=np.cos(dec)
	sd=np.sin(dec)

	ylambda= -0.93595354*ca*cd - 0.31910658*sa*cd + 0.14886895*sd
	xlambda= 0.21215555*ca*cd - 0.84846291*sa*cd -0.48487186*sd
	Lambda=np.arctan2(ylambda, xlambda)

	argbeta= 0.28103559*ca*cd -0.42223415*sa*cd + 0.86182209*sd
	Beta= np.arcsin(argbeta)

	return Lambda*180/np.pi,  Beta*180./np.pi

def radec_to_gnomic(ra,dec,ra_c,dec_c):


	dtr=np.pi/180
	c_centre= coord.SkyCoord(ra=ra_c*u.degree, dec=dec_c*u.degree)
	c = coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
	rc= c.separation(c_centre).degree

	bottom = np.sin(dec*dtr)*np.sin(dec_c*dtr) + np.cos(dec*dtr)*np.cos(dec_c*dtr)*np.cos( (ra-ra_c)*dtr )

	#XI
	xi= np.cos(dec*dtr) * np.sin( (ra-ra_c)*dtr ) / bottom
	xi= xi / dtr

	#ETA
	eta= (np.sin(dec*dtr)*np.cos(dec_c*dtr) - np.cos(dec*dtr)*np.sin(dec_c*dtr)*np.cos( (ra-ra_c)*dtr )) / bottom
	eta= eta / dtr
	rgnomic=np.sqrt(xi*xi+eta*eta)

	return xi, eta, rgnomic

def equatorial_to_pole(ra, dec, pole, pmra=None, pmdec=None, center=(0,0), degree=True):
	"""
	Transform the equatorial coordinates to a new set of spherical coordinates with a pole defined in
	pole. The line of nodes between the the equatorial and the new system is defined with center.
	If both pmra and pmdec are not None, it transform also the proper motions (following Edmonson, 34).
	:param ra: Equatorial right ascension [deg or rad]
	:param dec: Equatorial declination  [deg or rad]
	:param pole: tuple with ra,dec coordinates of the pole of the new system [deg or rad]
	:param pmra: Proper motion along ra [any units]
	:param pmdec: Proper motion along dec [any units]
	:param center: tuple with ra and dec of what will the the phi1=0 of the new coordinate system.
	:param degree: If True ra, dec, pole and center needs to be in degrees, if False  in radians.
	:return: phi1 (longitude in the new system), phi2 (latitude in the new system), pm_phi1 (Proper motin along the longitude), pm_phi2 (Proper motion along the latitude).
	"""
	lo=ra
	la=dec
	if degree:  lo, la, center, pole_deg, pole_rad = np.radians(lo), np.radians(la), np.radians(center), pole, np.radians(pole)
	else: pole_deg, pole_rad=np.degrees(pole), pole

	# Position on the sphere
	clo, slo =  np.cos(lo), np.sin(lo)
	cla, sla =  np.cos(la), np.sin(la)
	cloc, sloc = np.cos(center[0]), np.sin(center[0])
	clac, slac = np.cos(center[1]), np.sin(center[1])

	#######NEW POSITIONS#########
	x, xc = clo * cla, cloc * clac
	y, yc = slo * cla, sloc * clac
	z, zc = sla, slac

	# Pole
	xn, yn, zn = rt.align_frame((x, y, z), pos_vec=pole_deg, ax='z', unpack=True, unpacked=True,
								cartesian=False, spherical=True)
	xnc, ync, znc = rt.align_frame(([xc, ], [yc, ], [zc, ]), pos_vec=pole_deg, ax='z', unpack=True,
								   unpacked=True, cartesian=False, spherical=True)
	#Angles
	phi1c = np.arctan2(ync, xnc)
	phi1  =  np.arctan2(yn, xn) - phi1c
	phi2  =  np.arcsin(zn)
	if degree: phi1, phi2 = np.degrees(phi1), np.degrees(phi2)

	#######PM
	###Transformation of PM following Eq. 5 by Edmonson, 34
	if pmra is not None and pmdec is not None:

		#Pole on the sphere
		rap, decp=pole_rad
		L=np.cos(rap)*np.cos(decp)
		M=np.sin(rap)*np.cos(decp)
		N=np.sin(decp)
		tgtheta=(L*slo-M*clo)/(-L*clo*sla-M*slo*sla+N*cla)
		theta=np.arctan(tgtheta)
		cth,sth=np.cos(theta), np.sin(theta)
		pm_phi1=  pmra*cth + pmdec*sth
		pm_phi2= -pmra*sth + pmdec*cth
	else:
		pm_phi1=pm_phi2=None

	phi1=np.where(phi1<-180,360+phi1, phi1)
	phi1=np.where(phi1>180, phi1-360, phi1)

	return phi1, phi2, pm_phi1, pm_phi2

def equatorial_to_Pal5(ra, dec, pmra=None, pmdec=None, degree=True):
	"""
	Tranform equatorial coordinates to a system aligne with the Pal5 stream (Erkal et al. 2017)
	:param ra: Equatorial right ascension [deg or rad]
	:param dec: Equatorial declination  [deg or rad]
	:param pmra: Proper motion along ra [any units]
	:param pmdec: Proper motion along dec [any units]
	:param degree: If True ra, dec needs to be in degrees, if False  in radians.
	:return: phi1 (longitude in the Pal5 system), phi2 (latitude in the Pal5 system), pm_phi1 (Proper motin along the Pal5 longitude), pm_phi2 (Proper motion along the Pal5 latitude).
	"""

	#Pal 5 prop
	Pal5_pole=(138.95, 53.78) #From Erkal et al. 2017
	Pal5_centre=(229.02,0.11) #Position of Pal5 (from Simbad)

	if degree==False: Pal5_pole, Pal5_centre = np.radians(Pal5_pole), np.radians(Pal5_centre)

	phi1, phi2, pm_phi1, pm_phi2 = equatorial_to_pole(ra, dec, Pal5_pole, pmra, pmdec, Pal5_centre, degree)

	return phi1, phi2, pm_phi1, pm_phi2


if __name__=='__main__':
	def Pal5(alfa, delta, alfapole, deltapole, alfac=0, deltac=0):
		# Position on the sphere
		ra = np.radians(alfa)
		dec = np.radians(delta)
		alfac = np.radians(alfac)
		deltac = np.radians(deltac)
		ca, sa = np.cos(ra), np.sin(ra)
		cd, sd = np.cos(dec), np.sin(dec)
		cac, sac = np.cos(alfac), np.sin(alfac)
		cdc, sdc = np.cos(deltac), np.sin(deltac)
		x, xc = ca * cd, cac * cdc
		y, yc = sa * cd, sac * cdc
		z, zc = sd, sdc

		# Pole
		xn, yn, zn = rt.align_frame((x, y, z), pos_vec=(alfapole, deltapole), ax='z', unpack=True, unpacked=True,
									cartesian=False, spherical=True)
		xnc, ync, znc = rt.align_frame(([xc, ], [yc, ], [zc, ]), pos_vec=(alfapole, deltapole), ax='z', unpack=True,
									   unpacked=True, cartesian=False, spherical=True)


		# Angles
		phi1c = np.arctan2(ync, xnc)
		phi1 = np.degrees(np.arctan2(yn, xn) - phi1c)
		phi2 = np.degrees(np.arcsin(zn))

		# phi1[phi1<0]+=360

		return phi1, phi2


	print(Pal5(229.022, 0.111389, 138.95, 53.78, 0, 0))
	print(change_pole(229.022, 0.111389, pole=(138.95, 53.78), pmlongitude=	-2.343, pmlatitude=-2.3085, center=(0, 0), degree=True))
	print(change_pole(229.022, 0.111389, pole=(192.85948, 27.12825), center=(266.41683708333335, -29.007810555555555), degree=True))
