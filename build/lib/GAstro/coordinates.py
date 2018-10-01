import numpy as np
import gala.coordinates as gala
import astropy.coordinates as coord
import astropy.units as u

def pmradec_solar_correction(ra, dec, dist, pmra, pmdec,vsun=(11.1, 12.2, 7.25),vlsr=235, vrad=0):

	vsun=np.array(vsun)
	vsun[1]=vsun[1]+vlsr
	
	
	c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=0*u.km/u.s)

	vsun = coord.CartesianDifferential(vsun*u.km/u.s)
	gc_frame = coord.Galactocentric(galcen_v_sun=vsun, z_sun=0*u.kpc)
	ccorr=gala.reflex_correct(c,gc_frame) 
	
	return ccorr.pm_ra_cosdec.value, ccorr.pm_dec.value
	
def pmlb_solar_correction(l, b, dist, pml, pmb,vsun=(11.1, 12.2, 7.25),vlsr=235, vrad=0):

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
		
	