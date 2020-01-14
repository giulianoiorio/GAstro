import GAstro.transform as tr
import numpy as np
import math


def test_sample_obs_error():

	tollerance=0.1


	Rsun=8.13
	zsun=0
	Vlsr=238
	U=11.1
	V=12.24
	W=7.25

	#Select a point in (lfh) GC coordinate
	x=np.random.uniform(-30,30)
	y=np.random.uniform(-30,30)
	z=np.random.uniform(-30,30)
	vx=np.random.uniform(-200,200)
	vy=np.random.uniform(-200,200)
	vz=np.random.uniform(-200,200)

	import astropy.units as u
	import astropy.coordinates as coord
	import gala.coordinates as gala
	from GAstro.constant import _K

	vu=u.km/u.s
	Vsun = coord.CartesianDifferential(d_x=U*vu, d_y=(V+Vlsr)*vu, d_z=W*vu)
	c = coord.Galactocentric(x=-x*u.kpc, y=y*u.kpc, z=z*u.kpc, v_x=((-vx))*vu, v_y=(vy)*vu, v_z=(vz)*vu, z_sun=zsun*u.kpc, galcen_distance=Rsun*u.kpc, galcen_v_sun=Vsun)
	sc=c.transform_to(coord.ICRS)
	gc=c.transform_to(coord.Galactic)
	gc_frame = coord.Galactocentric(galcen_v_sun=Vsun, z_sun=0*u.kpc,  galcen_distance=Rsun*u.kpc)
	sccorr=gala.reflex_correct(sc,gc_frame)
	gccorr=gala.reflex_correct(gc,gc_frame)


	ra, dec, l, b = sc.ra.value, sc.dec.value, gc.l.value, gc.b.value
	pmra, pmdec = sc.pm_ra_cosdec.value, sc.pm_dec.value
	dhelio = sc.distance.value
	dhelio_err= dhelio*0.05
	pmra_err = 1e-3
	pmdec_err = 1e-3
	cov_pmra_pmdec = 0
	gc_ = None
	id=0
	id_internal = None
	#id, ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc, distance, distance_error, internal_id = property_list

	if l>180: l2=l-360
	else: l2=l

	param_list=(id, ra, dec, l2, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc_, dhelio, dhelio_err, id_internal)
	arr,dic=tr.sample_obs_error_5D(property_list=param_list, Mg=0.64, Mg_err=0.24, Rsun=Rsun, Rsun_err=None, U=U, V=V, W=W, U_err=None, V_err=None, W_err=None, Vlsr=Vlsr, Vlsr_err=None, N=1000)

	print(l,l2)
	print(dic['dsun'])
	print(dic['dsun_err'])
	print(_K*gccorr.pm_l_cosb.value*sccorr.distance.value, dic['Vl_c'])

	param_list=(id, ra, dec, l2, b, pmra, pmdec, pmra_err, pmdec_err, cov_pmra_pmdec, gc_, dhelio, dhelio_err, id_internal)
	arr,dic=tr.sample_obs_error_5D(property_list=param_list, Mg=0.64, Mg_err=0.24, Rsun=Rsun, Rsun_err=None, U=U, V=V, W=W, U_err=None, V_err=None, W_err=None, Vlsr=Vlsr, Vlsr_err=None, N=1000)

	print(_K*gccorr.pm_l_cosb.value*sccorr.distance.value, dic['Vl_c'])



	assert math.isclose(x, dic['x'], rel_tol=tollerance)
	assert math.isclose(y, dic['y'], rel_tol=tollerance)
	assert math.isclose(z, dic['z'], rel_tol=tollerance)
	assert math.isclose(sccorr.distance.value, dic['dsun'], rel_tol=tollerance)
	assert math.isclose(sccorr.pm_ra_cosdec.value, dic['pmra_c'], rel_tol=tollerance)
	assert math.isclose(sccorr.pm_dec.value, dic['pmdec_c'], rel_tol=tollerance)
	assert math.isclose(gccorr.pm_l_cosb.value, dic['pml_c'], rel_tol=tollerance)
	assert math.isclose(gccorr.pm_b.value, dic['pmb_c'], rel_tol=tollerance)
	assert math.isclose(gc.pm_l_cosb.value, dic['pml'], rel_tol=tollerance)
	assert math.isclose(gc.pm_b.value, dic['pmb'], rel_tol=tollerance)
	assert math.isclose(_K*gccorr.pm_l_cosb.value*sccorr.distance.value, dic['Vl_c'], rel_tol=tollerance)
	assert math.isclose(_K*gccorr.pm_b.value*sccorr.distance.value, dic['Vb_c'], rel_tol=tollerance)
	assert math.isclose(_K*gc.pm_l_cosb.value*sccorr.distance.value, dic['Vl'], rel_tol=tollerance)
	assert math.isclose(_K*gc.pm_b.value*sccorr.distance.value, dic['Vb'], rel_tol=tollerance)

if __name__=="__main__":
	test_sample_obs_error()