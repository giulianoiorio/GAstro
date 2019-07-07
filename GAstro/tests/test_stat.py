from GAstro import stat
import math
import numpy as np



def test_mad():

	tollerance=0.01
	s=np.random.normal(0,1,int(1e5))
	med=np.median(s)
	std=np.std(s)
	m=stat.mad(s)

	assert med==m[0]
	assert math.isclose(m[1], std, rel_tol=tollerance)

def test_calc_covariance():


	tollerance=0.1
	meanl=np.array((1,5,10))
	stdl=np.array((1,2,3))
	rhol=np.array((0.1,-0.3,0.8))
	COV = np.zeros(shape=(3,3))
	np.fill_diagonal(COV, stdl**2)
	COV[0,1]=COV[1,0]=rhol[0]*stdl[0]*stdl[1]
	COV[0,2]=COV[2,0]=rhol[1]*stdl[0]*stdl[2]
	COV[1,2]=COV[2,1]=rhol[2]*stdl[1]*stdl[2]
	print(COV)

	X=np.random.multivariate_normal(meanl, COV,100000)
	mean, std, rho = stat.calc_covariance(X[:,0],X[:,1],X[:,2])

	assert math.isclose(mean[0], meanl[0], rel_tol=tollerance)
	assert math.isclose(mean[1], meanl[1], rel_tol=tollerance)
	assert math.isclose(mean[2], meanl[2], rel_tol=tollerance)
	assert math.isclose(std[0], stdl[0], rel_tol=tollerance)
	assert math.isclose(std[1], stdl[1], rel_tol=tollerance)
	assert math.isclose(std[2], stdl[2], rel_tol=tollerance)
	assert math.isclose(rho[0], rhol[0], rel_tol=tollerance)
	assert math.isclose(rho[1], rhol[1], rel_tol=tollerance)
	assert math.isclose(rho[2], rhol[2], rel_tol=tollerance)



if __name__=='__main__':

	test_calc_covariance()