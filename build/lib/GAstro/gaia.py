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

		#BP
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
		A0=3.1*ebv

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

	def kg_iterative(self, bp_rp, ebv, Nmax=1000, abs_tollerance=0.001, rel_tollerace=0.001):

		bp_rp_0 = self.kbr_iterative(bp_rp, ebv, Nmax=Nmax, abs_tollerance=abs_tollerance, rel_tollerace=rel_tollerace)

		return self.kg(bp_rp_0, ebv)


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