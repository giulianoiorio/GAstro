import numpy as np



def make_simplex(array):
	"""
	Generate a K-dimensional uniform Simplex (N values between 0 and 1 and with total sum=1), starting from
	an array of Uniform generated value between 0 an 1. It uses the Exponential sampling method (http://blog.geomblog.org/2005/10/sampling-from-simplex.html).
	:param array: 2D Array with dimension NxK, where N are the number of point we want to generate a simples and K is the simplex dimension.
	:return:  a 2D Array with dimension NxK storing the simplex value obtained from the poin in input. The simplex distribution is uniform
	but this can be used as a domain for a Dirichlet distribution to generate non-uniform simplex samples.
	"""

	input_arr = np.atleast_2d(array)
	if  not np.all( (input_arr<1)&(input_arr>0) ): raise ValueError('Input elements of simplex needs to be within 0-1')
	Wy = -np.log(input_arr)
	Wy = Wy / np.sum(Wy, axis=1)[:, None]

	#Old implementation, it should be excatly the same but with a not useful(?) intermediate passage
	#N = input_arr.shape[0]
	#Ay = np.c_[np.ones(N), input_arr]
	#Wy = -np.log(Ay)  # sampled from e^-1, this means uniform samplig on the difference that naturally sum to 1
	#Wy = np.cumsum(Wy, axis=1) / np.sum(Wy, axis=1)[:, None]  # These are not sorted number in ascending order
	#Wy = np.diff(Wy)  # Here we go

	return Wy




if __name__=='__main__':


	N=100000
	A = np.random.uniform(0, 1, N * 3).reshape(N, 3)

	W=make_simplex(A)
	print(np.sum(W)==N)


	import matplotlib.pyplot as plt
	from GAstro.plot import ploth2
	fig, axl = plt.subplots(1, 1)

	print(np.mean(W,axis=0))
	print(np.std(W,axis=0))
	print(np.std(W[:, 0]), np.sqrt(1/18))

	ploth2(W[:, 0], W[:, 1], ax=axl, bins=20, vmin=0., vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	plt.show()



	'''
	N = 10000

	A = np.random.uniform(0, 1, N * 2).reshape(N, 2)
	Ay = np.c_[np.ones(N), np.random.uniform(0, 1, N * 3).reshape(N, 3)]
	Wy = -np.log(Ay)  # sampled from e^-1, this means uniform samplig on the difference that naturally sum to 1
	Wy = np.cumsum(Wy, axis=1) / np.sum(Wy, axis=1)[:, None]  # These are not sorted number in ascending order
	Wy = np.diff(Wy)  # Here we go

	W = np.array([weights_from_simplex(*t) for t in A])

	Ar = np.random.uniform(0, 1, N * 3).reshape(N, 3)
	Wr = Ar / np.sum(Ar, axis=1)[:, None]

	import matplotlib.pyplot as plt
	from GAstro.plot import ploth2

	fig, axl = plt.subplots(6, 3, figsize=(10, 20))
	axl = axl.flatten()
	label_size = 18

	ploth2(Wy[:, 0], Wy[:, 1], ax=axl[0], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(Wy[:, 0], Wy[:, 2], ax=axl[1], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(Wy[:, 1], Wy[:, 2], ax=axl[2], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	axl[0].set_ylabel('Difference sampling', fontsize=label_size)

	ploth2(Wr[:, 0], Wr[:, 1], ax=axl[3], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(Wr[:, 0], Wr[:, 2], ax=axl[4], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(Wr[:, 1], Wr[:, 2], ax=axl[5], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	axl[3].set_ylabel('Renormalise', fontsize=label_size)

	ploth2(W[:, 0], W[:, 1], ax=axl[6], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)),
		   cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W[:, 0], W[:, 2], ax=axl[7], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)),
		   cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W[:, 1], W[:, 2], ax=axl[8], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)),
		   cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	axl[6].set_ylabel('Breaking stick', fontsize=label_size)

	W2 = np.random.dirichlet([1, 1, 1], N)
	ploth2(W2[:, 0], W2[:, 1], ax=axl[9], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)),
		   cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W2[:, 0], W2[:, 2], ax=axl[10], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W2[:, 1], W2[:, 2], ax=axl[11], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	axl[9].set_ylabel('Dirichlet(1,1,1)', fontsize=label_size)

	W2 = np.random.dirichlet([50, 50, 50], N)
	ploth2(W2[:, 0], W2[:, 1], ax=axl[12], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W2[:, 0], W2[:, 2], ax=axl[13], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	ploth2(W2[:, 1], W2[:, 2], ax=axl[14], bins=20, vmin=0, vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	axl[12].set_ylabel('Dirichlet(50,50,50)', fontsize=label_size)

	W2 = np.random.dirichlet([1.9, 1, 1], N)
	ploth2(W2[:, 0], W2[:, 1], ax=axl[15], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', aspect='auto')
	ploth2(W2[:, 0], W2[:, 2], ax=axl[16], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', aspect='auto')
	ploth2(W2[:, 1], W2[:, 2], ax=axl[17], bins=20, vmin=0., vmax=0.01, vminmax_option='absolute',
		   range=((0, 1), (0, 1)), cmap='viridis', norm='tot', aspect='auto')
	axl[15].set_ylabel('Dirichlet(1.9,1,1)', fontsize=label_size)

	fig.tight_layout()
	plt.savefig('Test_Dirichlet.png')


	import emcee
	from scipy.stats import dirichlet


	def logp(theta):

		if np.all((theta>0)&(theta<1)):
			theta = np.insert(theta,0,1)
			Wy = -np.log(theta)  # sampled from e^-1, this means uniform samplig on the difference that naturally sum to 1
			Wy = np.cumsum(Wy) / np.sum(Wy)  # These are not sorted number in ascending order
			Wy = np.diff(Wy)  # Here we go
			return dirichlet.logpdf(Wy, (2,1,1)), Wy

		else:

			return -np.inf, np.array([np.nan,np.nan,np.nan])



	nwalkers, ndim = 100,3
	pos = np.random.uniform(0, 1, nwalkers*ndim).reshape(nwalkers,ndim)
	nwalkers, ndim = pos.shape

	sampler = emcee.EnsembleSampler(nwalkers, ndim, logp)
	sampler.run_mcmc(pos, 3000, progress=True)
	Wblob = sampler.get_blobs(flat=True)

	fig, axl = plt.subplots(1, 1)

	print(Wblob.shape)

	ploth2(Wblob[:, 0], Wblob[:, 1], ax=axl, bins=20, vmin=0., vmax=0.01, vminmax_option='absolute', range=((0, 1), (0, 1)), cmap='viridis', norm='tot', colorbar=False, aspect='auto')
	plt.show()
	'''