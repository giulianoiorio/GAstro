################################################
#
#
#
# Utils to generate mock dataset
#
#
#
#
################################################
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from pycam.plot import ploth2, ploth1
import matplotlib.pyplot as plt

label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams['contour.negative_linestyle'] = 'solid'
mpl.rcParams['axes.facecolor'] = 'white'
prop_cycle = mpl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def Generate_CMD(N,fCMD,collim=(-0.3,2.5),MGlim=(-2.5,8)):
	"""
	Generate Color, Magnitude diagram from a f(col,Mag) function.
	It uses a simple 2D acceptance rejection method.
	:param N: Guess number of particles
	:param fCMD: 2D callable functions
	:param collim: color range
	:param MGlim: absolute magnitude range
	:return: color, Magnitude
	"""
	colS = np.random.uniform(collim[0], collim[1], N)
	MGS = np.random.uniform(MGlim[0], MGlim[1], N)
	u = np.random.uniform(0, 1, N)
	dens = fCMD(colS, MGS, grid=False)
	idx = u <= dens

	return colS[idx], MGS[idx]

def assign_CMD(Ndata, fCMD, collim=(-0.3,2.5),MGlim=(-2.5,8), Nini=None, plotname=None, verbose=True):

	"""
	Generata color and magnitude given a Color-Magnitude diagram model.
	It iteratively calls Generate_CMD, until the final number of data match the requested number
	:param Ndata: Number of data to generate
	:param fCMD:  2D callable functions
	:param collim: color range
	:param MGlim: absolute magnitude range
	:param Nini: initial guess of data to generate
	:param plotname:  Name of the summary plot. It could contain the directory. It could be None
	(the plot will be not generated).
	:param verbose:  If True print runtime messages
	:return:
	"""


	if Nini is not None: Nguess=Nini
	elif Ndata>1e6: Nguess=int(Ndata)
	else: Nguess=int(2*Ndata)


	Ncurrent=0
	count=0
	col=np.zeros(int(Ndata))
	MG=np.zeros(int(Ndata))

	while(Ncurrent<Ndata):

		if verbose: print('Attempt %i: \nNguess=%i'%(count, Nguess)  )

		col_tmp,MG_tmp=Generate_CMD(Nguess, fCMD, collim=collim, MGlim=MGlim)
		Ngenerate=len(col_tmp)

		if Ngenerate>Ndata:
			col=col_tmp[:Ndata]
			MG=MG_tmp[:Ndata]
		elif (Ncurrent+Ngenerate)>Ndata:
			print('len',len(col_tmp))
			print(Ngenerate,Ncurrent+Ngenerate)
			col[Ncurrent:]= col_tmp[:Ndata-Ncurrent]
			MG[Ncurrent:] =  MG_tmp[:Ndata-Ncurrent]
		else:
			col[Ncurrent:Ncurrent+Ngenerate]= col_tmp
			MG[Ncurrent:Ncurrent+Ngenerate] =  MG_tmp


		Ncurrent+=Ngenerate
		try:
			frac=(Nguess/Ngenerate)
			Nguess=int(frac*(Ndata-Ncurrent))
		except ZeroDivisionError:
			Nguess=100000

		if verbose: print('Ndata=%i    Ngenerate=%i  Ncurrent=%i Ntogenerate=%i'%(Ndata,Ngenerate,Ncurrent,(Ndata-Ncurrent)))

		count+=1

	if plotname is not  None:

		f = plt.figure(figsize=(12, 6))
		ax1 = plt.subplot(121)
		ax2 = plt.subplot(122)

		ploth2(col, MG, ax=ax1, inverty=True, bins=(100, 100), range=(collim, MGlim), gamma=0.4, cmap='gray_r',
		   interpolation='bilinear', norm='max', zero_as_blank=False)
		ploth1(MG, ax=ax2, bins=30, xlim=None, ylim=None, xlabel=None, ylabel=None, fontsize=14, cmap='gray_r',
		   invertx=False, inverty=False, title=None, norm=None, range=MGlim, label='', mode='step',
		   cumulative=False)

		ax1.set_xlabel('(BP-RP)', fontsize=label_size)
		ax1.set_ylabel('$M_G$', fontsize=label_size)
		ax2.set_xlabel('$M_G$', fontsize=label_size)
		ax2.set_ylabel('$N$', fontsize=label_size)
		f.savefig(plotname+'.png')

	return col, MG
