################################################
#
#
#
# Utils to plot stuff
#
#
#
#
################################################
from __future__ import division
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.colors import LogNorm, PowerNorm
from scipy.stats import binned_statistic_2d as bd2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_ellipse(mean,covar,ax=None):

	v, w = linalg.eigh(covar)
	v = 2*np.sqrt(2)*np.sqrt(v)
	u = w[0] / linalg.norm(w[0])
	angle=np.arctan(u[1]/u[0])*(180./np.pi)
	ell = mpl.patches.Ellipse(mean, v[0], v[1], 180+angle)

def ploth2(x=[],y=[],z=None, statistic='mean', H=None,edges=None,ax=None,bins=100,weights=None, colorbar=True, colorbar_label='',linex=[],liney=[],func=[],xlim=None,ylim=None,xlabel=None,ylabel=None,fontsize=14,cmap='gray_r',gamma=1,invertx=False,inverty=False,interpolation='none',title=None,vmax=None,norm=None,range=None,vmin=None, vminmax_option='percentile',zero_as_blank=True,levels=None,xlogbin=False,ylogbin=False, aspect='equal'):
	"""

	:param x:
	:param y:
	:param z:
	:param statistic:
	:param H:
	:param edges:
	:param ax:
	:param bins:
	:param weights:
	:param colorbar:
	:param colorbar_label:
	:param linex:
	:param liney:
	:param func:
	:param xlim:
	:param ylim:
	:param xlabel:
	:param ylabel:
	:param fontsize:
	:param cmap:
	:param gamma:
	:param invertx:
	:param inverty:
	:param interpolation:
	:param title:
	:param vmax:
	:param norm:
	:param range:
	:param vmin:
	:param vminmax_option:
	:param zero_as_blank:
	:param levels:
	:param xlogbin:
	:param ylogbin:
	:param aspect:
	:return:
	"""

	if H is None:

		if range is None: range = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
		else: range = range

		if isinstance(bins,float) or isinstance(bins,int):
			bins_t=[bins,bins]
			samebin=True
		else:
			bins_t=[bins[0],bins[1]]
			samebin=False

		bins=[[0,],[0,]]



		if xlogbin:
			bins[0]=np.logspace(np.log10(range[0][0]),np.log10(range[0][1]),bins_t[0]+1)
		else:
			bins[0]=np.linspace(range[0][0], range[0][1],bins_t[0]+1)

		if ylogbin:
			bins[1]=np.logspace(np.log10(range[1][0]),np.log10(range[1][1]),bins_t[1]+1)
		else:
			bins[1]=np.linspace(range[1][0], range[1][1],bins_t[1]+1)




		if z is None:
			sample=np.vstack([x,y]).T
			H,edges=np.histogramdd(sample,bins,weights=weights,range=range)
			xedges,yedges=edges
			extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]


		elif len(z)==len(x):
			H, xedges, yedges,_=bd2(x, y, z, statistic=statistic, bins=bins, range=range, expand_binnumbers=False)
			extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

		else:
			raise ValueError('Z needs to be None or an array with the same length of z and y')
	else:
		H=H
		xedges,yedges=edges
		extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

	if norm is not None:
		if norm=='max':
			H=H/np.nanmax(H)
		elif norm=='tot':
			H=H/np.nansum(H)
		elif norm=='maxrows':
			Hm=np.nanmax(H,axis=0)
			H=H/Hm
		elif norm=='totrows':
			Hm=np.nansum(H,axis=0)
			H=H/Hm
		elif norm=='maxcols':
			Hm=np.nanmax(H,axis=1)
			H=(H.T/Hm).T
		elif norm=='totcols':
			Hm=np.nansum(H,axis=1)
			H=(H.T/Hm).T
		elif norm[:10]=='percentile':
			q=float(norm[10:12])
			Hm=np.nanpercentile(H,q=q)
			H=H/Hm
		else: raise ValueError('norm option %s not recognised (valide values: max, tot, maxcols, maxrows, totcols, totrows)'%str(norm))

	if zero_as_blank:
		H=np.where(H==0,np.nan,H)

	if ax is not None:

		if vminmax_option=='percentile':
			if vmax is None: vmaxM=np.nanpercentile(H,q=95)
			else: vmaxM=np.nanpercentile(H,q=vmax)
			if vmin is None: vminM=np.nanpercentile(H,q=5)
			else: vminM=np.nanpercentile(H,q=vmin)
		elif vminmax_option=='absolute':
			if vmax is None: vmaxM=np.nanmax(H)
			else: vmaxM=vmax
			if vmin is None: vminM=np.nanmin(H)
			else: vminM=vmin
		#X,Y=np.meshgrid(xedges,yedges)


		if gamma==0: im=ax.imshow(H.T,origin='low',extent=extent, aspect=aspect,cmap=cmap,norm=LogNorm(),interpolation=interpolation,vmax=vmaxM,vmin=vminM)
		else: im=ax.imshow(H.T,origin='low',extent=extent, aspect=aspect,cmap=cmap,norm=PowerNorm(gamma=gamma),interpolation=interpolation,vmax=vmaxM,vmin=vminM)
		#if gamma==0: im=ax.pcolormesh(X,Y,H.T, cmap=cmap,norm=LogNorm(),vmax=vmax,vmin=vmin)
		#else: im=ax.pcolormesh(X,Y,H.T, cmap=cmap,norm=PowerNorm(gamma=gamma),vmax=vmax,vmin=vmin)

		if len(linex)>0:
			for c in linex:
				ax.plot([c,c],[yedges[0],yedges[-1]],c='blue',lw=2,zorder=1000)

		if len(liney)>0:
			for c in liney:
				ax.plot([xedges[0],xedges[-1]],[c,c],c='blue',lw=2,zorder=1000)

		if len(func)>0:
			if xlim is not None: xf=np.linspace(xlim[0],xlim[1])
			else: xf=np.linspace(range[0][0],range[0][1])
			for f in func:
				ax.plot(xf,f(xf),c='blue',lw=2,zorder=1000)

		if levels is not None:
			ax.contour(H.T,origin='lower',extent=extent,levels=levels)


		if xlim is not None: ax.set_xlim(xlim)
		if ylim is not None: ax.set_ylim(ylim)

		if invertx: ax.invert_xaxis()
		if inverty: ax.invert_yaxis()

		if xlabel is not None: ax.set_xlabel(xlabel,fontsize=fontsize,labelpad=2)
		if ylabel is not None: ax.set_ylabel(ylabel,fontsize=fontsize,labelpad=2)
		if title is not None: ax.set_title(str(title),fontsize=fontsize)
	else:
		im=None

	if colorbar and (ax is not None):
		divider = make_axes_locatable(ax)
		cax     = divider.append_axes('right', size='5%', pad=0.05)
		cbar	= plt.colorbar(im, cax=cax)
		cbar.ax.set_ylabel(colorbar_label, fontsize=fontsize)

	edges=(xedges,yedges)
	return H,edges,im