################################################
#
#
#
# Utils to manage fits
#
#
#
#
################################################
import numpy as np
import astropy.io.fits as ft

def filter_idx(tab,index,cols=()):
	"""
	Define a new fits table filtered with the index:
	@tab: a astropy fits table object
	@index: Boolean numpy array
	@cols: name of the cols to transfer to the new table, default all the columns.
	"""

	if len(cols)>=1: cols=cols
	else: cols=tab.columns.names


	#check dimension
	if len(index)!=len(tab.data[cols[0]]): raise ValueError

	i=0
	newcols=[]
	for name in cols:
		colformat=tab.columns.formats[i]
		col_tmp=ft.Column(name=name,array=tab.data[name][index],format=colformat)
		newcols.append(col_tmp)
		i+=1

	new_colsdef=ft.ColDefs(newcols)

	tabo=ft.BinTableHDU.from_columns(new_colsdef)


	return tabo
	
def _check_condition_filter(array, condition):
	
	if isinstance(condition, tuple) or isinstance(condition, list) or isinstance (condition, np.ndarray):
		if len(condition)==2:
			
			if isinstance(condition[0],str) or isinstance(condition[1],str):
				
				if isinstance(condition[0],str): cond, val = condition
				else: val, cond = condition
		
				if cond=='>':
					idx=array>val
				elif cond=='>=':
					idx=array>=val
				elif cond=='<':
					idx=array<val
				elif cond=='<=':
					idx=array<=val
				elif cond=='=':
					idx=array==val
				elif cond=='!=':
					idx=array!=val
				else:
					raise ValueError('Condition %s is invalid'%cond)
					
			else:
				minv=np.min(condition)
				maxv=np.max(condition)
				idx=(array>=minv)&(array<=maxv)
				
		else:
			raise ValueError('Conditions with tuple or list need to have exactly two elements')
			
	elif isinstance(condition, float) or isinstance(condition, int) or isinstance(condition, bool):
		
		idx=array==condition
		
	elif isinstance(condition, str):
		
		if condition in ('finite', 'finit', 'fini', 'fin', 'fi', 'f'):
			idx=np.isfinite(array)
		elif condition in ('notfinite', 'notfinit', 'notfini', 'notfin', 'notfi', 'notf', 'nfinite', 'nfinit', 'nfini', 'nfin', 'nfi', 'nf', ):
			idx=~np.isfinite(array)
		elif condition in ('exist','ex','e'):
			idx=np.isnan(condition)
		elif condition in ('notexist', 'notex', 'note', 'nexist', 'nex', 'ne'):
			idx=~np.isnan(condition)
		else:
			raise ValueError('Conditions with strings can be only finite/notfinite/exist/notexist')
			
	else:
		raise ValueError('Invalid condition')
	
	return idx
	
def filter(tab,filters=({},),cols=()):
	
	lentable=len(tab.data[tab.columns.names[0]])
	idx= np.ones(lentable, dtype=np.bool)
	
	for orfilter in filters:
		
		for andfilter in orfilter:
			
			colname=andfilter
			condition=orfilter[colname]
			array=tab.data[colname]
			idtt=_check_condition_filter(array, condition)
			idx= idx & idtt
			
		idx = idx | idx
		
	filteredtab = filter_idx(tab,idx,cols)
		
	return filteredtab


def make_fits(dict,outname=None,header_key={}):
	'''
	Make a fits table from a numpy array
	args must be dictionary containing the type  and the columnf of the table, e.g.
	{'l':(col1,'D'),'b':(col2,'D')}
	'''


	col=[]
	for field in dict:
		if len(dict[field])==2:
			format=dict[field][1]
			array=dict[field][0]
		else:
			format='D'
			array=dict[field]

		col.append(ft.Column(name=field,format=format,array=array))

	cols = ft.ColDefs(col)
	tab = ft.BinTableHDU.from_columns(cols)
	for key in header_key:
		item=header_key[key]
		if item is None: tab.header[key]=str(item)
		else: tab.header[key]=item


	if outname is not None: tab.writeto(outname,overwrite=True)

	return tab

def addcol_fits(fitsfile,newcols=({},),idtable=1,outname=None):
	"""
	fitsfile: name of fitsfile or table hdu
	newxols: a tuole of dics with keyword 'name', 'format' and 'array'
	idtable: the id of the table to modify
	outname: if not None the name of the outputted fits file
	"""

	if idtable is not None:

		try:
			orig_table = ft.open(fitsfile)[idtable].data
		except:
			orig_table = fitsfile[idtable].data

	else:

		try:
			orig_table = ft.open(fitsfile).data
		except:
			orig_table = fitsfile.data

	orig_cols = orig_table.columns

	col_list=[]
	for dic in newcols:
		coltmp=ft.Column(name=dic['name'], format=dic['format'], array=dic['array'])
		col_list.append(coltmp)
	new_cols=ft.ColDefs(col_list)
	hdu = ft.BinTableHDU.from_columns(orig_cols + new_cols)

	if outname is not None: hdu.writeto(outname,overwrite=True)

	return hdu



