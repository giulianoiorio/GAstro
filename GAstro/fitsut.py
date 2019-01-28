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


	tabo.header['COMMENT']='Filtered the %s'%strftime("%Y-%m-%d %H:%M:%S", gmtime())

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
	
	lentable=len(tab.columns.names[0])
	idx= np.ones(lentable, dtype=np.bool)
	
	for orfilter in filters:
		
		for andfilter in orfilter:
			
			colname=andfilter
			condition=orfilter[colname]
			array=tab.data[colname]
			idtt=_check_condition_filter(array, condition)
			print(idtt)
			print(idx)
			input()
			idx*=idtt
			
		idx = idx | idx
		
	filteredtab = filter_idx(tab,index,cols)
		
	return filteredtab
			





