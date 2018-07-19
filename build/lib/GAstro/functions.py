import numpy as np



def Bailey_posterior(r, parallax, parallax_error, Lsph, parallax_offset=0):

	rinv=1/r
	A=r/Lsph
	B=(parallax-parallax_offset-rinv)/parallax_error
	posterior=r*r*np.exp(-A-0.5*B*B)

	ret=np.where(r>0, posterior, 0)

	return ret


Lsph=2
omega=-0.3
somega=0.3
rr=np.linspace(0,20,1000)
p=Bailey_posterior(rr,omega, somega, Lsph)

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as us

f=us(rr,p,k=2,s=0)
pro=[]
for r in rr:
	pro.append(f.integral(0,r))
pro=np.array(pro)

plt.plot(rr,p)
plt.plot(rr,f(rr))
plt.plot(rr,pro)
plt.show()