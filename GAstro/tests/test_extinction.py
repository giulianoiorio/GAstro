from GAstro.gaia import Extinction
import numpy as np

ext=Extinction()

bp_rp=0.95851
ebv=0.453996
g=16.7241

print(ext.kg(bp_rp, ebv))
print(ext.kg_iterative(np.array([bp_rp,]), np.array([ebv,])))
kbr=ext.kbr_iterative(np.array([bp_rp,]), np.array([ebv,]))
print(kbr)
print(bp_rp-kbr*ebv)
print(ext.kg(bp_rp-kbr*ebv, ebv))


bp_rp = np.array([0.1,2, 5])
ebv = np.array([0.4,0.1,0.2])
ebv_err = 0.16*ebv


Nerror=5
Abr=ext.Abr_iterative_error_sample(bp_rp,ebv,ebv_error=ebv,Nerror=Nerror)
bp_rp_l=np.repeat(bp_rp,Nerror)
bp_rp_0_l=bp_rp_l-Abr
print(bp_rp_0_l)

bp_rp_0_l=bp_rp_0_l.reshape(-1, Nerror).T
print(bp_rp_0_l)

print(np.mean(bp_rp_0_l,axis=0))

print(Abr.reshape(-1, Nerror).T)
print(bp_rp_0_l.shape)

