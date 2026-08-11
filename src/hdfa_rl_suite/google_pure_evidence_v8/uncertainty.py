from __future__ import annotations
import numpy as np

def bootstrap_interval(values:list[float],*,seed:int=15101,draws:int=4000)->list[float]|None:
    x=np.asarray(values,dtype=float)
    if len(x)<2:return None
    rng=np.random.default_rng(seed);stats=np.asarray([np.median(x[rng.integers(0,len(x),len(x))]) for _ in range(draws)])
    return [float(np.quantile(stats,.025)),float(np.quantile(stats,.975))]

def wilson_interval(successes:int,trials:int,z:float=1.959963984540054)->list[float]:
    if trials<=0:raise ValueError("positive trial count required")
    p=successes/trials;den=1+z*z/trials;centre=(p+z*z/(2*trials))/den;half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return [float(centre-half),float(centre+half)]

