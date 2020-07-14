import numpy as np

# specify the priors we will use for this model; we use ln(P) = -0.5*chisq
# here, omitting the normalization term since it is the same for all models

# utilities

def chi_prior(x,mu,sg):
    return (x-mu)**2/sg**2

def chi_bound(x,lo,hi,sg):
    chitmp = 0.0
    if x<lo: chitmp = (x-lo)**2/sg**2
    if x>hi: chitmp = (x-hi)**2/sg**2
    return chitmp

# specifics for this cluster

def func(p):
    chisq = 0.0
    # basically fix los truncation radius
    a0 = p[2]
    chisq = chisq + chi_prior(a0,2.0,0.001)
    # bounds on member parameters
    bgal = p[0]
    agal = p[1]
    chisq = chisq + chi_bound(bgal,-1.0,1.5,0.001)
    chisq = chisq + chi_bound(agal, 0.0,2.5,0.001)
    # bounds on halo parameters
    phalo = np.array(p[3:-2]).reshape((-1,6))
    for ptmp in phalo:
        bhalo = ptmp[0]
        shalo = ptmp[5]
        chisq = chisq + chi_bound(bhalo,-1.0,2.0,0.001)
        chisq = chisq + chi_bound(shalo,-1.0,2.0,0.001)
    # Gaussian prior on ellipticity of halo3
    # chisq += chi_prior(phalo[2,0],-1.0,0.001)
    # chisq += chi_prior(phalo[2,1], 0.0, 0.01) + chi_prior(phalo[2,2], 0.0, 0.01)
    ec = phalo[2,3]
    es = phalo[2,4]
    sg = 0.1
    chisq = chisq + (ec**2+es**2)/sg**2
    # return lnP
    return -0.5*chisq

