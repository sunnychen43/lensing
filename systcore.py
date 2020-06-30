import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyOffsetFrame, ICRS
from astropy import stats
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import sys
import fitclus2d as myfit
import priors


fieldcenter = {'a2744' : SkyCoord('00h14m21.2s','-30d23m50.1s'),
               'm0416' : SkyCoord('4h16m8.9s','-24d4m28.7s')}
zclus = {'a2744' : 0.308,
         'm0416' : 0.396}
pgals = {'a2744' : [1.689791e-01, 1.965115e+00, 2.0],
        'm0416' : [3.737113e-01, 1.322081e+00, 2.0]}

def create_memdat(cluster, catalog, outfile,
        dzcut=0.03, sigclip=3, sigcut=3,
        radialcut=120, maglim=23.5, colorcut=1.0):
    """
    Create file with cluster member x, y, and magnitude in relation to BCG.

    Parameters
    ----------
    cluster : str
        'a2744' or 'm0416'
    catalog : csv
        Should already be cleaned
    outfile : str
    dzcut : float
        Cutoff around the cluster redshift to determine membership
    sigclip : int
        Level data should be sigma-clipped to determine membership
    sigcut : int
        How many sigma to cut membership based on redshift
    radialcut : float
        Radial limit for membership
    maglim : float
        Magnitude limit on f814w for membership
    colorcut : float
        How many sigma to cut membership based on color-magnitude relation
    """

    master = pd.read_csv(catalog)
    zdat = master['master_z']
    memberindx = np.where((zdat>zclus[cluster]-dzcut)&(zdat<zclus[cluster]+dzcut))[0]

    if sigclip != 0 :

        z1 = zdat[memberindx]
        z2 = stats.sigma_clip(z1,sigma=sigclip)
        dz = sigcut*np.std(z2)
        memberindx = np.where((zdat>zclus[cluster]-dz)&(zdat<zclus[cluster]+dz))[0]

    ddat = master['master_d']
    zdat = master['master_z']
    mag814 = master['master_mag814']
    mag606 = master['master_mag606']

    master_cut = master.iloc[np.where((ddat<radialcut)&(~np.isnan(mag606))&(~np.isnan(mag814))&(mag814<=maglim))]

    zdat   = master_cut['master_z']
    mag814 = master_cut['master_mag814']
    err814 = master_cut['master_err814']
    mag606 = master_cut['master_mag606']
    err606 = master_cut['master_err606']
    col_606_814 = mag606-mag814
    col_err = np.sqrt(err814**2+err606**2)

    fgindx = np.where(zdat<zclus[cluster]-dz)[0]
    bgindx = np.where(zdat>zclus[cluster]+dz)[0]
    memindx = np.where((zdat>zclus[cluster]-dz)&(zdat<zclus[cluster]+dz))[0]

    # fit a straight line to the color-magnitude relation
    x = mag814.iloc[memindx]
    y = col_606_814.iloc[memindx]
    redseq_fit = np.polyfit(x,y,1)
    redseq_fit_fn = np.poly1d(redseq_fit)

    cdiff = col_606_814 - redseq_fit_fn(mag814)

    # member galaxies
    cdiff0 = cdiff.iloc[memindx]
    color_mu = cdiff0.mean()#np.mean(cdiff0)
    color_sig = cdiff0.std()
    #print('mean,stdev:',color_mu,color_sig)

    photmemindx = np.where( (np.fabs(cdiff)<=colorcut*color_sig) & (np.isnan(master_cut['master_z'])) )[0]

    indx = np.concatenate((memindx,photmemindx))
    # we combine the spec member list with the added phot members
    members_tmp = master_cut.iloc[indx]
    # take only those that are flagged as valid
    members = members_tmp[members_tmp['master_valid']>0].sort_values(by='master_mag814')
    bcg814 = members.master_mag814.min()
    members['magdiff'] = members['master_mag814']-bcg814

    members.to_csv(outfile, index=False, sep=' ', columns=['master_x','master_y','magdiff'], header=False)

def create_galdef(cluster,imgfile,etaa,etab,memfile,losfile,memout,losout):
    """
    Run deflection calculation for galaxies, both cluster members & LOS

    Parameters
    ----------
    cluster : str
        'a2744' or 'm0416'
    imgfile : str
        file containing image info in lensmodel format
    etaa : float
        Controls slope of trunc. radius-luminosity relation: r_200 propto L^etaa
    etab : float
        Controls slope of mass-luminosity reltion: R_E propto L^etab
    memfile : str
        File with structure (x y flux) for each cluster member
    losfile : str
        File with structure (x y flux) for each LOS galaxy
    memout : str
    losout : str
    """


    zlens = zclus[cluster]
    # load the images
    imgdat = myfit.imgclass(imgfile,zlens,cosmo)
    # ultimately the fit will optimize the two normalization parameters;
    # since b is multiplicative, we can fix the normalization to unity here
    # and later rescale; we do need to run with different a values (see below)
    b0 = 0.0
    a0 = 2.0
    nran = 3000

    # members, referenced to BCG
    memdat = myfit.galclass(memfile,logflags=[True,True])
    memdat.scale(b0,etab,0.1,a0,etaa,0.03,ref=[0])

    # generate the deflection distributions
    aarr = np.linspace(1.0,2.5,31)
    memdef = myfit.defclass()
    memdef.draw(memdat,imgdat,b0,aarr,nran,basename=memout,useD=True)

    # los, previously los2, referenced to BCG in memdat; note that we use a=2.0 for all
    m0 = memdat.ref[1]
    losdat = myfit.galclass(losfile,logflags=[True,True])
    losdat.scale(b0,etab,0.3,a0,0.0,0.00,ref=[-1,m0])

    # generate the deflection distributions
    aarr = np.linspace(1.9,2.1,3)
    losdef = myfit.defclass()
    losdef.draw(losdat,imgdat,b0,aarr,nran,basename=losout,useD=True)

def run_mcmc(cluster,imgfile,halofile,memfile,losfile,mcmcout,nburn=10000,nstep=10000):
    """
    Run MCMC analysis

    Parameters
    ----------
    cluster : str
        'a2744' or 'm0416'
    imgfile : str
        file containing image info in lensmodel format using sigma=0.5
    halodat : str
        file containing halo info
    memfile : str
        File with structure (x y flux) for each cluster member
    losfile : str
        File with structure (x y flux) for each LOS galaxy
    memout : str
    losout : str
    """

    zlens = zclus[cluster]

    # load the images
    imgdat = myfit.imgclass(imgfile,zlens,cosmo)

    # halo
    halo = myfit.haloclass()
    halo.load(halofile,logflags=[True,True])

    # deflection distributions (simulated separately)
    memdef = myfit.defclass()
    memdef.load(memfile)
    losdef = myfit.defclass()
    losdef.load(losfile)

    # lensmodel setup; remember to include the specific priors
    lm = myfit.lensmodel(imgdat,halo,memdef,losdef)
    lm.setprior(priors.func)

    # initialize the fit
    fit = myfit.fitclass(lm.lnP)

    # set the parameters
    pgal = pgals[cluster]
    phalo = np.array(halo.p).flatten().tolist()
    pshr = halo.pshr
    pref = pgal + phalo + pshr
    print(pref)
    plabels = ['bgal', 'agal', 'alos',
     'b1', 'x1', 'y1', 'ec1', 'es1', 's1',
     'b2', 'x2', 'y2', 'ec2', 'es2', 's2',
     'b3', 'x3', 'y3', 'ec3', 'es3', 's3',
     'gc', 'gs']

    # check one set of parameters
    tmp = lm.lnP(pref)
    print(tmp)
    print('chisq:',-2.0*tmp['chisq'])

    # optimize
    best = fit.optimize(pref,restart=5)
    print(fit.best.message)
    print(fit.best.x)
    print(fit.best.fun)
    tmp = lm.lnP(fit.best.x)
    print(tmp)
    print('chisq:',-2.0*tmp['chisq'])

    # compute Fisher matrix
    fit.Fisher(step=0.01)
    print(fit.grad)
    print(fit.Fmat)
    print(fit.Finv)

    # run MCMC
    fit.MCset(nburn=nburn,nstep=nstep,basename=mcmcout+'-mc')
    fit.MCrun()

    # make plots
    fit.MCplot(mcmcout+'-mc.pdf',labels=plabels,fmt='.3f',truths=pref)
    #fit.plot_Fisher(outbase+'-fish.pdf',nsamp=1000,labels=plabels,truths=pref)

def get_params(mem,etab,etaa,los,losb,parms,scatter=True):
    IDarr = []; params = []

    IDarr.append('pj1')
    params.append([parms[0],mem[0][0],mem[0][1],0,0,0,parms[1]])
    for m in mem[1:]:
        if scatter:
            logb = parms[0]-etab*0.4*m[2] + np.random.normal(0,0.1,1)[0]
            loga = parms[1]-etaa*0.4*m[2] + np.random.normal(0,0.03,1)[0]
        else:
            logb = parms[0]-etab*0.4*m[2]
            loga = parms[1]-etaa*0.4*m[2]
        IDarr.append('pj1')
        params.append([logb,m[0],m[1],0,0,0,loga])
    for i, l in enumerate(los):
        IDarr.append('pj1')
        params.append([np.random.normal(losb[i][0],losb[i][1],1)[0],l[0],l[1],0,0,0,parms[2]])

    IDarr.append('iso');IDarr.append('iso');IDarr.append('iso');IDarr.append('shr')
    params.append([parms[3],parms[4],parms[5],parms[6],parms[7],parms[8]])
    params.append([parms[9],parms[10],parms[11],parms[12],parms[13],parms[14]])
    params.append([parms[15],parms[16],parms[17],parms[18],parms[19],parms[20]])
    params.append([parms[21],parms[22]])

    return IDarr, params

def get_mags(N,etab,etaa,chain,mems,loss,losb,xarr):

    mags = []
    mem=np.loadtxt(mems)
    los=np.loadtxt(loss)

    for d in np.random.permutation(np.arange(len(chain)))[:N]:
        IDs,params = get_params(mem,etab,etaa,los,losb,chain[d],scatter=True)
        mu = myfit.calcmag(IDs,params,np.array(xarr),logflags=[True,True])
        mags.append(mu)

    return mags
