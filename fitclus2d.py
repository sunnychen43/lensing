import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import dill as pickle
import emcee
import matplotlib.pyplot as plt
import corner
from numba import njit
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

testmode = False

######################################################################
# class to hold image data
######################################################################

class imgclass:

    ##################################################################
    # initialize with empty arrays
    ##################################################################
    zlens = -1.0
    nsrc = 0
    nimg = 0
    xarr = []  # positions
    Carr = []  # list of variances
    Cmat = []  # covariance matrix
    zsrc = []  # source redshifts
    Darr = []  # distance ratios
    Umat = []

    ##################################################################
    # read from a lensmodel data file and compute key quantities
    ##################################################################
    def __init__(self,filename,zlens,cosmo,verbose=True):
        # read the file
        f = open(filename,'rt')
        alllines = []
        for l in skip_comments(f):
            alllines.append(l)
        f.close()
        # initialize various variables
        self.zlens = zlens
        self.nsrc = int(alllines[5][0])
        self.nimg = 0
        self.xarr = []  # positions
        self.Carr = []  # variances
        self.zsrc = []  # source redshifts
        self.Darr = []  # distance ratios
        self.Umat = []
        # loop over sources
        iline = 6
        for isrc in range(self.nsrc):
            nimg = int(alllines[iline][0])
            zsrc = -1.0
            Dfac = 1.0
            if self.zlens>0.0 and len(alllines[iline])>=5:
                zsrc = float(alllines[iline][4])
                Dos  = cosmo.angular_diameter_distance(zsrc).value
                Dls  = cosmo.angular_diameter_distance_z1z2(self.zlens,zsrc).value
                Dfac = Dls/Dos
            self.nimg = self.nimg + nimg
            iline += 1
            for iimg in range(nimg):
                x = float(alllines[iline][0])
                y = float(alllines[iline][1])
                s = float(alllines[iline][3])
                iline += 1
                self.xarr.append([x,y])
                self.Carr.append(s*s)
                self.zsrc.append(zsrc)
                self.Darr.append(Dfac)
                Utmp0 = np.zeros(2*self.nsrc)
                Utmp1 = np.zeros(2*self.nsrc)
                Utmp0[2*isrc  ] = 1
                Utmp1[2*isrc+1] = 1
                self.Umat.append(Utmp0)
                self.Umat.append(Utmp1)
        self.xarr = np.array(self.xarr)
        self.Carr = np.array(self.Carr)
        self.zsrc = np.array(self.zsrc)
        self.Darr = np.array(self.Darr)
        self.Umat = np.array(self.Umat)
        # full covariance matrix
        self.Cmat = np.zeros((2*self.nimg,2*self.nimg))
        for iimg in range(self.nimg):
            self.Cmat[2*iimg  ,2*iimg  ] = self.Carr[iimg]
            self.Cmat[2*iimg+1,2*iimg+1] = self.Carr[iimg]
        # done
        if verbose: print('Read image data from file',filename)

    ##################################################################
    # recompute cosmic distance ratios
    ##################################################################
    def calcdist(self,cosmo):
        # if lens redshift is not set, all distance ratios are 1
        if self.zlens<=0.0:
            self.Darr[:] = 1.0
            return
        # loop over images
        for i in range(self.nimg):
            if self.zsrc[i]>0.0:
                Dos  = cosmo.angular_diameter_distance(self.zsrc[i]).value
                Dls  = cosmo.angular_diameter_distance_z1z2(self.zlens,self.zsrc[i]).value
                self.Darr[i] = Dls/Dos
            else:
                self.Darr[i] = 1.0



######################################################################
# class to hold halo data
######################################################################

class haloclass:

    ##################################################################
    # initialize with empty arrays
    ##################################################################
    nhalo = 0
    ID    = []     # list of ID's for the model components
    p     = []     # lens model parameters
    pshr  = []     # shear parameters
    logflags = []  # flags for whether (Rein,rscale) are logarithmic

    ##################################################################
    def __init__(self,ID=[],phalo=[],pshr=[],logflags=[False,False]):
        self.nhalo    = len(ID)
        self.ID       = ID
        self.logflags = logflags
        self.setp(phalo,pshr)

    ##################################################################
    def load(self,filename,logflags=[False,False],verbose=True):
        # initialize
        ID = []
        phalo = []
        # work through file
        f = open(filename,'rt')
        for l in skip_comments(f):
            if l[0]=='shr':
                pshr = [float(l[i]) for i in range(1,len(l))]
            else:
                ID.append(l[0])
                phalo.append([float(l[i]) for i in range(1,len(l))])
        f.close()
        # store what we got
        self.nhalo    = len(ID)
        self.ID       = ID
        self.logflags = logflags
        self.setp(phalo,pshr)
        if verbose: print('Read halo data from file',filename)

    ##################################################################
    def setp(self,phalo,pshr=[]):
        self.p = phalo
        if len(pshr)>0: self.pshr = pshr
        if len(phalo)!=self.nhalo:
            print('ERROR: number of halo components is incompatible')

    ##################################################################
    def calcdef(self,img):
        ID = self.ID
        p  = self.p
        if len(self.pshr)>0:
            ID = ID + ['shr']
            p  = p + [self.pshr]
        return calcdef(ID,p,img.xarr,logflags=self.logflags)



######################################################################
# class to hold galaxy data
######################################################################

class galclass:

    ##################################################################
    # initialize with empty arrays
    ##################################################################
    ngal = 0
    dat  = []      # list of [x,y,magnitude]
    ID   = []      # list of ID's for the model components
    p    = []      # lens model parameters
    logflags = []  # flags for whether (Rein,rscale) are logarithmic
    ref  = []      # reference [iref,m0,b0,a0]
    mod  = []      # model [etab,sigb,etaa,siga]

    ##################################################################
    # file should contain list of (x,y,magnitude)
    # note: all that matters is *relative* magnitude compared to the
    # reference value m0
    ##################################################################
    def __init__(self,filename,logflags=[False,False],verbose=True):
        self.dat      = np.loadtxt(filename)
        self.ngal     = len(self.dat)
        self.ID       = ['pj1' for i in range(self.ngal)]
        self.logflags = logflags
        if verbose: print('Read galaxy data from file',filename)

    ##################################################################
    # store the scaling relations
    ##################################################################
    def scale(self,b0=1.0,etab=0.5,sigb=0.1,a0=1.0,etaa=0.4,siga=0.03,ref=[0]):
        self.mod = [etab,sigb,etaa,siga]
        # set reference
        if len(ref)==1:
            # take a galaxy in this set to be the reference
            iref = ref[0]
            self.ref = [iref,self.dat[iref,2],b0,a0]
        else :
            # use m0 as the reference magnitude
            iref,m0 = ref
            self.ref = [iref,m0,b0,a0]
        # go ahead and draw one realization
        self.draw()

    ##################################################################
    # update the normalizations of the scaling relations
    ##################################################################
    def rescale(self,b0=None,a0=None):
        if b0!=None: self.ref[2] = b0
        if a0!=None: self.ref[3] = a0

    ##################################################################
    # impose the scaling relations (with scatter) and construct p arr
    ##################################################################
    def draw(self):
        iref,m0,b0,a0 = self.ref
        etab,sigb,etaa,siga = self.mod
        # initialize
        self.p = np.zeros((self.ngal,7))
        # positions
        self.p[:,1] = self.dat[:,0]
        self.p[:,2] = self.dat[:,1]
        # b and a, from scaling relations
        self.p[:,0] = b0 - etab*0.4*(self.dat[:,2]-m0)
        self.p[:,6] = a0 - etaa*0.4*(self.dat[:,2]-m0)
        # compute the scatter
        db = np.random.normal(scale=sigb,size=self.ngal)
        da = np.random.normal(scale=siga,size=self.ngal)
        # if the reference is one of these galaxies, turn off its scatter
        if iref>=0 and iref<self.ngal:
            db[iref] = 0.0
            da[iref] = 0.0
        # add the scatter
        self.p[:,0] = self.p[:,0] + db
        self.p[:,6] = self.p[:,6] + da

    ##################################################################
    def calcdef(self,img):
        return calcdef(self.ID,self.p,img.xarr,logflags=self.logflags)



######################################################################
# class to characterize deflection distribution
######################################################################
 
class defclass:

    ##################################################################
    # initialize with empty arrays
    ##################################################################
    nimg = 0
    aarr = []
    marr = []
    Garr = []
    Cmat = []

    ##################################################################
    def __init__(self):
        self.nimg = 0

    ##################################################################
    # set up test case where means are zero and covariance matrix for
    # each image has the structure [[b^2, b^2 a], [b^2 a, b^2]], so a
    # is correlation coefficient; have option for logarithmic variables
    ##################################################################
    def test(self,img,b0,aarr,logflags=[False,False],verbose=True):
        if verbose: print('Creating test deflection distribution')
        self.nimg = img.nimg
        self.aarr = aarr
        # initialize
        self.marr = np.zeros((len(aarr),self.nimg,2))
        self.Garr = np.zeros((len(aarr),self.nimg,2,2))
        self.Cmat = np.zeros((len(aarr),self.nimg*2,self.nimg*2))
        # loop over a values
        fb = b0
        if logflags[0]: fb = 10.0**b0
        for ia,a0 in enumerate(aarr):
            fa = a0
            if logflags[0]: fa = 10.0**a0
            # diagonal
            for i in range(self.nimg*2):
                self.Cmat[ia,i,i] = fb*fb
            # off-diagonal
            for i in range(self.nimg*2-1):
                self.Cmat[ia,i,i+1] = fb*fb*fa
                self.Cmat[ia,i+1,i] = fb*fb*fa

    ##################################################################
    # draw random samples
    ##################################################################
    def draw(self,gal,img,b0,aarr,nran,basename='',savep=False,useD=False,verbose=True):
        self.nimg = img.nimg
        self.aarr = aarr
        # initialize
        self.marr = np.zeros((len(aarr),self.nimg,2))
        self.Garr = np.zeros((len(aarr),self.nimg,2,2))
        self.Cmat = np.zeros((len(aarr),self.nimg*2,self.nimg*2))
        # loop over a values
        pall = []
        defall = []
        for ia,a0 in enumerate(aarr):
            if verbose: print('Generating deflection distribution: a =',a0)
            gal.rescale(b0=b0,a0=a0)
            # do many realizations
            parr = []
            defarr = []
            for iran in range(nran):
                if verbose and iran%500==0: print('Step',iran)
                # generate a new galaxy draw
                gal.draw()
                # compute the deflections
                deftmp,Gamtmp = gal.calcdef(img)
                # if desired, apply the Dls/Ds scalings
                if useD:
                    deftmp = deftmp*np.reshape(img.Darr,(self.nimg,1))
                    Gamtmp = Gamtmp*np.reshape(img.Darr,(self.nimg,1,1))
                # update the arrays
                self.marr[ia] = self.marr[ia] + deftmp
                self.Garr[ia] = self.Garr[ia] + Gamtmp
                parr.append(gal.p)
                defarr.append(deftmp)
            parr = np.array(parr)
            defarr = np.array(defarr)
            # add to full arrays
            pall.append(parr)
            defall.append(defarr)
            # compute covariance
            self.Cmat[ia] = np.cov(np.reshape(defarr,(nran,-1)),rowvar=False)
        pall = np.array(pall)
        defall = np.array(defall)
        # normalize the means
        self.marr = self.marr/nran
        self.Garr = self.Garr/nran
        # if desired, save arrays
        if len(basename)>0:
            if verbose: print('Writing to files with basename',basename)
            # save the deflections and parameters as npy files
            np.save('{}-def.npy'.format(basename),defall)
            if savep: np.save('{}-p.npy'.format(basename),pall)
            # save the matrices using pickle
            f = open(basename+'.pkl','wb')
            pickle.dump([self.aarr,self.marr,self.Cmat,self.Garr],f)
            f.close()

    ##################################################################
    # read arrays from a pickle file
    ##################################################################
    def load(self,basename,verbose=True):
        f = open(basename+'.pkl','rb')
        self.aarr,self.marr,self.Cmat,self.Garr = pickle.load(f)
        f.close()
        # number of images
        self.nimg = int(len(self.marr[0])/2)
        if verbose: print('Read deflection data from file',basename+'.pkl')

    ##################################################################
    # rescale arrays to new values of b and a
    ##################################################################
    def rescale(self,b,a):
        if a<self.aarr[0]:
            # a is out of bounds low
            mvec = self.marr[0]
            Cmat = self.Cmat[0]
            Garr = self.Garr[0]
            okay = False
        elif a>self.aarr[-1]:
            # a is out of bounds high
            mvec = self.marr[-1]
            Cmat = self.Cmat[-1]
            Garr = self.Garr[-1]
            okay = False
        else:
            # a is in bounds, so interpolate (linear)
            i = np.searchsorted(self.aarr,a)
            f = (a-self.aarr[i-1])/(self.aarr[i]-self.aarr[i-1])
            mvec = (1.0-f)*self.marr[i-1] + f*self.marr[i]
            Cmat = (1.0-f)*self.Cmat[i-1] + f*self.Cmat[i]
            Garr = (1.0-f)*self.Garr[i-1] + f*self.Garr[i]
            okay = True
        # done; include b scaling, which is logarithmic
        fb = 10.0**b
        return okay,mvec*fb,Cmat*fb*fb,Garr*fb



######################################################################
# class for lens modeling; the 1-d parameter list is as follows:
# p = [bgal, agal, alos, <halo params>, gc, gs]
######################################################################

class lensmodel:

    ##################################################################
    img    = None
    halo   = None
    galdef = None
    losdef = None
    p = []             # see above
    bgal = 1.0
    agal = 1.0
    alos = 1.0
    phalo = []
    pshr = []
    maxellip  = 0.95   # upper limit on ellipticity
    sigellip  = 0.0    # if >0, stdev for Gaussian prior on ellipticity
    priorfunc = None   # function for any extra priors; must return ln(P)

    ##################################################################
    def __init__(self,img,halo,galdef,losdef,p=[]):
        self.img    = img
        self.halo   = halo
        self.galdef = galdef
        self.losdef = losdef
        if len(p)>0: self.setp(p)

    ##################################################################
    def setp(self,p):
        self.p = p
        # extract the parameters
        self.bgal = p[0]
        self.agal = p[1]
        self.alos = p[2]
        self.pshr = p[-2:]
        self.phalo = np.array(p[3:-2]).reshape((self.halo.nhalo,-1)).tolist()
        self.halo.setp(self.phalo,self.pshr)

    ##################################################################
    def setprior(self,func=None,maxellip=0.95,sigellip=0.0):
        self.maxellip  = maxellip
        self.sigellip  = sigellip
        self.priorfunc = func

    ##################################################################
    def prior(self):
        # priors on ellipticity
        chiprior = 0.0
        for p in self.phalo:
            ec = p[3]
            es = p[4]
            ep = np.sqrt(ec*ec+es*es)
            if ep>self.maxellip:
                chiprior = np.inf
            if self.sigellip>0.0:
                chiprior = chiprior + (ep/self.sigellip)**2
        lnP = -0.5*chiprior
        # see if there are any other priors
        if self.priorfunc!=None:
            lnP = lnP + self.priorfunc(self.p)
        # done
        return lnP

    ##################################################################
    # here is the lens modeling core; take everything and compute chisq
    ##################################################################
    def lnP(self,p):

        # here is the dictionary structure we will fill with log-likelihood
        lnPdict = {'posterior':-np.inf, 'chisq':-np.inf, 'norm':-np.inf, 'prior':-np.inf, 'chipri':-np.inf}

        # update the parameters
        self.setp(p)

        # get galaxy quantities
        galOK,agal,Cgal,Ggal = self.galdef.rescale(self.bgal,self.agal)
        # get los quantities, if appropriate
        if self.losdef!=None:
            losOK,alos,Clos,Glos = self.losdef.rescale(self.bgal,self.alos)
        else:
            # there is no los component, so set arrays to 0
            losOK = True
            alos = 0.0*agal
            Clos = 0.0*Cgal
            Glos = 0.0*Ggal

        # check whether parameters are in bounds
        if galOK==False or losOK==False:
            return lnPdict

        # get halo contributions; remember to include D factors
        ahalo,Ghalo = self.halo.calcdef(self.img)
        ahalo = ahalo*np.reshape(self.img.Darr,(self.img.nimg,1))
        Ghalo = Ghalo*np.reshape(self.img.Darr,(self.img.nimg,1,1))

        # combine
        atot = ahalo + agal + alos
        Gtot = Ghalo + Ggal + Glos
        minv = Gam2minv(Gtot)
        # in test mode, minv is the identity
        if testmode: minv = np.identity(len(minv))

        # the Smarg matrix requires some care
        Ceff = dotABAT(minv,self.img.Cmat) + Cgal + Clos
        Seff = np.linalg.inv(Ceff)
        UTSU = dotABAT(self.img.Umat.T,Seff)
        UTSUinv = np.linalg.inv(UTSU)
        SU = np.dot(Seff,self.img.Umat)
        Smarg = Seff - dotABAT(SU,UTSUinv)

        # compute chisq and normalization factor
        darr = self.img.xarr - atot
        dtmp = darr.flatten()
        chisq = np.dot(dtmp,np.dot(Smarg,dtmp))
        s1,ldet1 = np.linalg.slogdet(UTSU)
        s2,ldet2 = np.linalg.slogdet(Seff)
        s3,ldet3 = np.linalg.slogdet(minv)
        lnPchi = -0.5*chisq
        lnPnorm = -0.5*(ldet1-ldet2) + ldet3

        # prior
        lnPpri = self.prior()

        # combined
        lnPtot = lnPchi + lnPnorm + lnPpri

        lnPdict['posterior'] = lnPtot
        lnPdict['chisq']     = lnPchi
        lnPdict['norm']      = lnPnorm
        lnPdict['prior']     = lnPpri
        lnPdict['chipri']    = lnPchi + lnPpri
        return lnPdict



######################################################################
# class for handling a fit: optimization, MCMC, Fisher matrix
# NOTE: the function lnP must return a log-likelihood in the form of
# a dictionary whose entries can be access with the 'mode' argument
######################################################################

class fitclass:

    ndim = 0
    # for optimize
    best = None
    # gradient and Hessian for Fisher matrix analysis
    grad = []
    Hmat = []
    Fmat = []
    Finv = []
    # for MCMC
    nwalk = 50
    nburn = 1000
    nstep = 10000
    savesteps = False
    basename = ''
    MCsave = False
    MCverb = True
    sampler = None

    ##################################################################
    def __init__(self,lnP,mode='posterior'):
        self.lnP = lnP
        self.mode = mode

    ##################################################################
    # optimize the model; for available methods, see:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    ##################################################################

    def optimize(self,p0,restart=5,method='Nelder-Mead',verbose=True):
        # for minimize we need a minus sign
        f = lambda x: -self.lnP(x)[self.mode]
        p = p0.copy()
        i = 0
        while i<restart:
            i += 1
            self.best = minimize(f,p,method=method)
            # save result to be starting point for next iteration
            p = self.best.x.copy()
            if verbose:
                print('Optimize step',i,'done' if self.best.success else 'not done')
                print(self.best.x)
                print(f(self.best.x))
            if self.best.success: break
        return self.best

    ##################################################################
    # compute Fisher matrix; requires numdifftools:
    # https://numdifftools.readthedocs.io/en/latest/
    # https://pypi.org/project/numdifftools/
    # https://readthedocs.org/projects/numdifftools/downloads/pdf/latest/
    ##################################################################

    def Fisher(self,p0=[],step=None,refine=1):
        f = lambda x: self.lnP(x)[self.mode]
        # if p0 is given, use it; if not, use result from optimization
        if len(p0)==0:
            p0 = self.best.x.copy()
        # compute the derivatives; note minus sign given definition of
        # Fisher matrix
        if step==None:
            self.grad = nd.Gradient(f)(p0)
            self.Hmat = nd.Hessian(f)(p0)
        else:
            self.grad = nd.Gradient(f,step)(p0)
            self.Hmat = nd.Hessian(f,step)(p0)
        # if desired, iterate by updating the step size based on
        # what would be required to achieve delta=1
        for i in range(refine):
            steparr = self.calcstep(1.0)
            self.grad = nd.Gradient(f,steparr)(p0)
            self.Hmat = nd.Hessian(f,steparr)(p0)
        # factor to go from Hessian of figure of merit to Fisher matrix
        if self.mode=='posterior':
            Ffac = -1.0
        elif self.mode=='chisq' or self.mode=='chipri':
            Ffac = 0.5
        else:
            print('ERROR: unknown mode in Fisher matrix routine')
            return
        self.Fmat = Ffac*self.Hmat
        # also save the inverse
        self.Finv = np.linalg.inv(self.Fmat)

    ##################################################################
    # calculate how far to step in each dimension to achieve a certain
    # delta in the figure of merit
    ##################################################################

    def calcstep(self,delta):
        # we need gradient and Hessian
        if len(self.grad)==0:
            self.Fisher()
        d1arr = self.grad
        d2arr = -np.diag(self.Hmat)
        # figure out how far to go in each dimension
        steparr = []
        for i in range(len(self.best.x)):
            d1 = d1arr[i]
            d2 = d2arr[i]
            tmp1 = d1**2 + 2.0*d2*delta
            tmp2 = d1**2 - 2.0*d2*delta
            tmpx = []
            if tmp1>0.0:
                tmpx.append((-d1-np.sqrt(tmp1))/d2)
                tmpx.append((-d1+np.sqrt(tmp1))/d2)
            if tmp2>0.0:
                tmpx.append((-d1-np.sqrt(tmp2))/d2)
                tmpx.append((-d1+np.sqrt(tmp2))/d2)
            dx = np.amax(np.absolute(np.array(tmpx)))
            steparr.append(dx)
        # done
        return np.array(steparr)

    ##################################################################
    # corner plot showing slices of lnP
    ##################################################################

    def plot_slice(self,filename,delta=10.0,nstep=40,cmap='bone',labels=[],fontsize=10,verbose=False):

        p0 = self.best.x.copy()
        ndim = len(p0)
        self.ndim = ndim

        # estimate step size needed to achieve the desired delta,
        # and use that to set the parameter ranges we will sample
        steparr = self.calcstep(delta)
        xrange = []
        for i in range(ndim):
            xtmp = np.linspace(p0[i]-steparr[i],p0[i]+steparr[i],nstep)
            xrange.append(xtmp)
        xrange = np.array(xrange)

        f,ax = plt.subplots(ndim,ndim,figsize=(4*ndim,4*ndim))
        for i in range(ndim):
            # diagonal
            if verbose: print('plot_slice diagonal',i)
            xarr = xrange[i]
            farr = np.zeros(nstep)
            for k in range(nstep):
                ptmp = p0.copy()
                ptmp[i] = xarr[k]
                farr[k] = self.lnP(ptmp)[self.mode]
            ax[i][i].plot(xarr,farr)
            ax[i][i].axvline(x=p0[i],linestyle='dotted',color='black')
            if len(labels)>0:
                ax[i][i].set_xlabel(labels[i],fontsize=fontsize)
            ax[i][i].tick_params(labelsize=fontsize)
            # upper triangle: empty
            for j in range(i): ax[j][i].axis('off')
            # lower triangle
            for j in range(i+1,ndim):
                if verbose: print('plot_slice lower triangle',i,j)
                xarr = xrange[i]
                yarr = xrange[j]
                farr = np.zeros((nstep,nstep))
                for kx in range(nstep):
                    for ky in range(nstep):
                        ptmp = p0.copy()
                        ptmp[i] = xarr[kx]
                        ptmp[j] = yarr[ky]
                        # note index order
                        farr[ky,kx] = self.lnP(ptmp)[self.mode]
                ax[j][i].imshow(farr,origin='lower',interpolation='nearest',aspect='auto',cmap=cmap,extent=(xarr[0],xarr[-1],yarr[0],yarr[-1]))
                ax[j][i].axvline(x=p0[i],linestyle='dotted',color='black')
                ax[j][i].axhline(y=p0[j],linestyle='dotted',color='black')
                if len(labels)>0:
                    ax[j][i].set_xlabel(labels[i],fontsize=fontsize)
                    ax[j][i].set_ylabel(labels[j],fontsize=fontsize)
                ax[j][i].tick_params(labelsize=fontsize)
    
        f.savefig(filename,bbox_inches='tight')

    ##################################################################
    # functions to wrap emcee
    ##################################################################

    ##################################################################
    def MCset(self,nwalk=50,nburn=1000,nstep=10000,savesteps=False,basename='',verbose=True):
        self.nwalk = nwalk
        self.nburn = nburn
        self.nstep = nstep
        self.savesteps = savesteps
        self.basename = basename
        self.MCsave = (len(basename)>0)
        if self.MCsave==False: self.savesteps = False
        self.MCverb = verbose
        if self.MCverb:
            print('MCMC parameters:')
            print('nwalk',self.nwalk)
            print('nburn',self.nburn)
            print('nstep',self.nstep)
            if self.MCsave: print('saving to',basename)

    ##################################################################
    def MCrun(self,p0=[]):

        #self.nwalk,ndim,f = pickle.load(open("f.pkl", "rb"))
        f = lambda x: self.lnP(x)[self.mode]

        #if p0 is given, use it; if not, use result from optimization
        if len(p0)==0:
            p0 = self.best.x.copy()
        ndim = len(p0)
        self.ndim = ndim

        # set up walkers; initial positions are slightly perturbed from p0 as
        # recommended by Dan Foreman-Mackey: http://dan.iel.fm/emcee/current/user/line/)
        pstart = []
        for i in range(self.nwalk):
            p = p0 + 1.0e-4*np.random.randn(ndim)
            pstart.append(p)
        if self.MCsave: np.savetxt(self.basename+'-init.txt',np.array(pstart))

        # initialize the sampler
        self.sampler = emcee.EnsembleSampler(self.nwalk,ndim,f)

        # run some steps as a burn-in
        if self.MCverb: print('emcee: burn-in run')
        result = self.sampler.run_mcmc(pstart, self.nburn, progress=True)

        if self.MCsave:
            np.save(self.basename+'-burn',self.sampler.chain.reshape((-1,ndim)))
            np.save(self.basename+'-burn-chi',-2.0*self.sampler.lnprobability.reshape((-1)))
        #
        pos,prob,state = result

        # reset the chain to remove the burn-in samples
        self.sampler.reset()

        # start from end of burn-in and sample many more steps
        if self.MCverb: print('emcee: main run')
        self.sampler.run_mcmc(pos, self.nstep, rstate0=state, progress=True)

        if self.MCsave:
            np.save(self.basename+'-main',self.sampler.chain.reshape((-1,ndim)))
            np.save(self.basename+'-main-chi',-2.0*self.sampler.lnprobability.reshape((-1)))

        if self.MCverb: print('emcee: done')

    ##################################################################
    def MCplot(self,filename,labels=None,fmt='.2f',truths=None):
        if self.sampler==None:
            print('Error: you must run MCMC before plotting')
            return
        if self.MCverb: print('emcee: corner plot',filename)
        fig = corner.corner(self.sampler.chain.reshape((-1,self.ndim)),quantiles=[0.16,0.50,0.84],show_titles=True,title_fmt=fmt,labels=labels,truths=truths)
        fig.savefig(filename)

    ##################################################################
    def plot_Fisher(self,filename,color='blue',alpha=0.5,truths=[],linecolor='gray',nsamp=0,labels=[],fontsize=10):

        # check that we have what we need
        if self.best==None or len(self.Finv)==0:
            print('ERROR: must run optimize and Fisher before plotting')
            return

        # if truths is given, used that for the lines; if not, use best point
        if len(truths)>0:
            p0 = truths.copy()
        else:
            p0 = self.best.x.copy()

        # for drawing ellipses
        t = np.linspace(0.0,2.0*np.pi,200)
        xprime = np.array([np.cos(t),np.sin(t)])

        # if desired, subsample from the MCMC results for plotting here
        if nsamp>0:
            dat = self.sampler.chain.reshape((-1,self.ndim))
            if nsamp<len(dat):
                indx = np.random.choice(len(dat),nsamp,replace=False)
                samp = dat[indx]
            else:
                samp = dat

        nn = self.ndim-1
        f,ax = plt.subplots(nn,nn,figsize=(4*nn,4*nn))
        for i in range(nn):
            for j in range(1,i+1): ax[j-1][i].axis('off')
            for j in range(i+1,nn+1):
                tmpcov = self.Finv[i:(j+1):(j-i),i:(j+1):(j-i)]
                eigval,eigvec = np.linalg.eig(tmpcov)
                if np.any(eigval<0.0):
                    print('plot_Fisher: encountered a negative eigenvalue')
                    print('indices',i,j)
                    print('matrix',tmpcov)
                    print('eigval',eigval)
                else:
                    tmpmat = np.sqrt(eigval)*eigvec
                    x = self.best.x[i:(j+1):(j-i),np.newaxis] + np.dot(tmpmat,xprime)
                    ax[j-1][i].fill(x[0],x[1],color=color,alpha=alpha)
                ax[j-1][i].axvline(x=p0[i],linestyle='dotted',color=linecolor)
                ax[j-1][i].axhline(y=p0[j],linestyle='dotted',color=linecolor)
                if nsamp>0:
                    ax[j-1][i].plot(samp[:,i],samp[:,j],',',color='black')
                if len(labels)>0:
                    ax[j-1][i].set_xlabel(labels[i],fontsize=fontsize)
                    ax[j-1][i].set_ylabel(labels[j],fontsize=fontsize)
                ax[j-1][i].tick_params(labelsize=fontsize)
        f.tight_layout()
        f.savefig(filename,bbox_inches='tight')
        print('Fisher plot:',filename)



######################################################################
# routines for lensing calculations
######################################################################

######################################################################
# softened isothermal elliptical mass distribution
######################################################################
@njit
def isoemd(params,xarr):

    b  = params[0]
    x0 = params[1]
    y0 = params[2]
    ec = params[3]
    es = params[4]
    s  = params[5]

    if abs(s)<1.0e-8: s = 1.0e-4

    ep = np.sqrt(ec*ec+es*es)
    te = 0.5*np.arctan2(es,ec)
    sfac = np.sin(te)
    cfac = np.cos(te)

    # transform to ellipse coordinate system
    dx = xarr[:,0]-x0
    dy = xarr[:,1]-y0
    x  = -sfac*dx+cfac*dy
    y  = -cfac*dx-sfac*dy
    nimg = len(xarr)

    # if (linear) b is sufficiently small, everything is 0
    if abs(b)<1.0e-8:
        defarr = np.zeros((nimg,2))
        Gamarr = np.zeros((nimg,2,2))
        return defarr,Gamarr

    # utility quantities
    q   = 1.0-ep;
    q2  = q*q;
    om  = 1.0-q2;
    rt  = np.sqrt(abs(om));
    # this corresponds to alphanorm=1, which is default in lensmodel
    b   = b*np.sqrt(0.5*(1.0+q2))/q
    s   = s*np.sqrt(0.5*(1.0+q2))/q
    x2  = x*x;
    y2  = y*y;
    s2  = s*s;
    psi = np.sqrt(q2*(s2+x2)+y2);

    # deflection
    if rt<1.0e-6:
        # for q near 1, use series in sqrt(1-q^2) with errors O[1-q^2]^2
        psi  = np.sqrt(s2+x2+y2)
        tmp  = psi+s
        tmp  = 6.0*psi*tmp*tmp*tmp
        tmpx = x*(x2*(psi+3.0*s)+3.0*s2*(psi+s))
        t1   = -s2*(psi+3.0*s)
        t2   = 3.0*psi*(psi+s)*(psi+2.0*s)
        tmpy = s*(t1+t2)
        phix = b*q*(x/(psi+s)+tmpx*om/tmp)
        phiy = b*q*(y/(psi+s)+tmpy*om/tmp)
    elif om>0.0:
        phix = b*q/rt*np.arctan (rt*x/(psi+s   ))
        phiy = b*q/rt*np.arctanh(rt*y/(psi+s*q2))
    else:
        phix = b*q/rt*np.arctanh(rt*x/(psi+s   ))
        phiy = b*q/rt*np.arctan (rt*y/(psi+s*q2))

    # magnification
    den   = psi*(om*x2+(psi+s)*(psi+s))
    phixx =  b*q*(psi*(psi+s)-q2*x2)/den
    phiyy =  b*q*(x2+s*(psi+s))/den
    phixy = -b*q*x*y/den

    # rotate back to original coordinate system
    defx  = -sfac*phix-cfac*phiy
    defy  =  cfac*phix-sfac*phiy
    defxx =  sfac*sfac*phixx+cfac*cfac*phiyy+2.0*cfac*sfac*phixy
    defyy =  cfac*cfac*phixx+sfac*sfac*phiyy-2.0*cfac*sfac*phixy
    defxy =  cfac*sfac*(phiyy-phixx)-(cfac*cfac-sfac*sfac)*phixy

    defarr = np.column_stack((defx,defy))
    Gamarr = np.column_stack((defxx,defxy,defxy,defyy)).reshape((nimg,2,2))

    return defarr,Gamarr

def calcdef_iso(p,xarr,logflags):
    if logflags[0]: p[0] = 10.0**p[0]    # b
    if logflags[1]: p[5] = 10.0**p[5]    # s
    return isoemd(p,xarr)

######################################################################
# pseudo-Jaffe model can be written as difference of isoemd's
######################################################################

def pjaffe(params,xarr):
    p1 = params.copy()
    p2 = params.copy()
    p2[5] = p1[6]
    def1,Gam1 = isoemd(p1,xarr)
    def2,Gam2 = isoemd(p2,xarr)
    return def1-def2,Gam1-Gam2

def calcdef_pj1(params,xarr,logflags):
    # pjaffe that is singular and truncated
    p = params.copy()
    if logflags[0]: p[0] = 10.0**p[0]    # b
    if logflags[1]: p[6] = 10.0**p[6]    # a
    return pjaffe(p,xarr)

def calcdef_pj2(params,xarr,logflags):
    # pjaffe that is cored and truncated
    p = params.copy()
    if logflags[0]: p[0] = 10.0**p[0]    # b
    if logflags[1]: p[5] = 10.0**p[5]    # s
    if logflags[1]: p[6] = 10.0**p[6]    # a
    return pjaffe(p,xarr)

######################################################################
# shear
######################################################################
@njit
def calcdef_shr(p,xarr,logflags=[False,False]):
    # shear components (Cartesian)
    gc = p[0]
    gs = p[1]
    # origin (optional; 0 by default)
    x0 = 0.0
    y0 = 0.0
    if len(p)>4:
        x0 = p[2]
        y0 = p[3]
    # offset
    x = xarr[:,0]-x0
    y = xarr[:,1]-y0
    # compute
    defx = gc*x + gs*y
    defy = gs*x - gc*y
    deftmp = np.column_stack((defx,defy))
    Gamtmp = np.zeros((len(x),2,2))
    Gamtmp[:,0,0] = gc
    Gamtmp[:,0,1] = gs
    Gamtmp[:,1,0] = gs
    Gamtmp[:,1,1] = -gc
    return deftmp,Gamtmp

######################################################################
# compute total deflection and Gamma
######################################################################

calcdef_dict = {
  'iso' : calcdef_iso,
  'pj1' : calcdef_pj1,
  'pj2' : calcdef_pj2,
  'shr' : calcdef_shr
}

def calcdef(IDarr,parr,xarr,logflags=[False,False]):
    nimg = len(xarr)
    defarr = np.zeros((nimg,2))
    Gamarr = np.zeros((nimg,2,2))
    for i,ID in enumerate(IDarr):
        ptmp = parr[i].copy()
        deftmp,Gamtmp = calcdef_dict[ID](ptmp,xarr,logflags)
        defarr = defarr + deftmp
        Gamarr = Gamarr + Gamtmp
    return defarr,Gamarr

def calcmag(IDarr,parr,xarr,logflags=[False,False],signed=False):
    defarr,Gamarr = calcdef(IDarr,parr,xarr,logflags=logflags)
    marr = 1.0/((1.0-Gamarr[:,0,0])*(1.0-Gamarr[:,1,1])-Gamarr[:,0,1]*Gamarr[:,1,0])
    if signed==False: marr = np.abs(marr)
    return marr


######################################################################
# utility
######################################################################

######################################################################
# read a file and skip lines that are blank or start with comment '#';
# this is modified from
# https://stackoverflow.com/questions/1706198/python-how-to-ignore-comment-lines-when-reading-in-a-file
######################################################################

def skip_comments(file):
    for line in file:
        if not line.strip().startswith('#'):
            if len(line.split())>0:
                yield line.split()

######################################################################
# take a list of Gamma matrices and construct the big minverse matrix
######################################################################
@njit
def Gam2minv(Garr):
    nimg = len(Garr)
    Gmat = np.zeros((2*nimg,2*nimg))
    for iimg in range(nimg):
        for i in range(2):
            for j in range(2):
                Gmat[2*iimg+i,2*iimg+j] = Garr[iimg,i,j]
    minv = np.identity(2*nimg) - Gmat
    return minv

######################################################################
# the matrix product A.B.AT comes up several times
######################################################################
@njit
def dotABAT(A,B):
    return np.dot(A,np.dot(B,A.T))

