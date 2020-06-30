import numpy as np
import systcore as core
import time

cluster = 'm0416'
catalog = cluster+'_master.csv'
if cluster == 'a2744':
    parmfile = cluster+'/dat/a2744-parms-2d-take2.dat'
if cluster == 'm0416':
    parmfile = cluster+'/dat/m0416-parms-2d.dat'
etab = 0.5; etaa = 0.5
for val in np.arange(30,31,30):
    rc = 'rc'+str(val)
    scale = 'scale1'
    imgdatfile = cluster+'/dat/images.dat'
    imgsig05datfile = cluster+'/dat/images-sig0.5.dat'
    memdatfile = cluster+'/dat/'+rc+'.dat'
    losdatfile = cluster+'/dat/los2.dat'
    memdeffile = cluster+'/def/'+scale+'-'+rc
    losdeffile = cluster+'/def/'+scale+'-los'
    halodatfile = cluster+'/dat/halo.dat'
    mcmcoutfile = cluster+'/fit/fit2-'+scale+'-'+rc

    # for mag map
    xmin = -60; xmax = 60
    steps = 121
    draws = 1000


    # create member datfile from catalogs
    # core.create_memdat(cluster, catalog, memdatfile, radialcut=val)

    # deflection analysis, we'll do just def1 I think
    #core.create_galdef(cluster, imgdatfile, etaa, etab, memdatfile, losdatfile, memdeffile, losdeffile)

    # Run MCMC
    core.run_mcmc(cluster, imgsig05datfile, halodatfile, memdeffile, losdeffile, cluster+'/fit/fit2-scale1-'+rc)
    # mcmctime = time.time()

    # # Get mags
    # # for mags, we need los variation
    # oldparms = np.loadtxt(parmfile)
    # losb = []
    # if cluster == 'a2744':
    #     irange = 23+6
    # if cluster == 'm0416':
    #     irange = 23+19
    # for i in range(23,irange):
    #     losb.append([np.mean(oldparms.T[i]),np.std(oldparms.T[i])])

    # # we also need to give it coordinates for the map
    # xarr = []
    # x = np.linspace(xmin,xmax,steps) # -60 to 60 arcsec, 121 steps
    # for i in x:
    #     for j in x:
    #         xarr.append([j,i])

    # magsall = core.get_mags(draws, etab, etaa, np.load(mcmcoutfile+'-mc-main.npy'), memdatfile, 'm0416/dat/los2.dat', losb, xarr)
    # np.save('{}-mags.npy'.format(mcmcoutfile),magsall)
