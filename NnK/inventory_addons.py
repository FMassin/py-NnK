# -*- coding: utf-8 -*-
"""
source - Addon module for obspy inventory.

This module provides additionnal functionnalities for station inventory.
______________________________________________________________________

.. note::

Functions and classes are ordered from general to specific.

"""
import re
import numpy
import matplotlib
import obspy
from obspy.taup import TauPyModel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import obspy_addons



def traveltimes(tmax=20.,
                depth=10,
                model='iasp91',
                phase_list=['ttall']):
    # the first arrival at a range of distances

    model = TauPyModel(model=model)
    
    for d in numpy.linspace(.002,tmax*8./110.,128):
        arrivals = model.get_travel_times(depth, distance_in_degree=d, phase_list=['p'], receiver_depth_in_km=0.0)
        if arrivals[0].time>tmax:
            dmax = d
            tmax = arrivals[0].time
            break

    arrivals = model.get_travel_times(depth, distance_in_degree=.001, phase_list=phase_list, receiver_depth_in_km=0.0)
    while len(arrivals)>1:
        for i,a in enumerate(arrivals):
            if i>0:
                arrivals.remove(a)
    
    for d in numpy.linspace(.002,dmax,16):
        test=model.get_travel_times(depth, distance_in_degree=d, phase_list=phase_list, receiver_depth_in_km=0.0)
        for a in test:
            if a.time>0.:
                arrivals.append(a)
                break
    return arrivals,model,dmax,tmax



def traveltimesgrid(longitudes, latitudes,
                    distances,
                    tmax=20.,
                    depth=10,
                    model='iasp91',
                    N=[6],
                    style='p',
                    dt=None):


    arrivals,taupmodel,dmax,tmax = traveltimes(tmax=tmax,
                                               depth=depth,
                                               model=model)
        
    darrivals = numpy.asarray([a.distance for a in arrivals])
    tarrivals = numpy.asarray([a.time for a in arrivals])

    if style[0] in ['d','b']:
        Sarrivals,taupmodel,dmax,tmax = traveltimes(tmax=tmax,
                                               depth=depth,
                                               model=model,
                                               phase_list=['s'])
        
        dSarrivals = numpy.asarray([a.distance for a in Sarrivals])
        tSarrivals = numpy.asarray([a.time for a in Sarrivals])

    avsgrids = numpy.zeros([len(N), longitudes.shape[0],longitudes.shape[1]])
    avpgrids = numpy.zeros([len(N), longitudes.shape[0],longitudes.shape[1]])
    dgrids = numpy.zeros([len(N), longitudes.shape[0],longitudes.shape[1]])
    tgrids = numpy.zeros([len(N), longitudes.shape[0],longitudes.shape[1]])
    if dt is None:
        dt = numpy.zeros([len(N), longitudes.shape[0],longitudes.shape[1]])
    
    for col, lon in enumerate(longitudes[0,:]):
        for line, lat in enumerate(latitudes[:,0]):
            for i,n in enumerate(N):
                
                closest = 1./(distances[n-1,line,col] - darrivals)**2.
                closest = closest/numpy.nansum(closest)
                avp = (numpy.nansum(darrivals*closest/tarrivals))
                
                avpgrids[i,line,col] = avp
                
                tgrids[i,line,col] = distances[n-1,line,col]/avp+ dt[i,line,col]
                dgrids[i,line,col] = distances[n-1,line,col]

                if style[0] in ['d','b']:
                    closest = 1./(distances[n-1,line,col] - dSarrivals)**2.
                    closest = closest/numpy.nansum(closest)
                    avs = (numpy.nansum(dSarrivals*closest/tSarrivals))
                    
                    avsgrids[i,line,col] = avs
                    dgrids[i,line,col] = avs*(distances[n-1,line,col]/avp+dt[i,line,col])
                    tgrids[i,line,col] = distances[n-1,line,col]/avs - (distances[n-1,line,col]/avp+dt[i,line,col])

    return tgrids,dgrids,dmax,tmax,avpgrids,avsgrids


def plot_traveltimes(self,
                     tmax=20.,
                     depth=20,
                     model='iasp91',
                     N=range(1,7),
                     fig=None,
                     style='p',
                     bits=1024,
                     reflevel=5.,
                     latencies=None):
    
    statlons, statlats = obspy_addons.search(self, fields=['longitude','latitude'], levels=['networks','stations'])
    names = [re.sub(r' .*', '',x) for x in self.get_contents()['stations']]
    

    # generate 2 2d grids for the x & y bounds
    if style[0] in ['d','b']:
        tmax*=1.7
    dmax=tmax*8./110.
    dlat = ((numpy.nanmax(statlats)+dmax)-(numpy.nanmin(statlats)-dmax))/numpy.sqrt(bits)
    dlon = ((numpy.nanmax(statlons)+dmax)-(numpy.nanmin(statlons)-dmax))/numpy.sqrt(bits)

    latitudes, longitudes = numpy.mgrid[slice(numpy.nanmin(statlats)-(dmax),
                                              numpy.nanmax(statlats)+(dmax),
                                              dlat),
                                        slice(numpy.nanmin(statlons)-(dmax),
                                              numpy.nanmax(statlons)+(dmax),
                                              dlon)]

    dsgrid = numpy.zeros([len(statlons),latitudes.shape[0],latitudes.shape[1]])
    for col, lon in enumerate(longitudes[0,:]):
        for line, lat in enumerate(latitudes[:,0]):
            for i,s in enumerate(statlons):
                dsgrid[i,line,col] = numpy.sqrt((statlons[i]-lon)**2 + (statlats[i]-lat)**2.)

    lsgrid = numpy.zeros([len(statlons),latitudes.shape[0],latitudes.shape[1]])
    if  latencies is not None:
        latencies = [ numpy.nanmedian([ l for l in latencies[n] if l <120.]) for n in names]
        for col, lon in enumerate(longitudes[0,:]):
            for line, lat in enumerate(latitudes[:,0]):
                for i,s in enumerate(statlons):
                    closest = dsgrid[:,line,col].copy()
                    closest[closest>numpy.nanmin(closest)]=0.
                    closest[closest==numpy.nanmax(closest)]=1.
                    lsgrid[i,line,col] = numpy.nansum(latencies*closest)

        lsgrid[lsgrid == numpy.nan] =0

    dsgrid = numpy.sort(dsgrid,axis=0)
    dsgrid = dsgrid[:max(N),:,:]
    lsgrid = lsgrid[:max(N),:,:]

    tgrids,dgrids,dmax,tmax,avpgrids,avsgrids = traveltimesgrid(longitudes, latitudes,
                                                                dsgrid,
                                                                tmax=tmax,
                                                                depth=depth,
                                                                model=model,
                                                                N=N,
                                                                style=style,
                                                                dt=lsgrid)
    data=tgrids
    label=str(N[-1])+'$^{th}$ P travel time (s'
    ax2xlabel='P travel time (s'
    if style in ['d']:
        ax2xlabel='S delays (s'
        data=tgrids+lsgrid#dgrids*110.
        label='S delay after '+str(N[-1])+'$^{th}$ P travel time (s'#S radius at '+str(N[-1])+'$^{th}$ P travel time [km]'
    elif style in ['l']:
        data=lsgrid
        ax2xlabel='station delays (s'
        label=str(N[-1])+'$^{th}$ station delays (s'
    elif style in ['b']:
        ax2xlabel='S radius (km'
        kmindeg = obspy_addons.haversine(numpy.nanmin(longitudes),
                                         numpy.nanmin(latitudes),
                                         numpy.nanmax(longitudes),
                                         numpy.nanmax(latitudes))
        kmindeg /= 1000*numpy.sqrt((numpy.nanmin(longitudes)-numpy.nanmax(longitudes))**2+(numpy.nanmin(latitudes)-numpy.nanmax(latitudes))**2)
        tmax=dmax*kmindeg
        reflevel*= numpy.nanmedian(avsgrids)*kmindeg
        data=dgrids*kmindeg
        label='S radius at '+str(N[-1])+'$^{th}$ P travel time (km'#S radius at '+str(N[-1])+'$^{th}$ P travel time [km]'


    if  latencies is not None:
        ax2xlabel+= '$_{\ with\ station\ delays}$)'
        label+= '$_{\ with\ station\ delays}$)'
    else:
        ax2xlabel+= ')'
        label+= ')'

    if not fig:
        fig = self.plot(size=0,projection='local')
        axcb = fig.bmap.ax
    else:
        fig = self.plot(fig=fig, size=0,projection='local')
        axcb = fig.bmap.ax #fig.axes[1]

    ax = fig.bmap

    levels = numpy.asarray([0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25,50,100,250,500,1000])
    level = levels[numpy.nanargmin(abs((min([tmax,numpy.nanmax(data[-1])]) - numpy.nanmin(data[-1]))/10 - levels))]
    levels = numpy.arange(numpy.nanmin(data[-1]), min([tmax,numpy.nanmax(data[-1])]), level)
    levels -= numpy.nanmin(abs(levels))

    cf= ax.contourf(x=longitudes,
                    y=latitudes,
                    data=data[-1],
                    latlon=True,corner_mask=True,
                    zorder=999,
                    alpha=1/2.,linewidths=0.,
                    levels=levels)
    CS = ax.contour(x=longitudes,
               y=latitudes,
               data=data[-1],
               latlon=True,corner_mask=True,
               zorder=999,
               levels=[numpy.around(reflevel)])
    matplotlib.pyplot.clabel(CS,
                             fmt='%1.0f',
                             inline=1,
                             fontsize=10)

    if axcb:
        fig.cb = matplotlib.pyplot.colorbar(cf,
                                        ax=axcb,
                                        label=label,
                                        location='top',
                                        anchor= (axcb.get_position().bounds[0]+axcb.get_position().bounds[2]/1.,
                                                 axcb.get_position().bounds[1])  ,
                                        fraction=0.05,
                                        extend='both',
                                        spacing='uniform',
                                        shrink=.5)
        levels = numpy.asarray([0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25,50,100,250,500,1000])
        level = levels[numpy.nanargmin(abs((min([tmax,numpy.nanmax(data[-1])]) - numpy.nanmin(data[-1]))/5 - levels))]
        levels = numpy.arange(numpy.nanmin(data[-1]), min([tmax,numpy.nanmax(data[-1])]), level)
        levels -= numpy.nanmin(abs(levels))
        fig.cb.ax.set_xticks(levels)
        fig.cb.ax.axvline((numpy.around(reflevel)-min(fig.cb.get_clim()))/numpy.diff(fig.cb.get_clim()),zorder=999,color='k',linewidth=2)

    fig = self.plot(fig=fig,size=30, color='k')
    fig = self.plot(fig=fig,size=20, color='gray')


    fig.inset = inset_axes(fig.axes[0],
                           width="30%",  # width = 30% of parent_bbox
                           height="30%",  #
                           loc=2,
                           borderpad=0.)
    fig.inset.set_alpha(.5)
    fig.inset.set_ylabel('Number of P')
    fig.inset.set_xlabel(ax2xlabel)
    fig.inset.grid()
    fig.inset.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.inset.xaxis.set_ticks_position('top')
    fig.inset.xaxis.set_label_position('top')
    for i,n in enumerate(N):
        fig.inset.errorbar(data[i][data[i]<tmax],
                     numpy.repeat(n-.5,len(data[i][data[i]<tmax])),
                     yerr=.5,
                     color='k',
                     alpha=len(data[i][data[i]<tmax])*.5/len(data.flatten()),
                     linewidth=0,
                     elinewidth=1)
        cum = numpy.linspace(0,1,len(data[i][data[i]<tmax]))
        cum[cum>.5]=0.5-(cum[cum>.5]-0.5)
        cum -= numpy.nanmin(cum)
        cum /= numpy.nanmax(cum)
        p = numpy.diff(cum)/numpy.diff(numpy.sort(data[i][data[i]<tmax]))
        p = numpy.nanmean(obspy_addons.rolling(p,10),-1)
        p -= numpy.nanmin(p)
        p = p/numpy.nanmax(p)
        p = numpy.insert(p,0,0)
        t = numpy.nanmax(obspy_addons.rolling(numpy.sort(data[i][data[i]<tmax]),10),-1)
        fig.inset.plot(numpy.sort(data[i][data[i]<tmax]),
                       n-1+cum,
                       color='b')
    return fig


