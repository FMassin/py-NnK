# -*- coding: utf-8 -*-
"""
source - Addon module for obspy catalog.

This module provides additionnal functionnalities for event catalogs.
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

try:
    import obspy_addons
except:
    try:
        import eew.obspy_addons as obspy_addons
    except:
        import NnK.eew.obspy_addons as obspy_addons

def get(self, lst, att=None, types=[] , full=False, pref=False, last=False, first=False, nan=True):
    """
        1 use
        types = [ 'bests' [, 'all'] [, 'lasts' ] [, 'firsts' ] [, 'nans'] ]
        
        instead of 
        full=False, pref=False, last=False, first=False, nan=True
        
        2 use
        atts = [ level0, level1, etc ]
        instead 
        lst = level0 , att=level1
        
       
    """
    out = list()
    
    for e in self.events:
        patt = None
        if hasattr(e,'preferred_'+lst[:-1]):
            try :
                e['preferred_'+lst[:-1]][att]
                patt = 'preferred_'+lst[:-1]
            except:
                pass
        elif hasattr(e,'preferred_'+lst):
            try :
                e['preferred_'+lst][att]
                patt = 'preferred_'+lst
            except:
                pass
    
        if hasattr(e,lst) and ((set(['a','all']) & set(types)) or (full and not last and not first and not pref)):
            for o in e[lst]:
                if hasattr(o,att):
                    out.append(o[att])
                elif nan:
                    out.append(numpy.nan)

        elif patt and ((set(['p','pref']) & set(types)) or (pref or (not last and not first))):
            out.append(e[patt][att])

        elif hasattr(e,lst) and len(e[lst]) and ((set(['f','first']) & set(types)) or (first and not pref)):
            val = e[lst][0][att]
            mem=9999999999999999999999999999
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if hasattr(elt.creation_info,'creation_time'):
                        if elt.creation_info.creation_time:
                            if elt.creation_info.creation_time < mem:
                                mem = elt.creation_info.creation_time
                                val = elt[att]

            out.append(val)
                
        elif hasattr(e,lst) and len(e[lst]) and ((set(['l','last']) & set(types)) or (last or not pref)):
            val = e[lst][-1][att]
            mem=0
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if hasattr(elt.creation_info,'creation_time'):
                        if elt.creation_info.creation_time :
                            if elt.creation_info.creation_time > mem:
                                mem = elt.creation_info.creation_time
                                val = elt[att]
            out.append(val)
        else:
            if nan or set(['n','nan']) & set(types) : #:
                out.append(numpy.nan)
                
    return out

def eventsize(mags):
    
    min_size = 2
    max_size = 15
    
    steps=numpy.asarray([0.01,0.02,0.05,0.1,0.2,.5,1.,2.,5.])
    s = steps[numpy.argmin(abs((max(mags)-min(mags))/3-steps))]
    min_size_ = min(mags) - .1
    max_size_ = max(mags) + .1

    magsscale=[_i for _i in numpy.arange(-2,10,s) if _i>=min_size_ and _i<=max_size_]
    frac = [(0.2 + (_i - min_size_)) / (max_size_ - min_size_) for _i in magsscale]
    size_plot = [(_i * (max_size - min_size)) ** 2 for _i in frac]
    
    frac = [(0.2 + (_i - min_size_)) / (max_size_ - min_size_) for _i in mags]
    size = [(_i * (max_size - min_size)) ** 2 for _i in frac]
    
    x = (numpy.asarray(magsscale)-min(magsscale))/(max(magsscale)-min(magsscale))*.6+.2
    
    return size,size_plot,magsscale,x



def mapevents(self=obspy.core.event.catalog.Catalog(),
               bmap=None,
               fig=None,
               titletext='',
               eqcolorfield = 'depth',
               colorbar=None,
               fontsize=8):


    cf=[]
    mags = get(self,'magnitudes','mag',['b'] )
    times = get(self, 'origins','time', types=['b'])

    if len(mags) >0:
        reordered = numpy.argsort(mags)
        catalog = obspy.core.event.catalog.Catalog([ self.events[i] for i in reversed(reordered)])
        longitudes = get(self, 'origins','longitude', types=['b'])
        latitudes = get(self, 'origins','latitude', types=['b'])

        eqcolor = get(self, 'origins',eqcolorfield, types=['b'])
        if eqcolorfield == 'depth':
            eqcolorlabel = 'Event depth (km) and magnitude'
            eqcolor = numpy.asarray(eqcolor)/1000.

        sizes, sizesscale, labelsscale, x = eventsize( mags = get(self,'magnitudes','mag',['b'] ))

        bmap.scatter(longitudes,
                         latitudes,
                         sizes,
                         edgecolor='w',
                         lw=2)
        cf = bmap.scatter(longitudes,
                              latitudes,
                              sizes,
                              eqcolor,
                              edgecolor='None')
        titletext += '\n%s events (%s' % (len(times), str(min(times))[:10])
        if str(max(times))[:10] > str(min(times))[:10]:
            titletext += ' to %s)' % str(max(times))[:10]
        else:
            titletext += '%s)' % titletext
        
        if colorbar and len(catalog.events)>0 :
            fig.cb = obspy_addons.nicecolorbar(cf,
                                  axcb = bmap.ax,
                                  label = eqcolorlabel,
                                  data = eqcolor,
                                  fontsize=fontsize)
            fig.cb.ax.scatter(numpy.repeat(.1,len(sizesscale)),x,s=sizesscale, facecolor='None', linewidth=3, edgecolor='w')
            fig.cb.ax.scatter(numpy.repeat(.1,len(sizesscale)),x,s=sizesscale, facecolor='None', edgecolor='k')
            for i,v in enumerate(labelsscale):
                fig.cb.ax.text(-.2,x[i],'M'+str(v), va='center',ha='right', rotation='vertical',fontsize=fontsize)
    return titletext

def distfilter(self=obspy.core.event.catalog.Catalog(),
               dmax=None,
               dmin=0.,
               x1=None,
               y1=None,
               x2=None,
               y2=None,
               z1=None,
               save=True,
               out=False):
    
    distances=list()
    bad=list()
    good=list()
    for e in self.events:
        d=numpy.nan
        for o in e.origins: # IS THIS ALWAY THE RIGHT ORDER ? SHOULD IT BE SORTED BY CREATION TIME?
            if str(e.preferred_origin_id) == str(o.resource_id):
                break
        if (x1 and y1) or (x2 and y2):
            d = 110.*obspy_addons.DistancePointLine(o.longitude, o.latitude, x1, y1, x2, y2)
        else:
            d = o.quality.minimum_distance*110.
        if z1 is not None:
            d = numpy.sqrt(d**2. + (o.depth/1000-z1)**2)
        distances.append(d)

    for i,d in enumerate(distances):
        for o in self.events[i].origins: # IS THIS ALWAY THE RIGHT ORDER ? SHOULD IT BE SORTED BY CREATION TIME?
            if str(self.events[i].preferred_origin_id) == str(o.resource_id):
                break
        if str(self.events[i].event_type) == 'not existing' or str(o.evaluation_mode)=='automatic':
            pass
        else:
            if d<dmin or d>dmax:
                bad.append(self.events[i])
            else :
                good.append(self.events[i])


    if out in ['d','dist','distance']:
        return distances
    else:
        if out:
            return obspy.core.event.catalog.Catalog(events=good), obspy.core.event.catalog.Catalog(events=bad) #inside, outside
        else:
            return obspy.core.event.catalog.Catalog(events=good)


def plot_traveltime(self=obspy.core.event.catalog.Catalog(),
                    NumbersofP=[6,4],
                    NumbersofS=[1],
                    ax=None,
                    plots=True,
                    style='c',
                    iplot=None,
                    sticker_addons=None):

    if isinstance(self, list):
        letters = 'ABCDEFGH'
        f, (ax) = matplotlib.pyplot.subplots(len(self),1)
        # make a big axe so we have one ylabel for all subplots
        biga = f.add_subplot(111, frameon=False)
        # turn every element off the big axe so we don't see it
        biga.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

        for i,c in enumerate(self):

            if sticker_addons:
                if isinstance(sticker_addons, list):
                    sa = sticker_addons[i]
                else:
                    sa = letters[i] + '. ' + sticker_addons
            else:
                sa = letters[i] + '. '
            
            plot_traveltime(self=c,
                            NumbersofP=NumbersofP,
                            NumbersofS=NumbersofS,
                            ax=ax[i],
                            iplot=len(self)-i-1,
                            style=style,
                            sticker_addons=sa)
            #ax[i].grid()
            if i==0:
                if style in ['cum','c','cumulate']:
                    # add common labels
                    biga.set_ylabel(ax[i].get_ylabel())#'Number of phases')
                    ax[i].set_ylabel('')
                    #ax[i].set_xlabel('Observed travel Time')
                    obspy_addons.adjust_spines(ax[i], ['left', 'top'])
                else:
                    #ax[i].set_title('Observed travel times')
                    #ax[i].set_xlabel('Observed P travel time')
                    ax[i].set_ylabel('')#ax[i].set_ylabel('Observed S travel time')
                    obspy_addons.adjust_spines(ax[i], ['left', 'bottom'])
                
            elif i==len(self)-1:
                if style in ['cum','c','cumulate']:
                    ax[i].set_ylabel('')
                    obspy_addons.adjust_spines(ax[i], ['left', 'bottom'])
                else:
                    obspy_addons.adjust_spines(ax[i], ['right', 'bottom'])
            else:
                if style in ['cum','c','cumulate']:
                    ax[i].set_ylabel('')
                    obspy_addons.adjust_spines(ax[i], ['left'])
                else:
                    obspy_addons.adjust_spines(ax[i], ['bottom'])

            if style in ['cum','c','cumulate']:
                xmax=0
                for a in ax:
                    xmax = max([xmax, max(a.get_xlim())])
                for a in ax:
                    a.set_xlim([0, xmax])

        return ax
    maxarr=0
    for ie,e in enumerate(self.events):
        maxarr=max([maxarr, len(e.picks)])
    
    N=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    n=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    mags=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    
    

    for ie,e in enumerate(self.events):
        ns=0
        np=0
        VpVsRatio = 5./3.
        mag=e.magnitudes[0].mag
        for m in e.magnitudes:
            if str(e.preferred_magnitude_id) == str(m.resource_id):
                mag=m.mag
        for o in e.origins:
            if str(e.preferred_origin_id) == str(o.resource_id):
                t=list()
                for a in o.arrivals:
                    for p in e.picks:
                        if str(a.pick_id) == str(p.resource_id) :
                            t.append(p.time)
                
                for ia in numpy.argsort(t):
                    a=o.arrivals[ia]
                    for p in e.picks:
                        if str(a.pick_id) == str(p.resource_id) and p.time-o.time<30 and p.time-o.time>0:
                            if a.phase[0] == 'S':
                                ns+=1
                                n[ns][ie] = p.time-o.time
                                mags[np][ie] = mag
                            elif a.phase[0] == 'P':
                                np+=1
                                N[np][ie] = p.time-o.time
                                mags[np][ie] = mag
                AvVsVpRatio=0.
                AvN=0
                for p in range(np):
                    if n[p][ie] is not numpy.nan:
                        AvVsVpRatio += n[p][ie]/N[p][ie]
                        AvN+=1
                        VpVsRatio = AvVsVpRatio/AvN
                for p in range(np):
                    if n[p][ie] is numpy.nan:
                        n[p][ie] = N[p][ie]*VpVsRatio
                        
                break
    
    if not ax:
        f, (ax) = matplotlib.pyplot.subplots(1, 1)
        obspy_addons.adjust_spines(ax, ['left', 'bottom'])
    if style in ['vs','v','versus']:
        ax.set_ylabel('Observed S travel time')
        ax.set_xlabel('Observed P travel time')
        ax.set_aspect('equal')
    elif style in  ['cum','c','cumulate']:
        ax.set_ylabel('Number of phases')
        ax.set_xlabel('Observed travel Time')
    ax.grid()
    if sticker_addons:
        obspy_addons.sticker(sticker_addons, ax, x=0, y=1, ha='left', va='top')  # ,fontsize='xx-small')

    b=0
    if style in ['vs','v','versus']:
        for p in NumbersofP:
            for s in NumbersofS:
                b = max([b,numpy.nanmax(N[p]),numpy.nanmax(n[s])])
                if plots:
                    ax.scatter(N[p],
                               n[s],#                           s=nsta2msize(mags[s],[0,7]),norm=mag_norm([0,7])
                               marker='.',
                               alpha=0.25,
                               label=str(p)+'P || '+str(s)+'S')

    elif style in ['cum','c','cumulate']:

        l=['P','S']
        p=max(NumbersofP)+2
        b = max([b,numpy.nanmax(n[p])])
        b = max([b,numpy.nanmax(N[p])])

        for i in range(max(NumbersofP)):
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i,
                                                    n[i],
                                                    PorS='P',
                                                    upordown=1)
            
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i,
                                                    n[i],
                                                    PorS='P',
                                                    upordown=-1)
        ax.legend(ncol=2)
        if False:
            for e in range(len(n[p])):
                #n[0,e]=0
                ax.errorbar(n[:p,e],
                            numpy.arange(p)-.5,
                            yerr=.5,
                            color='g',
                            alpha=len(n[p])*0.02/500,
                            linewidth=0,
                            elinewidth=1)
            for e in range(len(N[p])):
                #N[0,e]=0
                ax.errorbar(N[:p,e],
                            numpy.arange(p)-.5,
                            yerr=.5,
                            color='b',
                            alpha=len(n[p])*0.02/500,
                            linewidth=0,
                            elinewidth=1)
            ax.errorbar(numpy.nanmedian(n[:p,:],1),
                        numpy.arange(p)-0.5,
                        yerr=.5,
                        color='g',
                        linewidth=0,
                        elinewidth=2,
                        label='S')
            ax.errorbar(numpy.nanmedian(N[:p,:],1),
                        numpy.arange(p)-0.5,
                        yerr=.5,
                        color='b',
                        linewidth=0,
                        elinewidth=2,
                        label='P')
            ax.errorbar(numpy.nanmedian(n[:p,:],1),
                        numpy.arange(p),
                        yerr=0,
                        xerr=numpy.nanstd(n[:p,:],1),
                        color='g',
                        linewidth=0,
                        elinewidth=2,capsize=4)
            ax.errorbar(numpy.nanmedian(N[:p,:],1),
                        numpy.arange(p),
                        yerr=0,
                        xerr=numpy.nanstd(N[:p,:],1),
                        color='b',
                        linewidth=0,
                        elinewidth=2,capsize=4)
    if plots:
        if not iplot:
            ax.legend(ncol=2)
        if style in ['vs','v','versus']:
            ax.plot([0,b],[0,b],color='grey')
    return ax, b

def plot_Mfirst(self=obspy.core.event.catalog.Catalog(),last=0, agency_id=['*'],minlikelyhood=None):
    
    solutions = Solutions(catalog=self,last=last, agency_id=agency_id, minlikelyhood = minlikelyhood)
    mags = solutions.mags
    profs = solutions.depths
    mags_stations = solutions.mags_station_counts
    m1_errors = solutions.mags_errors
    m1_types = solutions.mags_types
    m1_times = solutions.mags_creation_times
    m1_origint = solutions.origins_times
    m1_delays = solutions.mags_delays
        
    firstorlast='first'
    if last:
        firstorlast='last'
    
    f, (ax) = matplotlib.pyplot.subplots(1, 1)
    matplotlib.pyplot.ylabel('Error in '+firstorlast+' M')
    matplotlib.pyplot.title(firstorlast.title()+' M, with delays')
    matplotlib.pyplot.xlabel('Reference M')
    matplotlib.pyplot.grid()

    if len(mags) >0:

        for m in [min(mags_stations),numpy.median(mags_stations),max(mags_stations)]:
            sc = ax.scatter(numpy.mean(mags),
                            0,
                            nsta2msize(m,[0,32]),#mags_stations),
                            'w', 'o', label=str(int(m))+' stat.',alpha=0.9, edgecolor='w' )

        ax.axhline(0, linestyle='--', color='k') # horizontal lines

        if len(m1_delays)<32:
            for i, txt in enumerate(m1_delays):
                ax.text(mags[i], m1_errors[i], str(int(txt))+'s', weight="heavy",
                        color="k", zorder=100,
                        path_effects=[
                            matplotlib.patheffects.withStroke(linewidth=3,
                                                   foreground="white")])
        types = ['M*','Mfd','MVS']
        for i,m in enumerate(['+','^','o']):
            matches = [ j for j,t in enumerate(m1_types) if t == types[i] ]
            if matches:
                ax.scatter([mags[j] for j in matches] ,
                            [m1_errors[j] for j in matches] ,
                            nsta2msize([solutions.mags_orig_station_counts[j] for j in matches],[0,32]),# mags_stations)
                           [solutions.mags_orig_errors[j] for j in matches], #[profs[j] for j in matches],
                            m,
                            facecolor='w',alpha=1.,zorder=100,edgecolors='k')
                sc = ax.scatter([mags[j] for j in matches] ,
                                [m1_errors[j] for j in matches] ,
                                nsta2msize([solutions.mags_orig_station_counts[j] for j in matches],[0,32]),# mags_stations),
                                [solutions.mags_orig_errors[j] for j in matches],
                                m,
                                norm=depth_norm([0,100]),#profs),
                                label=types[i],alpha=.8,zorder=150,edgecolors='None')
        cb=matplotlib.pyplot.colorbar(sc)
        cb.set_label('Location error (km)')
        lg = matplotlib.pyplot.legend(loc=1, numpoints=1, scatterpoints=1, fancybox=True)
        lg.set_title('Marker & Sizes')
        lg.get_frame().set_alpha(0.1)
        lg.get_frame().set_color('k')
        matplotlib.pyplot.axis('equal')

    return f


