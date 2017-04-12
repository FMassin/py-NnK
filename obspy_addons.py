# -*- coding: utf-8 -*-
"""
source - Module for seismic sources modeling.

This module provides class hierarchy for earthquake modeling and
 representation.
______________________________________________________________________

.. note::

    Functions and classes are ordered from general to specific.

"""
import obspy
from obspy import read
import matplotlib
import matplotlib.patheffects
import numpy
from math import radians, cos, sin, asin, sqrt
import os
import glob




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
            val = e[lst][0]
            mem=9999999999999999999999999999
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if elt.creation_info.creation_time < mem:
                        mem = elt.creation_info.creation_time
                        val = elt[att]
            out.append(val)
                
        elif hasattr(e,lst) and len(e[lst]) and ((set(['l','last']) & set(types)) or (last or not pref)):
            val = e[lst][-1]
            mem=0
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if elt.creation_info.creation_time > mem:
                        mem = elt.creation_info.creation_time
                        val = elt[att]
            out.append(val)
        else:
            if nan or set(['n','nan']) & set(types) : #:
                out.append(numpy.nan)
                
    return out


class Solutions():
    def __init__(self,
                 catalog=obspy.core.event.catalog.Catalog(),
                 last=0,
                 arrivals=0,
                 agency_id=['*'],
                 magnitude_type=['MVS','Mfd'],
                 nan=False):

        self.mags=list()
        self.depths=list()
        
        self.mags_station_counts=list()
        self.mags_errors=list()
        self.mags_types=list()
        self.mags_creation_times=list()
        self.mags_delays=list()
        
        self.origins_times=dict()
        self.origins_creation_times=dict()
        self.origins_errors=dict()
        self.origins_delays=dict()
        self.origins_mags=dict()
        self.origins_station_counts=dict()
        self.origins_errorsdelays=dict()
        self.origins_errorsdelays_mags=dict()
        self.origins_mag_errors=dict()
        self.origins_mag_delays=dict()
        self.origins_mag_station_counts=dict()
        self.mags_errorsdelays=dict()
        
        self.picks_delays=list()
        self.picks_times=list()
        self.picks_oerrors=list()
    
        for e in catalog.events:
            Mtypes=list()
            #print(e)
            for m in e.magnitudes:
                if m.resource_id == e.preferred_magnitude_id:
                    preferred_magnitude_mag = m.mag
            for o in e.origins:
                if o.resource_id == e.preferred_origin_id:
                    #print(o)
                    preferred_origin_depth= o.depth
                    preferred_origin_latitude = o.latitude
                    preferred_origin_longitude = o.longitude
                    preferred_origin_time = o.time
            fn=list
            if last:
                fn=reversed
            
            found = 0
            sortedmarg = numpy.argsort([m.creation_info.creation_time for m in e.magnitudes])
            for mi in fn(sortedmarg) :# list(enumerate(e.magnitudes))):
                m = e.magnitudes[mi]
                
                if not hasattr(m.creation_info, 'creation_time'):
                    continue
                else:
                    if not m.creation_info.creation_time:
                        continue
                if (
                    (m.creation_info.agency_id in agency_id or '*' in agency_id) and
                    (m.magnitude_type in magnitude_type or '*' in magnitude_type) and
                    m.resource_id != e.preferred_magnitude_id and
                    e.preferred_magnitude_id is not None and
                    e.preferred_origin_id is not None
                    ) :
                    picks = e.picks.copy()
                    plist = []
                    #print(m)
                    
                    sortedoarg = numpy.argsort([o.creation_info.creation_time for o in e.origins])
                    
                    #for o in e.origins:
                    for oi in fn(sortedoarg) :
                        o = e.origins[oi]
                    
                        if not hasattr(o.creation_info, 'creation_time'):
                            continue
                        else:
                            if not o.creation_info.creation_time:
                                continue
                        dt=o.creation_info.creation_time-preferred_origin_time
                        d=haversine(preferred_origin_longitude,
                                    preferred_origin_latitude,
                                    o.longitude,
                                    o.latitude)/1000
                        OK_Maren=1
                        if m.magnitude_type in ['Mfd']:
                            if ('forel' in m.creation_info.author or
                                'alp' in m.creation_info.author):
                                OK_Maren=0
                                if ((0.39 * o.longitude + 44) < o.latitude and
                                    'forel' in m.creation_info.author):
                                    OK_Maren=1
                                elif ((0.39 * o.longitude + 44) >= o.latitude and
                                      'alp' in m.creation_info.author):
                                    OK_Maren=1
                    
                        try:
                            d = numpy.sqrt( d**2 + ((preferred_origin_depth-o.depth)/1000)**2)
                        except:
                            pass
                        
                        if (m.origin_id == o.resource_id and
                            o.resource_id != e.preferred_origin_id and
                            m.magnitude_type+m.creation_info.author+o.creation_info.author not in Mtypes  and
                            (m.magnitude_type not in ['MVS'] or ('NLoT_auloc' in o.creation_info.author or 'autoloc' in o.creation_info.author)) and
                            d>.001 and
                            dt>.01 and
                            dt<1000 and
                            OK_Maren == 1):

                            if m.magnitude_type is None :
                                m.station_count = o.quality.used_station_count
                            if m.mag is not None :
                                found = 1
                                if '*' is not last  :
                                    Mtypes.append(m.magnitude_type+m.creation_info.author+o.creation_info.author)
                                
                                if m.magnitude_type not in self.origins_errors:
                                    self.origins_errors[m.magnitude_type]=list()
                                    self.origins_delays[m.magnitude_type]=list()
                                    self.origins_creation_times[m.magnitude_type]=list()
                                    self.origins_times[m.magnitude_type]=list()
                                    self.origins_mags[m.magnitude_type]=list()
                                    self.origins_station_counts[m.magnitude_type]=list()
                                    self.origins_mag_errors[m.magnitude_type]=list()
                                    self.origins_mag_delays[m.magnitude_type]=list()
                                    self.origins_mag_station_counts[m.magnitude_type]=list()
                                
                                if str(e.resource_id)+o.creation_info.author not in self.origins_errorsdelays:
                                    self.origins_errorsdelays[str(e.resource_id)+o.creation_info.author]=list()
                                    self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author]=list()
                                    self.origins_errorsdelays_mags[str(e.resource_id)+o.creation_info.author]=preferred_magnitude_mag

                                self.mags.append(preferred_magnitude_mag)
                                try:
                                    self.depths.append(preferred_origin_depth/1000.)
                                except:
                                    self.depths.append(0)
                                
                                
                                if m.magnitude_type in ['Mfd'] :
                                    self.mags_station_counts.append(o.quality.used_station_count)
                                else:
                                    self.mags_station_counts.append(m.station_count)
                                
                                merror = m.mag-preferred_magnitude_mag
                                
                                self.mags_errors.append(merror)
                                self.mags_types.append(m.magnitude_type)
                                self.mags_creation_times.append(m.creation_info.creation_time)
                                
                                if m.creation_info.creation_time<o.creation_info.creation_time :
                                    self.mags_delays.append(o.creation_info.creation_time-preferred_origin_time)
                                else:
                                    self.mags_delays.append(m.creation_info.creation_time-preferred_origin_time)
                                
                                self.origins_station_counts[m.magnitude_type].append(o.quality.used_station_count)
                                self.origins_mags[m.magnitude_type].append(preferred_magnitude_mag)
                                self.origins_times[m.magnitude_type].append(preferred_origin_time)
                                self.origins_creation_times[m.magnitude_type].append(o.creation_info.creation_time)
                                self.origins_delays[m.magnitude_type].append(o.creation_info.creation_time-preferred_origin_time)
                                self.origins_errors[m.magnitude_type].append(d)
                                self.origins_mag_errors[m.magnitude_type].append(m.mag-preferred_magnitude_mag)
                                
                                if m.creation_info.creation_time<o.creation_info.creation_time :
                                    self.origins_mag_delays[m.magnitude_type].append(o.creation_info.creation_time-preferred_origin_time)
                                    self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([o.creation_info.creation_time-preferred_origin_time, merror])
                                else:
                                    self.origins_mag_delays[m.magnitude_type].append(m.creation_info.creation_time-preferred_origin_time)
                                    self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([m.creation_info.creation_time-preferred_origin_time, merror])
                                
                                if m.magnitude_type in ['Mfd'] :
                                    self.origins_mag_station_counts[m.magnitude_type].append(o.quality.used_station_count)
                                else:
                                    self.origins_mag_station_counts[m.magnitude_type].append(m.station_count)

                                self.origins_errorsdelays[str(e.resource_id)+o.creation_info.author].append([o.creation_info.creation_time-preferred_origin_time, d])
                                

                                
                                
                                if False:
                                    if arrivals and len(self.picks_times)<100:
                                        for a in o.arrivals:
                                            if a.time_weight is not None:
                                                if a.time_weight < 4 : #abs(a.time_residual)<(a.distance)/5./3:
                                                    for p in picks:
                                                        if (a.pick_id == p.resource_id and
                                                            p.time-preferred_origin_time > .5 and
                                                            p.time not in plist):
                                                            
                                                            self.picks_times.append(p.time-preferred_origin_time)
                                                            self.picks_delays.append(p.creation_info.creation_time-preferred_origin_time)
                                                            self.picks_oerrors.append(self.origins_errors[-1])
                                                            
                                                            picks.remove(p)
                                                            plist.append(p.time)
                                                            break


            if not found and nan:
                self.mags_delays.append(numpy.nan)


def plot_Mfirst(self=obspy.core.event.catalog.Catalog(),last=0, agency_id=['*']):
    
    solutions = Solutions(catalog=self,last=last, agency_id=agency_id)
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
                            nsta2msize([mags_stations[j] for j in matches],[0,32]),#mags_stations),
                            [profs[j] for j in matches],
                            m,
                            facecolor='w',alpha=1.,zorder=100,edgecolors='k')
                sc = ax.scatter([mags[j] for j in matches] ,
                                [m1_errors[j] for j in matches] ,
                                nsta2msize([mags_stations[j] for j in matches],[0,32]),#mags_stations),
                                [profs[j] for j in matches],
                                m,
                                norm=depth_norm([0,50]),#profs),
                                label=types[i],alpha=.8,zorder=150,edgecolors='None')
        cb=matplotlib.pyplot.colorbar(sc)
        cb.set_label('Reference depth (km)')
        lg = matplotlib.pyplot.legend(loc=1, numpoints=1, scatterpoints=1, fancybox=True)
        lg.set_title('Marker & Sizes')
        lg.get_frame().set_alpha(0.1)
        lg.get_frame().set_color('k')
        matplotlib.pyplot.axis('equal')
        print('set set_xlim([2,5.5]')
        ax.set_ylim([-1.1,1.1])

    return f

def plot_map(self=obspy.core.event.catalog.Catalog(),t='MVS',a='Pb',**kwargs):

    catalogcopy = self.copy()
    
    solutions = Solutions(catalog=self,last=0, arrivals=0, agency_id=a, magnitude_type=[t], nan=True)

    for i,e in enumerate(catalogcopy.events):
        if solutions.mags_delays[i]>30:
            solutions.mags_delays[i]=30
        e.depth = solutions.mags_delays[i]
        for o in e.origins:
            o.depth = solutions.mags_delays[i]*1000

    catalogcopy.events.insert(0,e)
    catalogcopy.events[0].depth = 0
    for o in catalogcopy.events[0].origins:
        o.depth = 0
    catalogcopy.events.insert(0,e)
    catalogcopy.events[0].depth = 30*1000
    for o in catalogcopy.events[0].origins:
        o.depth = 30*1000
        
    f = catalogcopy.plot(**kwargs)
    f.texts[0].set_text(f.texts[0].get_text().replace('depth', t+' delay'))
    f.texts[0].set_text(f.texts[0].get_text().replace(' - ', '\n'))
    #s.set_clim([0,30])

    return f


def plot_Mfirst_hist(self=obspy.core.event.catalog.Catalog(),agency_id=['*'],log=None):

    solutions = Solutions(catalog=self,last=0, arrivals=0, agency_id=agency_id)
    solutions_last = Solutions(catalog=self,last=1, arrivals=0, agency_id=agency_id)
    
    f, ax = matplotlib.pyplot.subplots(2, 1, sharey=True)
    ax[0].xaxis.set_ticks_position('top')
    ax[0].xaxis.set_label_position('top')
    ax[0].set_xlabel('Error in location (km)')
    ax[1].set_xlabel('Error in magnitude')
    medsigma12=[2.5, 15.9, 50, 84.1, 97.5]
    for a in ax:
        a.set_ylabel('%')
        a.grid()
        a.set_yticks(medsigma12)
        axr=a.twinx()
        axr.set_yticks(medsigma12)
        axr.set_yticklabels(['$2\sigma$','$\sigma$','$\widetilde{M}$','$\sigma$','$2\sigma$' ])
        axr.set_ylim([0, 100])
    
    ax[0].set_xscale('log')
    
    types = ['M*','Mfd','MVS']
    o=[0,0,0,0]
    for i,m in enumerate(['y','b','r']):
        matches = [ j for j,t in enumerate(solutions.mags_types) if t == types[i] ]
        if matches:
            
            first=[solutions.mags_errors[j] for j in matches]
            last=[solutions_last.mags_errors[j] for j in matches]
            
            #ax[1].plot([numpy.median(first),numpy.median(first)], [0,50], color=m)
            #ax[1].plot([numpy.median(last),numpy.median(last)], [0,50], linestyle=':', color=m)
            
            ax[1].fill(numpy.append(numpy.sort(last),numpy.sort(first)[::-1]), #numpy.sort(solutions.mags_errors),
                       numpy.append((numpy.arange(len(matches))+.5)*100/len(matches),((numpy.arange(len(matches))+.5)*100/len(matches))[::-1]),
                       linewidth=0,
                       color=m,
                       alpha=.2)
                       
            o[0] = ax[1].plot(numpy.sort(first), #numpy.sort(solutions.mags_errors),
                    (numpy.arange(len(matches))+.5)*100/len(matches),
                              label='$\widetilde{M_{%s}^{first}} %.2f$' % (types[i][1:].upper(), numpy.median(first) ),
                       color=m)
            
            o[1] = ax[1].plot(numpy.sort(last), #numpy.sort(solutions.mags_errors),
                      (numpy.arange(len(matches))+.5)*100/len(matches),
                       label='$\widetilde{M_{%s}^{last}} %.2f$' % (types[i][1:].upper(), numpy.median(last) ) ,
                       linestyle='--',
                       color=m)
        
            first=solutions.origins_errors[types[i]]
            last=solutions_last.origins_errors[types[i]]
            
            #ax[0].plot([numpy.median(first),numpy.median(first)], [0,50], color=m)
            #ax[0].plot([numpy.median(last),numpy.median(last)], [0,50],linestyle=':', color=m)
            
            ax[0].fill(numpy.append(numpy.sort(last),numpy.sort(first)[::-1]), #numpy.sort(solutions.mags_errors),
                       numpy.append((numpy.arange(len(matches))+.5)*100/len(matches),((numpy.arange(len(matches))+.5)*100/len(matches))[::-1]),
                       linewidth=0,
                       color=m,
                       alpha=.2)
                       
            o[2] = ax[0].plot(numpy.sort(first), #numpy.sort(solutions.mags_errors),
                      (numpy.arange(len(matches))+.5)*100/len(matches),
                      label='$\widetilde{M_{%s}^{first}} %.2f_{km}$' % (types[i][1:].upper(), numpy.median(first) ),
                      color=m )
            
            o[3] = ax[0].plot(numpy.sort(last), #numpy.sort(solutions.mags_errors),
                      (numpy.arange(len(matches))+.5)*100/len(matches),
                      label='$\widetilde{M_{%s}^{last}} %.2f_{km}$' % (types[i][1:].upper(), numpy.median(last) ),
                       linestyle='--',
                       color=m)

    ax[0].legend(loc=4, numpoints=1, scatterpoints=1, fancybox=True, framealpha=0.5)
    ax[1].legend(loc=2, numpoints=1, scatterpoints=1, fancybox=True, framealpha=0.5)
    ax[0].set_xlim([1,100])
    ax[1].set_xlim([-1.1,1.1])
    print('set set_xlim([1,100]) and [-1.1,1.1]')
    return f

def plot_eew(self=obspy.core.event.catalog.Catalog(),last=0,agency_id=['*'],log=None):
    
    solutions_first = Solutions(catalog=self,last=0, arrivals=0, agency_id=agency_id)
    solutions_all = Solutions(catalog=self,last='*', arrivals=0, agency_id=agency_id)
    
    f, (ax1,ax2) = matplotlib.pyplot.subplots(2, 1, sharex=True)
    ax1.set_ylabel('Error in location (km)')
    ax2.set_ylabel('Error in magnitude')
    ax1.set_xlabel('Time after origins (s)')
    ax2.set_xlabel('Time after origins (s)')
    if log:
        ax2.set_xscale('log')
        ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.grid()
    ax2.grid()
    ax2.axhline(0, linestyle='--', color='k') # horizontal lines

    if len(solutions_first.mags) >0:
        
        
        line_segments1 = matplotlib.collections.LineCollection([solutions_all.origins_errorsdelays[l]
                                                               for l in solutions_all.origins_errorsdelays],
                                                              linewidths=1,
                                                               linestyles='solid',zorder=1,alpha=.9,
                                                              norm=mag_norm([2,5]))#solutions_all.mags))
        line_segments1.set_array(numpy.asarray([solutions_all.origins_errorsdelays_mags[l] for l in solutions_all.origins_errorsdelays]))
        
        
        line_segments2 = matplotlib.collections.LineCollection([solutions_all.mags_errorsdelays[l]
                                                               for l in solutions_all.mags_errorsdelays],
                                                              linewidths=1,
                                                               linestyles='solid',zorder=1,alpha=.9,
                                                              norm=mag_norm([2,5]))#solutions_all.mags))
        line_segments2.set_array(numpy.asarray([solutions_all.origins_errorsdelays_mags[l] for l in solutions_all.origins_errorsdelays]))
        
        
        ax1.add_collection(line_segments1)
        ax2.add_collection(line_segments2)
        markers = {'MVS':'o','Mfd':'^'}
        for k in solutions_first.origins_delays:
            
            i_first = numpy.argsort(solutions_first.origins_mags[k])
            
            sizes = nsta2msize([solutions_first.origins_station_counts[k][i] for i in i_first],
                               [0,32])#solutions_first.mags_station_counts)
            
            ax1.scatter([solutions_first.origins_delays[k][i] for i in i_first],
                        [solutions_first.origins_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='k')
            sc1 = ax1.scatter([solutions_first.origins_delays[k][i] for i in i_first],
                              [solutions_first.origins_errors[k][i] for i in i_first],
                              sizes,
                              [solutions_first.origins_mags[k][i] for i in i_first],
                              marker=markers[k],
                              norm=mag_norm([2,5]),#solutions_all.mags),#label=e.short_str(),
                              linewidths=0,zorder=2,
                              alpha=.9,edgecolors='None')


            sizes = nsta2msize([solutions_first.origins_mag_station_counts[k][i] for i in i_first],
                               [0,32])#solutions_first.mags_station_counts)
            
            ax2.scatter([solutions_first.origins_mag_delays[k][i] for i in i_first],
                        [solutions_first.origins_mag_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='k')
            sc2 = ax2.scatter([solutions_first.origins_mag_delays[k][i] for i in i_first],
                              [solutions_first.origins_mag_errors[k][i] for i in i_first],
                              sizes,
                              [solutions_first.origins_mags[k][i] for i in i_first],
                              marker=markers[k],
                              norm=mag_norm([2,5]),#solutions_all.mags),
                              linewidths=0,zorder=2,
                              alpha=.9,edgecolors='None')
        
        
                    
        cb=matplotlib.pyplot.colorbar(line_segments1, ax=[ax1,ax2])#ax1, ax=[ax1,ax2])
        cb.set_label('Reference magnitude')
        ob2 = ax2.scatter(20,0,10,marker='o',alpha=0.1,color='b', edgecolors='none',zorder=-999)
        ob1 = ax2.scatter(20,0,10,marker='o',color='b',edgecolors='k',zorder=-999)
        ax2.scatter(20,0,10,marker='o',color='w',edgecolors='w',linewidths=3,zorder=-99)
        ax2.set_xlim([1,30])
        ax2.set_ylim([-1.1,1.1])
        ax1.set_ylim([1,100])
        xmin=9999999
        xmax=0
        for k in solutions_first.origins_mag_delays:
            xmin = numpy.min([xmin, numpy.min(solutions_first.origins_delays[k])])
            xmax = numpy.max([xmax, numpy.max(solutions_first.origins_mag_delays[k])])
        ax2.set_xlim([1,xmax*1.1])
        ax2.set_xlim([3,30])
            #lg = matplotlib.pyplot.legend((ob1, ob2),
            #                          ('Solutions (loc. or M)', 'Picks (t or A)'),
            #                          numpoints=1,
            #                          scatterpoints=1,
            #                          fancybox=True,
            #                          loc=4)
    return f

def evfind(self=obspy.core.event.catalog.Catalog(),
           toadd=obspy.core.event.catalog.Catalog(),
           client=None,
           v=False,
           x=True):
    """
    use get_values instead of loop
    """
    
    tofind = toadd.copy()
    matchs = obspy.core.event.catalog.Catalog()
    extras = obspy.core.event.catalog.Catalog()
    matchs.description = 'Intersections of '+str(self.description)+' and '+str(tofind.description)
    missed = obspy.core.event.catalog.Catalog()
    missed.description = 'Part of '+str(self.description)+' not in '+str(tofind.description)
    listfound=list()
    
    for e in self.events :
        
        o = e.preferred_origin() or e.origins[-1]
        
       
        found=False
        memdl=99999999
        memdt=99999999
        if v:
            print('---- event',e.short_str(),'...')
    
        if len(listfound) < len(self.events):
            
            filter = ["time >= "+str(o.time-30),
                      "time <= "+str(o.time+30),
                      "latitude >= "+str(o.latitude-1),
                      "latitude <= "+str(o.latitude+1),
                      "longitude >= "+str(o.longitude-1),
                      "longitude <= "+str(o.longitude+1)]
                      
            if client :
                eq_specs = {'starttime':str(o.time-30),
                            'endtime':str(o.time+30),
                            'minlatitude':o.latitude-1,
                            'maxlatitude':o.latitude+1,
                            'minlongitude':o.longitude-1,
                            'maxlongitude':o.longitude+1,
                            'includearrivals':True}
                tofind = client.get_events( **eq_specs )
            
            for candidate in tofind.filter( *filter ).events:

                i = candidate.resource_id
                candidateo = candidate.preferred_origin() or candidate.origins[-1]
                dt = abs(o.time-candidateo.time)
                dl = numpy.sqrt((o.latitude-candidateo.latitude)**2+(o.longitude-candidateo.longitude)**2)
                
                if (dt < 5 and dl <=.1 and
                    (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    meme = candidate
                    break
                    
                    if v:
                        print('fits nicely input catalog ',tofind.description,':\n  ', ref.short_str())
            
                elif (dt < 60 and dl <=.5 and
                      (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    meme = candidate
                    if v:
                        print('fits input catalog ',tofind.description,':\n  ', ref.short_str())

                elif (dt < 60 and dl <1 and
                      (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    meme = candidate
                    if v:
                        print('poorly fits input catalog ',tofind.description,':\n  ', ref.short_str())
                

        if not found:
            if v:
                print('does not exist in current catalog')
            missed.events.append(e)

        elif found:
            #matchs.events.append(e)
            matchs.events.append(meme)
            listfound.append(memi)
            
            #for prefatt in [ 'preferred_origin_id', 'preferred_magnitude_id' ]:
            #    if hasattr(meme, prefatt):
            #        matchs.events[-1][prefatt] = meme[prefatt]
        
            
            for listatt in [ 'origins' , 'magnitudes', 'picks' ]:
            #    if hasattr(meme, listatt):
            #        matchs.events[-1][listatt].extend( meme[listatt] )

                if hasattr(e, listatt):
                    matchs.events[-1][listatt].extend( e[listatt] )

            tofind.events.remove(meme)
            
                
            

    if x :
        matchs_otherversion, extras, trash = tofind.evfind(self,v=v,x=False)
    return matchs, missed, extras



def hasdata(self,data=obspy.core.stream.Stream()):
    inv_ok = obspy.core.inventory.Inventory(networks=[],source=[])
    inv_dead = obspy.core.inventory.Inventory(networks=[],source=[])
    l = []
    ld = []
    for t in data.traces:
        if t.stats.network+t.stats.station not in l:
            toadd = self.select(network=t.stats.network,station=t.stats.station)
            if len(toadd.networks) >0:
                found = False
                for n in inv_ok.networks:
                    if n.code == t.stats.network:
                        n.stations.append(toadd.networks[0].stations[0])
                        found = True
                if not found:
                    inv_ok.__iadd__(toadd)
                l.append(t.stats.network+t.stats.station)
        for n in self.networks:
            for s in n.stations:
                code = str(n.code)+str(s.code)
                if code not in l and code not in ld:
                    toadd = self.select(network=str(n.code),station=str(s.code))
                    if len(toadd.networks) > 0:
                        found = False
                        for nd in inv_dead.networks:
                            if str(nd.code) == str(n.code):
                                nd.stations.append(toadd.networks[0].stations[0])
                                found = True
                        if not found:
                            inv_dead.__iadd__(toadd)
                        ld.append(code)

    return inv_ok, inv_dead


def last_origin(self=obspy.core.event.event.Event()):
    """
    >>> obspy.core.event.event.Event.last_origin = obspy_addons.last_origin
    """
    last=self.origins[-1]
    for o in self.origins:
        if o.creation_info.creation_time > last.creation_info.creation_time:
            last = o
    return last

def get_event_waveform(self=obspy.core.event.event.Event(), client_eq=None, afterpick = 30):
    """
    >>> obspy.core.event.event.Event.get_event_waveform = obspy_addons.get_event_waveform
    """

    o = self.preferred_origin() or self.last_origin()
    
    pmax = max([p.time for p in self.picks])
    dmax = max([a.distance or 0. for a in o.arrivals])

    stations = client_eq.get_stations(latitude=o.latitude,
                                      longitude=o.longitude,
                                      maxradius=dmax,
                                      level='channel')
            
    bulk = [ (tuple(c.split('.')))+(o.time,pmax+afterpick)  for c in stations.get_contents()['channels']]

    try:
        self.stream = client_eq.get_waveforms_bulk(bulk, attach_response=True)
    except:
        self.stream = obspy.core.stream.Stream()

    self.stream = self.stream.slice(starttime=o.time, endtime=pmax+afterpick )
    self.stream.merge(method=1)
    self.stream.detrend()

    for t in self.stream.traces:
        coordinates = stations.get_coordinates(t.id)
        t.stats.distance = haversine(o.longitude, o.latitude, coordinates['longitude'], coordinates['latitude'])
        t.stats.distance = numpy.sqrt((coordinates['elevation']+o.depth)**2 + t.stats.distance**2)

        multibands = t.copy()
        print(multibands)
        tmp.remove_response(output="VEL")
        multicomponents = multibands.copy()
        
        multibands.stats.channel= multibands.stats.channel[-1]
        multibands.data = multibands.data**2.

        multicomponents.stats.channel= ''
        multicomponents.data = multicomponents.data**2.
        
        # update or create the multiban channel
        try:
            tmp = st.select(network=t.stats.network,
                              station=t.stats.station,
                              location=t.stats.location,
                              channel=t.stats.channel[-1])
            tmp.data += multibands.data
        except:
            self.stream.append(multibands.data)

        # update or create the multicomp channel
        try:
            t_sum = st.select(network=t.stats.network,
                              station=t.stats.station,
                              location=t.stats.location,
                              channel='')
            t_sum.data += multicomponents.data
        except:
            self.stream.append(multicomponents.data)
    
    # finalizes the euclidian sum of the hybrid channels
    for c in ['', 'Z', 'E', 'N', '1', '2', '3']:
        for t in self.stream.select(channel=''):
            t.data = sqrt(t.data)

    return self

def plot_eventsections(self, client_wf, afterpick = 30, file = None):

    fig = list()
    for ie,e in enumerate(self.events):

        picks = []
        arrivals = []
        st = obspy.core.stream.Stream()
        fileok=None
        fst = None
        if file:
            fst = read(file[ie])
            fileok = file
        else:
            for fn in glob.glob('/Users/fmassin/Google Drive/Projects/SED-EEW/events_playback/finder/100bigger-lastyear-20170101.arclink.ethz/'+(str(e.resource_id)).replace('/', '_')+'*mseed'):
                fst = read(fn)
                fileok = fn
            for fn in glob.glob('/Users/fmassin/Google Drive/Projects/SED-EEW/events_playback/eew-vs/100bigger-lastyear-20170101.eew-vs.gps.caltech/'+(str(e.resource_id)).replace('/', '_')+'*mseed'):
                fst = read(fn)
                fileok = fn

            if not fst and os.path.isfile('data/'+(str(e.resource_id)).replace('/', '_')):
                if  os.stat('data/'+(str(e.resource_id)).replace('/', '_')).st_size > 0:
                    fst = read('data/'+(str(e.resource_id)).replace('/', '_'))
                    fileok = 'data/'+(str(e.resource_id)).replace('/', '_')
                else:
                    os.remove('data/'+(str(e.resource_id)).replace('/', '_'))

        o = e.origins[-1]
        for co in e.origins:
            if co.resource_id == e.preferred_origin_id:
                o=co

        pmax = max([p.time for p in e.picks])

        for p in e.picks:
            for a in o.arrivals:
                if str(a.pick_id) == str(p.resource_id):
                    picks.append(p)
                    arrivals.append(a)

        distances = [ a.distance for a in arrivals ]

        for indexp in numpy.argsort(distances) :
            p = picks[indexp]
            a = arrivals[indexp]

            if ( a.distance*110. < 30.*3.):

                if not p.waveform_id.location_code:
                    p.waveform_id.location_code =''
                
                if not fileok:
                    try:
                        toadd = client_wf.get_waveforms(p.waveform_id.network_code,
                                                        p.waveform_id.station_code,
                                                        p.waveform_id.location_code,
                                                        p.waveform_id.channel_code,
                                                        starttime = o.time,
                                                        endtime = numpy.min([o.time+30,pmax+afterpick]) )
                    except:
                        toadd = obspy.core.stream.Stream()
                else:
                    toadd = fst.select(id=p.waveform_id.network_code+'.'+p.waveform_id.station_code+'.'+p.waveform_id.location_code+'.'+p.waveform_id.channel_code)
                
                for tr in toadd:
                    ifile = 'data/'+tr.stats.network+tr.stats.station+tr.stats.location+tr.stats.channel+str(o.time)+'.xml'
                    if not os.path.isfile(ifile):
                        try:
                            inv=client_wf.get_stations(startbefore = o.time,
                                                       endafter = numpy.min([o.time+30,pmax+afterpick]),
                                                       network=tr.stats.network,
                                                       station=tr.stats.station,
                                                       location=tr.stats.location,
                                                       channel=tr.stats.channel,
                                                       level="response")
                            inv.write(ifile, format='STATIONXML')
                        except:
                            print('removing'+tr.stats.network+tr.stats.station+tr.stats.location+tr.stats.channel)
                            toadd.remove(tr)
                    else:
                        inv = obspy.read_inventory(ifile)
                    
                    toadd.attach_response(inv)
                    
                st += toadd
                        
                for t in st.select(id=p.waveform_id.network_code+'.'+p.waveform_id.station_code+'.'+p.waveform_id.location_code+'.'+p.waveform_id.channel_code):
                    t.stats.distance = a.distance*110000.
                
                if len(st)>20 :
                    break
        
        if not fileok:
            if not os.path.exists('data'):
                os.makedirs('data')

            st.write('data/'+(str(e.resource_id)).replace('/', '_'), format="MSEED")


        if len(st)>0:

            st.remove_response(output="VEL")
            st.detrend()
            tmp=st.slice(starttime=o.time, endtime=o.time+30)
            tmp.merge(method=1)
            
            fig.append( matplotlib.pyplot.figure() )
            
            tmp.plot(type='section', # starttime=o.time-10,
                     reftime=o.time,
                     time_down=True,
                     linewidth=.5,
                     grid_linewidth=.25,
                     show=False,
                     fig=fig[-1],
                     color='network',
                     orientation='horizontal')
            ax = matplotlib.pyplot.gca()
            
            transform = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transAxes )
            transform_picks = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transAxes)

            for i,tr in enumerate(st):
                ax.text(30.0, tr.stats.distance / 1e3,  tr.stats.station, #rotation=270,
                        va="bottom", ha="left",#         transform=transform,
                        zorder=10)
            
            
            for i,p in enumerate(picks):
                if arrivals[i].distance*110 < tr.stats.distance/ 1e3:
                    ax.plot(picks[i].time - o.time,
                            arrivals[i].distance*110,
                            '+k',#                        transform=transform_picks,
                            zorder=10)
                            
                    ax.text(picks[i].time - o.time,
                            arrivals[i].distance*110,
                            str(arrivals[i].phase),
                            weight="heavy",
                            color="k",
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            zorder=-20,alpha=.3,
                            path_effects=[matplotlib.patheffects.withStroke(linewidth=3,
                                                                        foreground="white")])

            ax2 = fig[-1].add_subplot(311)
            pos1 = ax.get_position() # get the original position
            pos2 = [pos1.x0 , pos1.y0,  pos1.width, pos1.height *2/3.]
            ax.set_position(pos2) # set a new position
            markers = {'MVS':'o','Mfd':'^'}
            for cm in e.magnitudes:
                for co in e.origins:
                    if str(cm.origin_id) == str(co.resource_id):
                        
                        OK_pipeline=1
                        if cm.magnitude_type in ['Mfd']:
                            if ('forel' in cm.creation_info.author or
                                'alp' in cm.creation_info.author):
                                OK_pipeline=0
                                if ((0.39 * co.longitude + 44) < co.latitude and
                                    'forel' in cm.creation_info.author):
                                    OK_pipeline=1
                                elif ((0.39 * co.longitude + 44) >= co.latitude and
                                      'alp' in cm.creation_info.author):
                                    OK_pipeline=1
                            else:
                                OK_pipeline=1

                        elif (cm.magnitude_type in ['MVS'] and
                              ('NLoT_auloc' in co.creation_info.author or
                               'autoloc' in co.creation_info.author)):
                            OK_pipeline=1
                        else:
                            OK_pipeline=0
                            if False:#cm.creation_info.agency_id in ['Pb']:
                                print('rejected:')
                                print(cm)
                                print(co)
                        

                        if (cm.magnitude_type in ['MVS', 'Mfd'] and
                            cm.creation_info.agency_id in ['Pb'] and
                            OK_pipeline) :
                            
                            ct = max([cm.creation_info.creation_time, co.creation_info.creation_time ])
                            print(ct)
                            tmp=st.slice(starttime=o.time, endtime= ct )
                            
                            PGV = numpy.asarray([numpy.max(abs(tr.data)) for tr in tmp])
                            
                            ax.axvline(cm.creation_info.creation_time - o.time, linewidth=.1, linestyle=':', color='k')
                            
                            PGVerror = PGV - cuaheaton2007(magnitudes=[cm.mag], R = [tr.stats.distance/1000  for tr in tmp ])
                            
                            ax2.plot(numpy.tile(ct - o.time, PGV.shape),
                                     PGVerror,
                                     markers[cm.magnitude_type], alpha=.1)#, color='k')



                
            ax2.xaxis.set_ticks_position('top')
            ax2.xaxis.set_label_position('top')
            ax2.set_xlim([0,30])
            l = ax.get_legend()
            la = [ text.get_text() for text in l.get_texts()]
            [line.set_linewidth(3) for line in l.get_lines()]
            li = l.get_lines()
            l.remove()
            l = ax.legend(li,la, loc='lower right', ncol=7,prop={'size':6},title=e.short_str()+' \n '+str(e.resource_id))
            l.get_title().set_fontsize('6')
            ax.set_xlabel('Time after origin [s]')
            ax2.set_xlabel('Time after origin [s]')
            ax2.set_ylabel(r'PGV error [$m.s^{-1}$] ')
            ax2.yaxis.grid(True,linestyle='--')
            ax.xaxis.grid(False)
    

    return fig


def cuaheaton2007(R=[100],
                  magnitudes=[5.],
                  types=['rock'],
                  outputs=['velocity'],
                  phases=['S-wave'],
                  components=['Vertical amplitudes'],
                  corrections=[0]):
    """
        Cua, G., & Heaton, T. (2007). The Virtual Seismologist (VS) method: A Bayesian approach to earthquake early warning (pp. 97–132). Berlin, Heidelberg: Springer Berlin Heidelberg. http://doi.org/10.1007/978-3-540-72241-0_7
    """
    parameters = {'name': ['a','b','c1','c2','d','e','s'],
                    'values':
                    {'Root mean square horizontal amplitudes':
                        {'P-wave':
                            {'Acceleration':
                                {'rock': [ 0.72, 3.3*10**-3, 1.6,  1.05, 1.2, -1.06, 0.31 ],
                                'soil': [ 0.74, 3.3*10**-3, 2.41, 0.95, 1.26,-1.05, 0.29 ]},
                            'velocity':
                                {'rock': [0.80, 8.4*10**-4, 0.76, 1.03, 1.24, -3.103, 0.27],
                                'soil': [0.84, 5.4*10**-4, 1.21, 0.97, 1.28, -3.13, 0.26]},
                            'displacement':
                                {'rock': [0.95, 1.7*10**-7, 2.16, 1.08, 1.27, -4.96 , 0.28],
                                'soil': [0.94, -5.17*10**-7, 2.26, 1.02, 1.16, -5.01, 0.3]}},
                        'S-wave':
                            {'acceleration':
                                {'rock': [0.78, 2.6*10**-3, 1.48, 1.11, 1.35, -0.64, 0.31],
                                'soil': [0.84, 2.3*10**-3, 2.42, 1.05, 1.56, -0.34, 0.31]},
                            'velocity':
                                {'rock': [0.89, 4.3*10**-4, 1.11, 1.11, 1.44, -2.60, 0.28],
                                'soil': [0.96, 8.3*10**-4, 1.98, 1.06, 1.59, -2.35, 0.30]},
                            'displacement':
                                {'rock': [1.03, 1.01*10**-7, 1.09, 1.13, 1.43, -4.34, 0.27],
                                'soil': [1.08, 1.2*10**-6, 1.95, 1.09, 1.56, -4.1, 0.32]}}},
                    'Vertical amplitudes':
                        {'P-wave':
                            {'acceleration':
                                {'rock': [0.74, 4.01*10**-3, 1.75, 1.09, 1.2, -0.96, 0.29],
                                'soil': [0.74, 5.17*10**-7, 2.03, 0.97, 1.2, -0.77, 0.31]},
                            'velocity':
                                {'rock': [0.82, 8.54*10**-4, 1.14, 1.11, 1.36, -0.21, 0.26],
                                'soil': [0.81, 2.65*10**-6, 1.4, 1.0, 1.48, -2.55, 0.30]},
                            'displacement':
                                {'rock': [0.96, 1.98*10**-6, 1.66,1.16, 1.34, -4.79, 0.28],
                                'soil': [0.93, 1.09*10**-7, 1.5, 1.04, 1.23, -4.74, 0.31]}},
                        'S-wave':
                            {'acceleration':
                                {'rock': [0.78, 2.7*10**-3, 1.76, 1.11, 1.38, -0.75, 0.30],
                                'soil': [0.75,2.47*10**-3, 1.59, 1.01, 1.47, -0.36, 0.30]},
                            'velocity':
                                {'rock': [0.90, 1.03*10**-3, 1.39, 1.09, 1.51, -2.78, 0.25],
                                'soil': [0.88, 5.41*10**-4, 1.53, 1.04, 1.48, -2.54, 0.27]},
                            'displacement':
                                {'rock': [1.04, 1.12*10**-5, 1.38, 1.18, 1.37, -4.74, 0.25],
                                'soil': [1.03, 4.92*10**-6, 1.55, 1.08, 1.36, -4.57, 0.28]}}}}}


    magnitudes = tolisttoarray(magnitudes)
    components = tolisttoarray(components)
    phases  = tolisttoarray(phases)
    outputs  = tolisttoarray(outputs)
    types  = tolisttoarray(types)
    corrections  = tolisttoarray(corrections)
    R  = tolisttoarray(R)

    m=numpy.asarray([])
    a=numpy.asarray([])
    b=numpy.asarray([])
    c1=numpy.asarray([])
    c2=numpy.asarray([])
    d=numpy.asarray([])
    e=numpy.asarray([])
    s=numpy.asarray([])
    alpha=numpy.asarray([])

    for i_data,data in enumerate(R.flat):

        m = numpy.append( m,  magnitudes.flat[min([i_data, magnitudes.size-1])] )
        alpha = numpy.append( alpha,  corrections.flat[min([i_data, corrections.size-1])] )
        
        c = components.flat[min([i_data, components.size-1])]
        p =     phases.flat[min([i_data, phases.size-1])]
        o =    outputs.flat[min([i_data, outputs.size-1])]
        t =      types.flat[min([i_data, types.size-1])]
        
        a = numpy.append( a, parameters['values'][c][p][o][t][0])
        b = numpy.append( b, parameters['values'][c][p][o][t][1])
        c1 = numpy.append( c1, parameters['values'][c][p][o][t][2])
        c2 = numpy.append( c2, parameters['values'][c][p][o][t][3])
        d = numpy.append( d, parameters['values'][c][p][o][t][4])
        e = numpy.append( e, parameters['values'][c][p][o][t][5])
        s = numpy.append( s, parameters['values'][c][p][o][t][6])


    R1 = numpy.sqrt(R+9)
    
    C_M = c1*(numpy.arctan(m-5)+numpy.pi/2)* numpy.exp(c2*(m-5))

    Y = a*m - b*(R1 + C_M) - d*numpy.log10(R1 + C_M) + e + alpha

    return  numpy.reshape(Y,R.shape)/100


def tolisttoarray(input):
    if not isinstance(input, (list, tuple)):
        input = list(input)
    if not isinstance(input, numpy.ndarray):
        input = numpy.asarray(input)
    return input

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000 # Radius of earth in meters. Use 3956 for miles
    return c * r


def nsta2msize(n,nref=None):
    n = numpy.asarray(n)
    if nref:
        nref = numpy.asarray(nref)
    else:
        nref = numpy.asarray(n)
    size = (n-min(nref)+1)/(max(nref)-min(nref))*500
    
    return size

def mag_norm(mags):
    n = matplotlib.colors.PowerNorm(1,
                                    vmin=numpy.max([numpy.min(mags),0.01]),
                                    vmax=numpy.max(mags))
    return n
def depth_norm(depths):
    n = matplotlib.colors.LogNorm(vmin=numpy.max([numpy.min(depths),0.1]),
                                  vmax=numpy.max(depths))
    return n




