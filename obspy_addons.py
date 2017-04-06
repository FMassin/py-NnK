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


def get_values(self, lst, att=None, types=['b'] , full=False, pref=False, last=False, first=False, nan=True):
    """
        types = [ 'bests', 'all', 'lasts', 'firsts', 'nans' ]
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
    
    #if 'b' in type
        if hasattr(e,lst) and full and not last and not first and not pref:
            for o in e[lst]:
                if hasattr(o,att):
                    out.append(o[att])
                elif nan:
                    out.append(numpy.nan)

        elif patt and (pref or (not last and not first)):
            out.append(e[patt][att])

        elif hasattr(e,lst) and len(e[lst]) and first and not pref:
            val = e[lst][0]
            mem=9999999999999999999999999999
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if elt.creation_info.creation_time < mem:
                        mem = elt.creation_info.creation_time
                        val = elt[att]
            out.append(val)
                
        elif hasattr(e,lst) and len(e[lst]) and (last or not pref):
            val = e[lst][-1]
            mem=0
            for elt in e[lst]:
                if hasattr(elt,att) and hasattr(elt,'creation_info'):
                    if elt.creation_info.creation_time > mem:
                        mem = elt.creation_info.creation_time
                        val = elt[att]
            out.append(val)
        else:
            if nan:
                out.append(numpy.nan)
                
    return out

class Solutions():
    def __init__(self,
                 catalog=obspy.core.event.catalog.Catalog(),
                 last=0,
                 arrivals=0,
                 agency_id=['*'],
                 magnitude_type=['MVS','Mfd']):

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

            for mi,m in fn(list(enumerate(e.magnitudes))):
                if not hasattr(m.creation_info, 'creation_time'):
                    continue
                else:
                    if not m.creation_info.creation_time:
                        continue
                if (
                    (m.creation_info.agency_id in agency_id or '*' in agency_id) and
                    (m.magnitude_type in magnitude_type or '*' in magnitude_type) and
                    m.magnitude_type+m.creation_info.author not in Mtypes and
                    m.resource_id != e.preferred_magnitude_id and
                    e.preferred_magnitude_id is not None and
                    e.preferred_origin_id is not None
                    ) :
                    picks = e.picks.copy()
                    plist = []
                    #print(m)
                    for o in e.origins:
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
                            d>.001 and
                            dt>.01 and
                            dt<1000 and
                            OK_Maren == 1):
                            #print(o)
                            if m.magnitude_type is None :
                                m.station_count = o.quality.used_station_count
                            if m.mag is not None :
                                if last is not '*':
                                    Mtypes.append(m.magnitude_type+m.creation_info.author)
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
                                norm=depth_norm(profs),
                                label=types[i],alpha=.8,zorder=150,edgecolors='None')
        cb=matplotlib.pyplot.colorbar(sc)
        cb.set_label('Reference depth (km)')
        lg = matplotlib.pyplot.legend(loc=1, numpoints=1, scatterpoints=1, fancybox=True)
        lg.set_title('Marker & Sizes')
        lg.get_frame().set_alpha(0.1)
        lg.get_frame().set_color('k')
        matplotlib.pyplot.axis('equal')

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
            
            ax[1].plot([numpy.median(first),numpy.median(first)], [0,50], color=m)
            ax[1].plot([numpy.median(last),numpy.median(last)], [0,50], linestyle=':', color=m)
            
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
            
            ax[0].plot([numpy.median(first),numpy.median(first)], [0,50], color=m)
            ax[0].plot([numpy.median(last),numpy.median(last)], [0,50],linestyle=':', color=m)
            
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
        xmin=9999999
        xmax=0
        for k in solutions_first.origins_mag_delays:
            xmin = numpy.min([xmin, numpy.min(solutions_first.origins_delays[k])])
            xmax = numpy.max([xmax, numpy.max(solutions_first.origins_mag_delays[k])])
        ax2.set_xlim([1,xmax*1.1])
            #lg = matplotlib.pyplot.legend((ob1, ob2),
            #                          ('Solutions (loc. or M)', 'Picks (t or A)'),
            #                          numpoints=1,
            #                          scatterpoints=1,
            #                          fancybox=True,
            #                          loc=4)
    return f

def evfind(self=obspy.core.event.catalog.Catalog(),tofind=obspy.core.event.catalog.Catalog(),v=False,x=True):

    matchs = obspy.core.event.catalog.Catalog()
    extras = obspy.core.event.catalog.Catalog()
    matchs.description = 'Intersections of '+str(self.description)+' and '+str(tofind.description)
    missed = obspy.core.event.catalog.Catalog()
    missed.description = 'Part of '+str(self.description)+' not in '+str(tofind.description)
    listfound=list()

    for e in tofind.events:
        o = e.preferred_origin() or e.origins[-1]
        found=False
        memdl=99999999
        memdt=99999999
        if v:
            print('---- event',e.short_str(),'...')
    
        if len(listfound) < len(self.events):
            # optimize with select ?
            for i,ref in enumerate(self.events):
                
                refo = ref.preferred_origin() or ref.origins[-1]
                dt = abs(o.time-refo.time)
                dl = numpy.sqrt((o.latitude-refo.latitude)**2+(o.longitude-refo.longitude)**2)
                
                if (i not in listfound and
                    dt < 5 and dl <=.1 and
                    (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    break
                    
                    if v:
                        print('fits nicely input catalog ',tofind.description,':\n  ', ref.short_str())
            
                elif (i not in listfound and
                      dt < 60 and dl <=.5 and
                      (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    if v:
                        print('fits input catalog ',tofind.description,':\n  ', ref.short_str())

                elif (i not in listfound and
                      dt < 60 and dl <1 and
                      (dl<memdl or dt<memdt)):
                    
                    found = True
                    memi = i
                    memdl = dl
                    memdt = dt
                    if v:
                        print('poorly fits input catalog ',tofind.description,':\n  ', ref.short_str())


        if not found:
            if v:
                print('does not exist in current catalog')
            missed.events.append(e)
        elif found:
            matchs.events.append(self.events[memi]) #e)
            listfound.append(memi)
            
            matchs.events[-1].preferred_origin_id = e.preferred_origin_id
            matchs.events[-1].preferred_magnitude_id = e.preferred_magnitude_id

            for elt in e.origins: #self.events[memi].origins:
                matchs.events[-1].origins.append(elt)
            for elt in e.magnitudes: #self.events[memi].magnitudes:
                matchs.events[-1].magnitudes.append(elt)
            for elt in e.picks:
                matchs.events[-1].picks.append(elt)
            

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

def plot_eventsections(self, client_wf, afterpick = 30):

    fig = list()
    for e in self:



        picks = []
        arrivals = []
        st = obspy.core.stream.Stream()
        o = e.origins[-1]
        for co in e.origins:
            if co.resource_id == e.preferred_origin_id:
                o=co

        pmax = max([p.time for p in e.picks])
        for p in e.picks:
            if not p.waveform_id.location_code:
                p.waveform_id.location_code =''
            
            try:
                toadd = client_wf.get_waveforms(p.waveform_id.network_code,
                                                p.waveform_id.station_code,
                                                p.waveform_id.location_code,
                                                p.waveform_id.channel_code,
                                                starttime = o.time,
                                                endtime = numpy.min([60,pmax+afterpick]) )
            except:
                toadd = obspy.core.stream.Stream()
            
            break
            for t in toadd:
                found=False
                for a in o.arrivals:
                    if a.pick_id == p.resource_id:
                        t.stats.distance = a.distance*11000.
                        picks.append(p)
                        arrivals.append(a)
                        found=True
                if not found:
                    toadd.remove(t)

            st += toadd
        if len(st)>0:
            st.detrend()
            tmp=st.slice(starttime=o.time, endtime=pmax+afterpick )
            tmp.merge(method=1)

            fig.append( matplotlib.pyplot.figure() )
            tmp.plot(type='section', # starttime=o.time-10,
                     reftime=o.time,
                     time_down=True,
                     linewidth=.25,
                     grid_linewidth=.25,
                     show=False,
                     fig=fig[-1],
                     color='network')
            ax = matplotlib.pyplot.gca()
            transform = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            transform_picks = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transData)
            for i,tr in enumerate(st):
                ax.text(tr.stats.distance / 1e3, 1.0, tr.stats.station, rotation=270,
                        va="bottom", ha="center", transform=transform, zorder=10)
                ax.plot(tr.stats.distance / 1e3,
                        picks[i].time - o.time,
                        '+k',
                        transform=transform_picks,
                        zorder=10)
                ax.text(tr.stats.distance / 1e3,
                        picks[i].time - o.time,
                        str(picks[i].phase_hint),
                        weight="heavy",
                        color="k",
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        zorder=20,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=3,
                                                                        foreground="white")])
            for co in e.origins:
                ax.axhline(co.creation_info.creation_time - o.time, linestyle='--', color='k',transform=transform_picks) # horizontal lines

            l = ax.get_legend()
            la = [ text.get_text() for text in l.get_texts()]
            [line.set_linewidth(3) for line in l.get_lines()]
            li = l.get_lines()
            l.remove()
            l = ax.legend(li,la,ncol=7,prop={'size':6},title=e.short_str()+' \n '+str(e.resource_id))
            l.get_title().set_fontsize('6')
            ax.set_ylabel('Time after origin [s]')

    return fig



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




