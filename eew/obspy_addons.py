# -*- coding: utf-8 -*-
"""
source - Addon module for obspy.

This module provides additionnal functionnalities for obspy.
______________________________________________________________________

.. note::

    Functions and classes are ordered from general to specific.

"""
import obspy
from obspy import read
import matplotlib
import matplotlib.patheffects
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy
from math import radians, cos, sin, asin, sqrt
import os
import glob
#import datetime
from obspy import UTCDateTime

try:
    import catalog_addons
    import inventory_addons
except:
    try:
        import eew.inventory_addons as inventory_addons
        import eew.catalog_addons as catalog_addons
    except:
        import NnK.eew.inventory_addons as inventory_addons
        import NnK.eew.catalog_addons as catalog_addons

gold= (1 + 5 ** 0.5) / 2.


def codes2nums(data):
    #print(data)
    try :
        data[0]*1.
        nums = data
    except:
        used=list()
        nums=list()
        for code in data:
            if code not in used:
                used.append(code)
            nums.append(used.index(code))
    #print(nums)
    return nums

def rolling(a, window):
    a = numpy.asarray(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def search(self,
           fields=['longitude','latitude','elevation'],
           levels=['networks','stations']):
    
    out = list()
    for i in fields:
        out.append(list())
    
    if not levels:
        for i,a in enumerate(fields):
            if hasattr(self,a):
                out[i].append(getattr(self,a))
    else:
        for level0 in getattr(self,levels[0]) :
            if len(levels)>1 and hasattr(level0,levels[1]):
                
                for level1 in getattr(level0,levels[1]):
                    if len(levels)>2 and hasattr(level1,levels[2]):
                        
                        for level2 in getattr(level1,levels[2]):
                            if len(levels)>3 and hasattr(level2,levels[3]):
                            
                                for level3 in getattr(level2,levels[3]):
                                    if len(levels)>4 and hasattr(level3,levels[4]):
                                        print('Cannot go further than level 4')
                            
                                    elif len(levels)>4 and not hasattr(level3,levels[4]):
                                        for i,a in enumerate(fields):
                                            out[i].append(numpy.nan)
                                    else:
                                        for i,a in enumerate(fields):
                                            if hasattr(level3,a):
                                                out[i].append(getattr(level3,a))
                                            else:
                                                out[i].append(numpy.nan)
                        
                            elif len(levels)>3 and not hasattr(level2,levels[3]):
                                for i,a in enumerate(fields):
                                    out[i].append(numpy.nan)
                            else:
                                for i,a in enumerate(fields):
                                    if hasattr(level2,a):
                                        out[i].append(getattr(level2,a))
                                    else:
                                        out[i].append(numpy.nan)
                    
                    elif len(levels)>2 and not hasattr(level1,levels[2]):
                        for i,a in enumerate(fields):
                            out[i].append(numpy.nan)
                    else:
                        for i,a in enumerate(fields):
                            if hasattr(level1,a):
                                out[i].append(getattr(level1,a))
                            else:
                                out[i].append(numpy.nan)
        
            elif len(levels)>1 and not hasattr(level0,levels[1]):
                for i,a in enumerate(fields):
                    out[i].append(numpy.nan)
            else:
                for i,a in enumerate(fields):
                    if hasattr(level0,a):
                        out[i].append(getattr(level0,a))
                    else:
                        out[i].append(numpy.nan)

    return out




class Solutions():
    def __init__(self,
                 catalog=obspy.core.event.catalog.Catalog(),
                 last=0,
                 arrivals=0,
                 agency_id=['*'],
                 magnitude_type=['MVS','Mfd'],
                 nan=False,
                 minlikelyhood=0.99):

        if last is not '*':
            last = 1-last
        
        self.mags=list()
        self.depths=list()
        
        self.mags_station_counts=list()
        self.mags_errors=list()
        self.mags_types=list()
        self.mags_creation_times=list()
        self.mags_delays=list()
        self.mags_orig_errors=list()
        self.mags_lat=list()
        self.mags_lon=list()
        self.mags_orig_station_counts=list()
        
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
        
            fn=reversed
            if last:
                fn=list
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
                    (m.magnitude_type in magnitude_type or '*' in magnitude_type) and # m.resource_id != e.preferred_magnitude_id and
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
                        OK_MarenJohn=1
                        if m.magnitude_type in ['Mfd']:
                            if ('forel' in m.creation_info.author or
                                'alp' in m.creation_info.author):
                                OK_MarenJohn=0
                                if ((0.39 * o.longitude + 44) < o.latitude and
                                    'forel' in m.creation_info.author):
                                    OK_MarenJohn=1
                                elif ((0.39 * o.longitude + 44) >= o.latitude and
                                      'alp' in m.creation_info.author):
                                    OK_MarenJohn=1
                        elif m.magnitude_type in ['MVS']:
                            OK_MarenJohn=0
                            try :
                                likelyhood = float(m.comments[-1].text)
                            except:
                                likelyhood=0.
                            if minlikelyhood:
                                if o.creation_info.author:
                                    if (likelyhood >= minlikelyhood and
                                        ('NLoT_auloc' in o.creation_info.author or
                                         'NLoT_reloc' in o.creation_info.author or
                                         'autoloc' in o.creation_info.author)):
                                        OK_MarenJohn=1
                            else:
                                OK_MarenJohn=1
                    
                        try:
                            d = numpy.sqrt( d**2 + ((preferred_origin_depth-o.depth)/1000)**2)
                        except:
                            pass
                    
                        if (m.origin_id == o.resource_id and # o.resource_id != e.preferred_origin_id and
                            m.magnitude_type+m.creation_info.author+o.creation_info.author not in Mtypes  and
                            (m.magnitude_type not in ['MVS'] or ('NLoT_auloc' in o.creation_info.author or 'NLoT_reloc' in o.creation_info.author  or 'autoloc' in o.creation_info.author)) and # d>.001 and dt>.01 and dt<1000 and
                            OK_MarenJohn == 1):
                            
                            if m.magnitude_type is None :
                                m.station_count = o.quality.used_station_count
                            if m.mag is not None :
                                found = 1
                                
                                
                                # should I always use this approach ????
                                timestamp= str(m.resource_id).split('.')[-3][-14:]+'.'+str(m.resource_id).split('.')[-2]
                                timestamp = timestamp[:4]+'-'+timestamp[4:6]+'-'+timestamp[6:8]+'T'+timestamp[8:10]+':'+timestamp[10:12]+':'+timestamp[12:]
                                #m.creation_info.creation_time =  UTCDateTime(timestamp)
                                
                                timestamp= str(o.resource_id).split('.')[-3][-14:]+'.'+str(o.resource_id).split('.')[-2]
                                timestamp = timestamp[:4]+'-'+timestamp[4:6]+'-'+timestamp[6:8]+'T'+timestamp[8:10]+':'+timestamp[10:12]+':'+timestamp[12:]
                                #o.creation_info.creation_time =  UTCDateTime(timestamp)
                                
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
                                
                                self.mags_orig_station_counts.append(o.quality.used_station_count)
                                
                                merror = m.mag-preferred_magnitude_mag
                                
                                self.mags_orig_errors.append(d)
                                self.mags_errors.append(merror)
                                self.mags_types.append(m.magnitude_type)
                                self.mags_creation_times.append(m.creation_info.creation_time)
                                self.mags_lat.append(o.latitude)
                                self.mags_lon.append(o.longitude)
                                
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
                                    if len(self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author])>0:
                                        self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([o.creation_info.creation_time-preferred_origin_time, self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author][-1][1]])
                                    self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([o.creation_info.creation_time-preferred_origin_time, merror])
                                    
                                        
                                else:
                                    self.origins_mag_delays[m.magnitude_type].append(m.creation_info.creation_time-preferred_origin_time)
                                    
                                    if len(self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author])>0:
                                        self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([m.creation_info.creation_time-preferred_origin_time, self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author][-1][1]])
                                    
                                    self.mags_errorsdelays[str(e.resource_id)+o.creation_info.author].append([m.creation_info.creation_time-preferred_origin_time, merror])
                                
                                if m.magnitude_type in ['Mfd'] :
                                    self.origins_mag_station_counts[m.magnitude_type].append(o.quality.used_station_count)
                                else:
                                    self.origins_mag_station_counts[m.magnitude_type].append(m.station_count)


                                if len(self.origins_errorsdelays[str(e.resource_id)+o.creation_info.author])>0:
                                    self.origins_errorsdelays[str(e.resource_id)+o.creation_info.author].append([o.creation_info.creation_time-preferred_origin_time, self.origins_errorsdelays[str(e.resource_id)+o.creation_info.author][-1][1]])
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
                self.mags.append(numpy.nan)
                self.mags_lat.append(numpy.nan)
                self.mags_lon.append(numpy.nan)
                self.mags_orig_errors.append(numpy.nan)

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
    elif 'top' in spines:
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def sticker(t,a,x=0,y=0,ha='left',va='bottom'):
    
    o = list()
    colorin=['None','None']
    for i,c in enumerate(['w','k']):
        o.append( a.text(x,y,t,
            fontweight='bold',
            zorder=999,#            alpha=0.4,
            color=colorin[i],
            ha=ha,va=va,
            fontsize='large',
            transform=a.transAxes,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5-i, foreground=c)]))
        
    return o

def nicecolorbar(self,
                 axcb=matplotlib.pyplot.gca(),
                 reflevel=None,
                 label=None,
                 cmax=numpy.nan,
                 data=None,
                 loc='head right',
                 fontsize=8):
    
    divider = make_axes_locatable(axcb)
    # this code is from
    # http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#axes-grid1
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cb = matplotlib.pyplot.colorbar(self,
                                    cax=cax,
                                    label=label,
                                    orientation='vertical',
                                    extend='both',
                                    spacing='uniform')
        
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    if data is not None:
        levels = numpy.asarray([0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25,50,100,250,500,1000])
        level = levels[numpy.nanargmin(abs((numpy.nanmin([cmax,numpy.nanmax(data)]) - numpy.nanmin(data))/5 - levels))]
        levels = numpy.arange(numpy.nanmin(data), numpy.nanmin([cmax,numpy.nanmax(data)]), level)
        levels -= numpy.nanmin(abs(levels))
        cb.ax.set_yticks(levels)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), rotation='vertical',fontsize=fontsize)
    if reflevel:
        cb.ax.axhline((numpy.around(reflevel)-min(cb.get_clim()))/numpy.diff(cb.get_clim()),zorder=999,color='k',linewidth=2)
    return cb


def get_aspectratio(catalog=obspy.core.event.catalog.Catalog(),
                    inventory=obspy.core.inventory.inventory.Inventory([],''),
                    aspectratio=gold,
                    figsize=4):
    
    lons, lats = search(inventory, fields=['longitude','latitude'], levels=['networks','stations'])
    for x in get(catalog, 'origins','latitude', types=['p']):
        lats.append(x)
    for x in get(catalog, 'origins','longitude', types=['p']):
        lons.append(x)

    if max(lons)-min(lons) < max(lats)-min(lats):
        figsize=(figsize, figsize*aspectratio)
        aspectratio = 1/aspectratio
    else:
        figsize=(figsize*aspectratio,figsize)

    return aspectratio, lons, lats, figsize

def nicemap(catalog=obspy.core.event.catalog.Catalog(),
            inventory=obspy.core.inventory.inventory.Inventory([],''),
            aspectratio=gold,
            f=None,
            ax=None,
            alpha=1.,
            xpixels=900,
            resolution='h',
            fontsize=10,
            figsize=4,
            labels=[1,0,0,1],
            shift=0.):
    
    
    aspectratio, lons, lats, figsize = get_aspectratio(catalog=catalog,
                    inventory=inventory,
                    aspectratio=aspectratio,
                    figsize=figsize)
                    
    y = lats.copy()
    x = lons.copy()
    
    if max(lons)-min(lons) > aspectratio*(max(lats)-min(lats)):
        # Too large
        m = sum(y)/len(y)
        d = max(x)-min(x)
        lats.append(m + d/(2*aspectratio))
        lats.append(m - d/(2*aspectratio))
    else:
        # Too high
        m = sum(x)/len(x)
        d = max(y)-min(y)
        lons.append(m + d*aspectratio/2)
        lons.append(m - d*aspectratio/2)

    if ax:
        f = ax.get_figure()
    elif f:
        ax = f.add_subplot(111)
    else:
        f = matplotlib.pyplot.figure(figsize=figsize)
        ax = f.add_subplot(111)

    bmap = Basemap(llcrnrlon=min(lons)-(max(lons)-min(lons))*.1,
                 llcrnrlat=min(lats)-(max(lats)-min(lats))*.1,
                 urcrnrlon=max(lons)+(max(lons)-min(lons))*.1,
                 urcrnrlat=max(lats)+(max(lats)-min(lats))*.1,
                 epsg=4326,
                 resolution=resolution,
                     ax=ax)
    try:
        #im1 = bmap.arcgisimage(service='World_Physical_Map',xpixels=xpixels)
        im2 = bmap.arcgisimage(service='ESRI_Imagery_World_2D',xpixels=xpixels)
        #im1.set_alpha(1.*alpha)
        im2.set_alpha(0.5*alpha)
    except:
        pass

    steps=numpy.asarray([ 0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,.5,1,2.5,5,10,25,50,100])
    if aspectratio>1:
        step = steps[numpy.argmin(abs((max(lons)-min(lons))/5 - steps))]
    else:
        step = steps[numpy.argmin(abs((max(lats)-min(lats))/5 - steps))]

    bmap.resolution=resolution
    bmap.drawmeridians(meridians=numpy.arange(int(min(lons))-1,int(max(lons))+2,step),
                         labels=labels,
                         linewidth=.5,
                         color = 'w',
                         fontsize=fontsize)
    bmap.drawparallels(circles=numpy.arange(int(min(lats))-1,int(max(lats))+2,step),
                         labels=labels,
                         linewidth=.5,
                         color = 'w',
                         fontsize=fontsize)
    bmap.drawcoastlines(linewidth=.5,
                          color = 'w')
    bmap.drawstates(linewidth=1.,
                      color = 'w')
    bmap.drawcountries(linewidth=2,
                         color='w')
    bmap.drawstates(linewidth=.5)
    bmap.drawcountries(linewidth=1)
    bmap.readshapefile('/Users/fmassin/Documents/Data/misc/tectonicplates/PB2002_plates',
                         name='PB2002_plates',
                         drawbounds=True,
                         color='r')

    return f,ax,bmap


def mapall(self=None,
           others=[],
           xpixels=900,
           resolution='h',
           fontsize=8,
           alpha=.8,
           aspectratio=gold,
           markers='none',#networks',
           colors='instruments',
           label=False,
           showlegend=True,
           colorbar=True,
           title=True,
           labels=[1,0,0,1],
           titleaddons=None,
           insetfilter=None,
           fig=None,
           ax=None,
           **kwargs):
    #Inventory.plot(self, projection='global', resolution='l',continent_fill_color='0.9', water_fill_color='1.0', marker="v",s ize=15**2, label=True, color='#b15928', color_per_network=False, colormap="Paired", legend="upper left", time=None, show=True, outfile=None, method=None, fig=None, **kwargs)
    
    inventory=obspy.core.inventory.inventory.Inventory([],'')
    catalog=obspy.core.event.catalog.Catalog()
    others.append(self)
    
    for other in others:
        if isinstance(other, obspy.core.event.catalog.Catalog):
            catalog += other
        elif isinstance(other, obspy.core.inventory.inventory.Inventory):
            inventory += other


    fig, ax, bmap = nicemap(catalog=catalog,
                  inventory=inventory,
                  alpha=alpha,
                  aspectratio=aspectratio,
                  xpixels=xpixels,
                  resolution=resolution,
                  fontsize=fontsize,
                  f=fig,
                  ax=ax,
                  labels=labels,
                  shift=1/4.)
    
    titletext=''

    titletext += catalog_addons.mapevents(catalog,
                           bmap=bmap,
                           fig=fig,
                           eqcolorfield = 'depth',
                           titletext=titletext,
                           colorbar=colorbar,
                           fontsize=fontsize)

    titletext += inventory_addons.mapstations(inventory,
                             bmap=bmap,
                             fig=fig,
                             colors=colors,
                             markers=markers,
                             titletext=titletext,
                             fontsize=fontsize)




    if titleaddons:
        titletext = titleaddons+'\n'+titletext
    
    if title:
        bmap.ax.annotate(titletext,
                         xy=(0, 1),
                         xycoords='axes fraction',
                         horizontalalignment='left',
                         verticalalignment='bottom')
    if showlegend:
        fig.legend = bmap.ax.legend(prop={'size':fontsize},ncol=2, loc=3)


    if insetfilter:
        
        insetcatalog = catalog_addons.distfilter(catalog,
                                                **insetfilter)
        insetinventory = inventory_addons.distfilter(inventory,
                                                    **insetfilter)
        
        mainaspectratio, lons, lats, figsize = get_aspectratio(catalog=catalog,
                                                           inventory=inventory,
                                                           aspectratio=aspectratio)
        insetaspectratio, lons, lats, figsize = get_aspectratio(catalog=insetcatalog,
                                                           inventory=insetinventory,
                                                           aspectratio=aspectratio)
        nl=2
        nc=2
        pos=1
        if mainaspectratio>1:
            if insetaspectratio>1:
                print('HORIZONTAL & horizontal')
                width_ratios=[1, aspectratio-1]
                height_ratios = [aspectratio-1, 1]
            else:
                print('HORIZONTAL & vertical')
                width_ratios=[1*.86, aspectratio-1]
                height_ratios = [1]
                nl=1
        else:
            if insetaspectratio>1:
                print('VERTICAL & horizontal')
                nc=1
                pos=0
                width_ratios=[1,]
                height_ratios = [insetaspectratio-1,1*.85]
            else:
                print('VERTICAL & vertical')
                width_ratios=[1, insetaspectratio]
                height_ratios = [1/insetaspectratio-1,1]

        if colorbar and len(catalog.events)>0 :
            nc+=1
            width_ratios.append(.05)
            print('cb qspace')
        

        gs = matplotlib.gridspec.GridSpec(nl, nc,
                                          width_ratios=width_ratios,
                                          height_ratios=height_ratios)
        axinset = fig.add_subplot(gs[pos])
        fig, fig.axinset, fig.bmapinset = mapall(others=[insetinventory,
                                                         insetcatalog],
                                                           label=label,
                                                           xpixels=xpixels,
                                                           resolution=resolution,
                                                           fontsize=5,
                                                             labels=[0,0,0,0],#[0,1,1,0],
                                                             markers=markers,
                                                             colors=colors,
                                                           alpha=alpha,
                                                           showlegend=False,
                                                           aspectratio=aspectratio,
                                                           ax=axinset,
                                                           title=False,
                                                           colorbar=False,
                                                           **kwargs)
        insetbounds = [fig.bmapinset.boundarylons,
                       fig.bmapinset.boundarylats]
        insetbounds[0].append(fig.bmapinset.boundarylons[0])
        insetbounds[1].append(fig.bmapinset.boundarylats[0])
        bmap.plot(insetbounds[0],
                  insetbounds[1],
                  'r')
        fig.bmapinset.plot(insetbounds[0],
                           insetbounds[1],
                           'r',
                           linewidth=3)
    fig.bmap=bmap
    return fig, ax, bmap


def plot_map(self=obspy.core.event.catalog.Catalog(),t='MVS',a='Pb',color='delay',minlikelyhood=None,**kwargs):

    catalogcopy = self.copy()
    if color == 'delay':
        solutions = Solutions(catalog=self,last=0, arrivals=0, agency_id=a, magnitude_type=[t], nan=True, minlikelyhood=minlikelyhood)
        
        meme=None
        nane=None
        for i,e in enumerate(catalogcopy.events):
            if not meme:
                if solutions.mags_delays[i] > 0 and solutions.mags_delays[i] <30 :
                    meme=e
        
            if solutions.mags_delays[i]>30:
                solutions.mags_delays[i]=30.
            if solutions.mags_delays[i]<=1:
                solutions.mags_delays[i]=numpy.nan
            if solutions.mags_orig_errors[i] <= 0.001 or solutions.mags_orig_errors[i] is numpy.nan:
                solutions.mags_delays[i]=numpy.nan
            
            
            e.depth = solutions.mags_delays[i]*1000
            for o in e.origins:
                o.depth = solutions.mags_delays[i]*1000
            if solutions.mags_delays[i] is numpy.nan:
                if not nane:
                    nane = e

        indexes = numpy.argsort(solutions.mags)

        catalogcopy.events = [catalogcopy.events[i] for i in reversed(indexes)]

        if True:
            catalogcopy.events.insert(0,meme.copy())
            catalogcopy.events.insert(0,meme.copy()) #append(meme.copy())#
            catalogcopy.events[0].depth = 1.
            catalogcopy.events[0].latitude = 0.
            catalogcopy.events[0].longitude = 0.
            for o in catalogcopy.events[0].origins:
                o.depth = 1.
                o.latitude = 0.
                o.longitude = 0.
            catalogcopy.events[1].depth = 30000.
            catalogcopy.events[1].latitude = 0.
            catalogcopy.events[1].longitude = 0.
            for o in catalogcopy.events[1].origins:
                o.depth = 30000.
                o.latitude = 0.
                o.longitude = 0.
            for i in range(10):
                catalogcopy.events.insert(0,nane.copy())

    f, f.ax, f.bmap = nicemap(self,aspectratio=1.5)
    f = catalogcopy.plot(fig=f, color=color,edgecolor='k',title="", **kwargs)

    #f.texts[0].set_text(f.texts[0].get_text().replace(' - ', '\n'))
    if color in [ 'delay']:
        #f.texts[0].set_text(f.texts[0].get_text().replace(str(len(catalogcopy.events))+' ', str(len(catalogcopy.events)-12)+' '))
        #f.texts[0].set_text(f.texts[0].get_text().replace('depth', t+' '+color))
    
        for i,e in enumerate(self.events):
            for o in e.origins:
                if (o.resource_id == e.preferred_origin_id and
                    solutions.mags_lon[i] is not numpy.nan):
                    x, y = f.bmap([solutions.mags_lon[i],o.longitude],[solutions.mags_lat[i],o.latitude])
                    f.bmap.plot(x,y,'r-',zorder=99)

    tl=f.axes[1].get_xticklabels()
    [tl[i].set_text(l.get_text()+'km') for i,l in enumerate(tl)]
    f.axes[1].set_xticklabels(tl)
    min_size = 2
    max_size = 30
    mags=get(catalogcopy,'magnitudes','mag','b')
    steps=numpy.asarray([0.01,0.02,0.05,0.1,0.2,.5,1.,2.,5.])
    s = steps[numpy.argmin(abs((max(mags)-min(mags))/5-steps))+1]
    min_size_ = min(mags) - 1
    max_size_ = max(mags) + 1
    magsscale=[_i for _i in numpy.arange(-2,10,s) if _i>=min_size_ and _i<=max_size_]
    frac = [(0.2 + (_i - min_size_)) / (max_size_ - min_size_) for _i in magsscale] #mags]
    size_plot = [(_i * (max_size - min_size)) ** 2 for _i in frac]
    x = (numpy.asarray(magsscale)-min(magsscale))/(max(magsscale)-min(magsscale))*.6+.2
    f.axes[1].scatter(x,numpy.repeat(.9,len(magsscale)),s=size_plot, facecolor='None', linewidth=3, edgecolor='w')
    f.axes[1].scatter(x,numpy.repeat(.9,len(magsscale)),s=size_plot, facecolor='None', edgecolor='k')
    for i,v in enumerate(magsscale):
        f.axes[1].text(x[i],1.2,'M'+str(v), ha='center')

    #f.axes[1].set_xlabel(f.texts[0].get_text())
    #f.texts.remove(f.texts[0])
    return f


def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = numpy.sqrt(numpy.power((x2 - x1), 2)+ numpy.power((y2 - y1), 2))
    return lineMagnitude

#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
    
    if LineMag < 0.00000001:
        DistancePointLine = numpy.sqrt(numpy.power((px - x1), 2)+ numpy.power((py - y1), 2)) # 9999
        return DistancePointLine
    
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
    
    return DistancePointLine

def plot_Mfirst_hist(self=obspy.core.event.catalog.Catalog(),agency_id=['*'],log=None,minlikelyhood=None):

    solutions = Solutions(catalog=self,last=0, arrivals=0, agency_id=agency_id,minlikelyhood=minlikelyhood)
    solutions_last = Solutions(catalog=self,last=1, arrivals=0, agency_id=agency_id,minlikelyhood=minlikelyhood)
    
    f, ax = matplotlib.pyplot.subplots(2, 1, sharey=True)
    ax[0].xaxis.set_ticks_position('top')
    ax[0].xaxis.set_label_position('top')
    ax[0].set_xlabel('Error in location (km)')
    ax[1].set_xlabel('Error in magnitude')
    medsigma12=[50-34.1-13.6, 50-34.1, 50, 50+34.1, 50+34.1+13.6]
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
        
            #first=solutions.origins_errors[types[i]]
            #last=solutions_last.origins_errors[types[i]] # THIS IS NOT GIVING  SAME PLOT WHY ????
            first=[solutions.mags_orig_errors[j] for j in matches]
            last=[solutions_last.mags_orig_errors[j] for j in matches]
            
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
    ax[0].set_ylim([0,100])
    ax[1].set_ylim([0,100])
    print('set set_xlim([1,100]) and [-1.1,1.1]')
    return f

def eewtlines(self=obspy.core.event.catalog.Catalog(),last=0,agency_id=['*'],log=None,minlikelyhood=None):
    
    solutions_first = Solutions(catalog=self,last=0, arrivals=0, agency_id=agency_id,minlikelyhood=minlikelyhood)
    solutions_all = Solutions(catalog=self,last='*', arrivals=0, agency_id=agency_id,minlikelyhood=minlikelyhood)
    
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
                                                               norm=mag_norm(solutions_all.mags))#[2,5]))#
        line_segments1.set_array(numpy.asarray([solutions_all.origins_errorsdelays_mags[l] for l in solutions_all.origins_errorsdelays]))
        
        
        line_segments2 = matplotlib.collections.LineCollection([solutions_all.mags_errorsdelays[l]
                                                               for l in solutions_all.mags_errorsdelays],
                                                              linewidths=1,
                                                               linestyles='solid',zorder=1,alpha=.9,
                                                               norm=mag_norm(solutions_all.mags))#[2,5]))#
        line_segments2.set_array(numpy.asarray([solutions_all.origins_errorsdelays_mags[l] for l in solutions_all.origins_errorsdelays]))
        
        
        
        
        
        markers = {'MVS':'o','Mfd':'^'}
        for k in solutions_first.origins_delays:
            
            i_first = numpy.argsort(solutions_first.origins_mags[k])
            
            sizes = nsta2msize([solutions_first.origins_station_counts[k][i] for i in i_first],
                               [0,32])#solutions_first.mags_station_counts)
            
            ax1.scatter([solutions_first.origins_delays[k][i] for i in i_first],
                        [solutions_first.origins_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='k',facecolor='None')
            ax1.scatter([solutions_first.origins_delays[k][i] for i in i_first],
                        [solutions_first.origins_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='None',facecolor='w')
            sc1 = ax1.scatter([solutions_first.origins_delays[k][i] for i in i_first],
                              [solutions_first.origins_errors[k][i] for i in i_first],
                              sizes,
                              [solutions_first.origins_mags[k][i] for i in i_first],
                              marker=markers[k],
                              norm=mag_norm(solutions_all.mags),#[2,5]),#label=e.short_str(),
                              linewidths=0,zorder=2,
                              alpha=.8,edgecolors='None')


            sizes = nsta2msize([solutions_first.origins_mag_station_counts[k][i] for i in i_first],
                               [0,32])#solutions_first.mags_station_counts)
            
            ax2.scatter([solutions_first.origins_mag_delays[k][i] for i in i_first],
                        [solutions_first.origins_mag_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='k',facecolor='None')
            ax2.scatter([solutions_first.origins_mag_delays[k][i] for i in i_first],
                        [solutions_first.origins_mag_errors[k][i] for i in i_first],
                        sizes,
                        marker=markers[k],edgecolors='None',facecolor='w')
            sc2 = ax2.scatter([solutions_first.origins_mag_delays[k][i] for i in i_first],
                              [solutions_first.origins_mag_errors[k][i] for i in i_first],
                              sizes,
                              [solutions_first.origins_mags[k][i] for i in i_first],
                              marker=markers[k],
                              norm=mag_norm(solutions_all.mags), #[2,5]),#
                              linewidths=0,zorder=2,
                              alpha=.8,edgecolors='None')
                              
            if len(i_first) <5:
                for i in i_first:                             
                    ax1.text(solutions_first.origins_delays[k][i]-.2,
                             solutions_first.origins_errors[k][i],
                             str(solutions_first.origins_mags[k][i])[:3],
                             va='bottom',ha='right',
                             path_effects=[matplotlib.patheffects.withStroke(linewidth=3,
                                                                             foreground="white")])
                    ax2.text(solutions_first.origins_mag_delays[k][i]-.2,
                             solutions_first.origins_mag_errors[k][i],
                             str(solutions_first.origins_mags[k][i])[:3],
                             va='bottom',ha='right',
                             path_effects=[matplotlib.patheffects.withStroke(linewidth=3,
                                                                             foreground="white")])
    
        ax1.add_collection(line_segments1)
        ax2.add_collection(line_segments2)
        
        cb=matplotlib.pyplot.colorbar(line_segments1, ax=[ax1,ax2])#ax1, ax=[ax1,ax2])
        cb.set_label('Reference magnitude')
        ob2 = ax2.scatter(20,0,10,marker='o',alpha=0.1,color='b', edgecolors='none',zorder=-999)
        ob1 = ax2.scatter(20,0,10,marker='o',color='b',edgecolors='k',zorder=-999)
        ax2.scatter(20,0,10,marker='o',color='w',edgecolors='w',linewidths=3,zorder=-99)
        
        xmin=9999999
        xmax=0
        for k in solutions_first.origins_mag_delays:
            xmin = numpy.min([xmin, numpy.min(solutions_first.origins_delays[k])])
            xmax = numpy.max([xmax, numpy.max(solutions_first.origins_mag_delays[k])])
        ax2.set_xlim([1,xmax*1.1])
        ax2.set_xlim([3,30])
        ax2.set_ylim([-1.1,1.1])
        ax1.set_ylim([1,100])
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
                candidateo = candidate.preferred_origin() or candidate.origins[-1]
                if candidateo.time not in listfound:
                
                    dt = abs(o.time-candidateo.time)
                    dl = numpy.sqrt((o.latitude-candidateo.latitude)**2+(o.longitude-candidateo.longitude)**2)
                
                    if (dt < 3 and dl <=.1 and
                        (dl<memdl or dt<memdt)):
                    
                        found = True
                        memi = str(candidateo.time)
                        memdl = dl
                        memdt = dt
                        meme = candidate
                    
                        if v:
                            print('fits nicely input catalog ',tofind.description,':\n  ', candidate.short_str())
                        break
            
                    elif (dt < 8 and dl <=.4 and
                          (dl<memdl or dt<memdt)):
                    
                        found = True
                        memi = str(candidateo.time)
                        memdl = dl
                        memdt = dt
                        meme = candidate
                        if v:
                            print('fits input catalog ',tofind.description,':\n  ', candidate.short_str())

                    elif (dt < 15 and dl <.8 and
                          (dl<memdl or dt<memdt)):
                    
                        found = True
                        memi = str(candidateo.time)
                        memdl = dl
                        memdt = dt
                        meme = candidate
                        if v:
                            print('poorly fits input catalog ',tofind.description,':\n  ', candidate.short_str())
                

        if not found:
            if v:
                print('does not exist in current catalog')
            missed.events.append(e)

        elif found:
            if v:
                print('merged with ', meme.short_str())
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

            #tofind.events.remove(meme)
            
                
            

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

def plot_eventsections(self, client_wf, afterpick = 30, file = None,agencies=['*'],nminpgv=0):

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
        pm = e.magnitudes[-1]
        for cm in e.magnitudes:
            if cm.resource_id == e.preferred_magnitude_id:
                pm=cm
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
                    if (sum(abs(tr.data)) < .1 or
                        sum(abs(numpy.diff(tr.data))) < .1):
                        toadd.remove(tr)
                        print('no signal removing '+tr.stats.network+tr.stats.station+tr.stats.location+tr.stats.channel)
                    else:
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
                                toadd.attach_response(inv)
                            except:
                                print('no response removing '+tr.stats.network+tr.stats.station+tr.stats.location+tr.stats.channel)
                                toadd.remove(tr)
                        else:
                            inv = obspy.read_inventory(ifile)
                            toadd.attach_response(inv)
                        
                st += toadd
                        
                for t in st.select(id=p.waveform_id.network_code+'.'+p.waveform_id.station_code+'.'+p.waveform_id.location_code+'.'+p.waveform_id.channel_code):
                    t.stats.distance = sqrt((a.distance*110000.)**2+o.depth**2)
                
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
            #tmp.filter("highpass", freq=.5)
            #tmp.filter("lowpass", freq=25)
            
            fig.append( matplotlib.pyplot.figure() )
            
            tmp.plot(type='section', # starttime=o.time-10,
                     reftime=o.time,
                     time_down=True,
                     linewidth=.75,
                     #grid_linewidth=0.,
                     show=False,
                     fig=fig[-1],
                     color='network',
                     orientation='horizontal',
                     scale=3)
            ax = matplotlib.pyplot.gca()
            
            transform = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transAxes )
            transform_picks = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transAxes)
            
            if len(st)<5:
                for i,tr in enumerate(st):
                    ax.text(30.0, tr.stats.distance / 1e3,  tr.stats.station, #rotation=270,
                            va="bottom", ha="left",#         transform=transform,
                            zorder=10)
            
            markers = {'P':'+','S':'x','Pg':'+','Sg':'x'}
            colors = {'P':'g','S':'k','Pg':'g','Sg':'k'}
            textdone = list()
            for i,p in enumerate(picks):
                if arrivals[i].distance*110 < st[-1].stats.distance/ 1e3:
                    ax.plot(picks[i].time - o.time,
                            arrivals[i].distance*110,
                            marker=markers[str(arrivals[i].phase)],
                            color=colors[str(arrivals[i].phase)],
                            zorder=-20)
                    if str(arrivals[i].phase) not in textdone:
                        textdone.append(str(arrivals[i].phase))
                        ax.text(picks[i].time - o.time,
                                arrivals[i].distance*110,
                                str(arrivals[i].phase),
                                weight="heavy",
                                color=colors[str(arrivals[i].phase)],
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                zorder=-10,alpha=.3,
                                path_effects=[matplotlib.patheffects.withStroke(linewidth=3,
                                                                            foreground="white")])


            ax2 = fig[-1].add_subplot(311)
            ax3 = fig[-1].add_subplot(312)
            ax3r = ax3.twinx()
            pos1 = ax.get_position() # get the original position
            pos2 = [pos1.x0 , pos1.y0,  pos1.width, pos1.height *1/3.]
            ax.set_position(pos2) # set a new position

            markers = {'MVS':'o','Mfd':'^'}
            colors = {'MVS':'b','Mfd':'r'}
            plotted=list()
            plottedr=list()
            plottedl=list()
            plottedrl=list()
            plottedm=list()
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
                               'NLoT_reloc' in co.creation_info.author or
                               'autoloc' in co.creation_info.author)):
                            OK_pipeline=1
                        else:
                            OK_pipeline=0
                            if False:#cm.creation_info.agency_id in ['Pb']:
                                print('rejected:')
                                print(cm)
                                print(co)
                        

                        if (cm.magnitude_type in ['MVS', 'Mfd'] and
                            ('*' in agencies or cm.creation_info.agency_id in agencies) and
                            OK_pipeline) :
                            
                            
                            ct = max([cm.creation_info.creation_time, co.creation_info.creation_time ])
                            ax.axvline(ct - o.time, linewidth=.1, linestyle=':', color=colors[cm.magnitude_type])
                            
                            tmp=st.slice(starttime=o.time, endtime= ct )
                            phases = list()
                            for tr in tmp:
                                if tr.stats.distance/1000/3 < ct - o.time:
                                    phases.append('S-wave')
                                    #print(ct - o.time,str(tr.stats.distance/1000),': S')
                                else:
                                    phases.append('P-wave')
                                #print(ct - o.time,str(tr.stats.distance/1000),': P')
                            
                            R = [(tr.stats.distance/1000)                     for tr in tmp if tr.stats.distance/1000/3<ct - o.time]
                            PGV = numpy.asarray([numpy.max(abs(tr.data)) for tr in tmp if tr.stats.distance/1000/3<ct - o.time ])
                            PGVm = cuaheaton2007(magnitudes=[cm.mag], R = R, phases=phases)
                            PGVerror = abs(PGV - PGVm)/PGV
                            PGVerror=PGVerror[numpy.argsort(R)]
                            print(PGVerror,PGV,PGVm)
                            
                            LOCerror = haversine(o.longitude, o.latitude, co.longitude, co.latitude)/1000
                            try:
                                LOCerror = numpy.sqrt(LOCerror**2 + ((o.depth-co.depth)/1000)**2)
                            except:
                                print('no depth error in ')
                                print(co)
                            
                            ax2.scatter(numpy.tile(ct - o.time, PGV[nminpgv:].shape),
                                     PGVerror[nminpgv:],
                                        R[nminpgv:],
                                     marker=markers[cm.magnitude_type],
                                     alpha=.1,
                                     color=colors[cm.magnitude_type])

                            obl,=ax3.plot(ct - o.time,
                                         LOCerror,
                                         markers[cm.magnitude_type],
                                         color=colors[cm.magnitude_type])
                            obm,=ax3r.plot(ct - o.time,
                                           pm.mag - cm.mag,
                                           markers[cm.magnitude_type],
                                           markeredgecolor=colors[cm.magnitude_type],
                                           color='None')
                                
                            if (cm.magnitude_type not in plottedm ):
                                plottedm.append(cm.magnitude_type)
                                plotted.append(obl)
                                plottedr.append(obm)
                                plottedl.append(r'Loc$_{'+cm.magnitude_type[1:]+'}$')
                                plottedrl.append(r'Mag$_{'+cm.magnitude_type[1:]+'}$')



            ax2.xaxis.set_ticks_position('top')
            ax2.xaxis.set_label_position('top')
            ax3.xaxis.set_ticks_position('top')
            ax3.xaxis.set_label_position('top')
            minorLocator = matplotlib.ticker.MultipleLocator(1)
            majorLocator = matplotlib.ticker.MultipleLocator(5)
            ax3.xaxis.set_minor_locator(minorLocator)
            ax2.xaxis.set_minor_locator(minorLocator)
            ax3.xaxis.set_major_locator(majorLocator)
            ax2.xaxis.set_major_locator(majorLocator)
            ax2.set_xlim([0,30])
            l = ax.get_legend()
            if len(l.get_texts())>1:
                la = [ text.get_text() for text in l.get_texts()]
                [line.set_linewidth(3) for line in l.get_lines()]
                li = l.get_lines()
            else:
                li=[]
                la=[]
            l.remove()
            l = ax.legend(li,la, loc='lower right', ncol=7,prop={'size':6},title=e.short_str()+' \n '+str(e.resource_id))
            l.get_title().set_fontsize('6')
            ax.set_xlabel('Time after origin [s]')
            ax2.set_xlabel('Time after origin [s]')
            ax2.set_ylabel(r'PGV error [$/PGV$]')#cm.s^{-1}$] ')
            ax2.set_yscale('log')
            ax2.legend([],[], loc='upper left', prop={'size':6}, title='Size: distance')
            ax3.set_ylabel(r'Location error [km]')
            ax3r.set_ylabel(r'Magnitude error')
            #ax2.set_ylim([-.021,-.008])
            ax3.set_yscale('log')
            ax3.set_ylim([1,100])
            ax3.set_xlim([0,30])
            ax3r.set_ylim([-1.1,1.1])

            ax3.legend(plotted, plottedl, loc=2)
            ax3r.legend(plottedr, plottedrl, loc=1)

            ax2.yaxis.grid(True,linestyle='--')
            ax3.yaxis.grid(True,linestyle='--')
            ax3.axes.xaxis.set_ticklabels([])
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


    R1 = numpy.sqrt(R**2+9)
    
    C_M = c1*(numpy.arctan(m-5)+numpy.pi/2)* numpy.exp(c2*(m-5))

    Y = a*m - b*(R1 + C_M) - d*numpy.log10(R1 + C_M) + e + alpha

    return  numpy.reshape(Y,R.shape)#/100


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


obspy.core.event.catalog.Catalog.plot_eventsections = plot_eventsections
obspy.core.inventory.Inventory.hasdata = hasdata
obspy.core.event.catalog.Catalog.evfind = evfind
obspy.core.event.catalog.Catalog.plot_Mfirst = plot_Mfirst
obspy.core.event.catalog.Catalog.eewtlines = eewtlines
obspy.core.event.catalog.Catalog.get = get
obspy.core.event.catalog.Catalog.plot_map = plot_map

