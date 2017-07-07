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

try:
    import obspy_addons
except:
    try:
        import eew.obspy_addons as obspy_addons
    except:
        import NnK.eew.obspy_addons as obspy_addons

gold= (1 + 5 ** 0.5) / 2.


def makenetwork(self=obspy.core.inventory.inventory.Inventory([],''),
                n=10,
                start=[11.054118, -85.621419],
                end=[12.985013, -87.644974],
                network_desc={'code':'00'},
                channel_desc={'codes': ['HGE', 'HGN', 'HGZ'],
                              'azimuths': [90, 0, 0],
                              'dips': [0, 0, 90],
                              'sample_rates': [200, 200, 200],
                              'location_codes': ['00','00','00',]},
                ):



    coordinates = [numpy.linspace(start[0],end[0],n),
                   numpy.linspace(start[1],end[1],n)]

    station_addons=list()
    for i in range(n):
        channel_addons=list()
        for j in range(len(channel_desc['codes'])):
            channel_addons.append(obspy.core.inventory.channel.Channel(
                code=channel_desc['codes'][j],
                location_code=channel_desc['location_codes'][j],
                latitude=coordinates[0][i],
                longitude=coordinates[1][i],
                elevation=100.,
                depth=-100.,
                azimuth=channel_desc['azimuths'][j],
                dip=channel_desc['dips'][j],
                types=None,
                external_references=None,
                sample_rate=channel_desc['sample_rates'][j],
                sample_rate_ratio_number_samples=None,
                sample_rate_ratio_number_seconds=None,
                storage_format=None,
                clock_drift_in_seconds_per_sample=None,
                calibration_units=None,
                calibration_units_description=None,
                sensor=None,
                pre_amplifier=None,
                data_logger=None,
                equipment=None,
                response=None,
                description=None,
                comments=None,
                start_date=obspy.UTCDateTime(),
                end_date=None,
                restricted_status=None,
                alternate_code=None,
                historical_code=None,
                data_availability=None))

        station_addons.append(obspy.core.inventory.station.Station(
            code='{0:04}'.format(i),
            latitude=coordinates[0][i],
            longitude=coordinates[1][i],
            elevation=100.0,
            channels=channel_addons,
            site=None,
            vault=None,
            geology=None,
            equipments=None,
            operators=None,
            creation_date=obspy.UTCDateTime(),
            termination_date=None,
            total_number_of_channels=len(channel_desc['codes']),
            selected_number_of_channels=len(channel_desc['codes']),
            description=None,
            comments=None,
            start_date=obspy.UTCDateTime(),
            end_date=None,
            restricted_status=None,
            alternate_code=None,
            historical_code=None,
            data_availability=None))


    return self.__add__(obspy.core.inventory.network.Network(network_desc['code'],
                                         stations=station_addons,
                                         total_number_of_stations=len(station_addons),
                                         selected_number_of_stations=len(station_addons),
                                         description=None,
                                         comments=None,
                                         start_date=obspy.UTCDateTime(),
                                         end_date=None,
                                         restricted_status=None,
                                         alternate_code=None,
                                         historical_code=None,
                                         data_availability=None))

def channelmarker(s,
                  instruments={'HN':'^','HH':'s','EH':'P'},#,'BH':'*'},
                  instruments_captions={'HN':'Ac.','HH':'Bb.','EH':'Sp.'}):#,'BH':'Long p.'}):
    
    chanlist=[ c.split('.')[-1][:2] for c in s.get_contents()['channels']]
    
    for instrument_type in instruments.keys():
        if instrument_type in chanlist:
            return instruments[instrument_type], instruments_captions[instrument_type]

    return '*','others'

def get_best_instrument(self,
                        instruments_markers,
                        preforder=['HG','HN','HH','BH','EH','SH']):

    channels = self.get_contents()['channels']
    for instrument_type in preforder :
        if (instrument_type in instruments_markers.keys() and
            instrument_type in [str(cs.split('.')[-1][:2]) for cs in channels]):
            return instrument_type
    print(channels)
    return 'none'

def get_best_orientation(self,orientations_markers,
                         preforder=['N','1','Z','V']):

    channels = self.get_contents()['channels']
    for orientations_type in preforder:
        if (orientations_type in orientations_markers.keys() and
            orientations_type in [cs.split('.')[-1][-1] for cs in channels]):
            return orientations_type
    print(channels)
    return 'none'

def get_best_samplerate(self,
                        samplerates_markers):

    sample_rates, = obspy_addons.search(self, fields=['sample_rate'], levels=['networks','stations'])
    return max(sample_rates)


def make_datacaption(self,
                     dim,
                     instruments_markers,
                     instruments_captions,
                     orientations_markers,
                     orientations_captions,
                     samplerates_markers,
                     samplerates_captions):
    data = list()
    captions=list()
    for n in  self.networks:
        for s in n.stations:
            if dim == 'networks':
                data.append(n.code)
                captions.append(n.code)
            elif dim == 'locations':
                data.append(s.code)
                captions.append(s.code)
            elif dim == 'elevation':
                data.append(self.get_coordinates(s)['elevation'])
                captions.append(self.get_coordinates(s)['elevation'])
            elif dim == 'local_depth':
                data.append(self.get_coordinates(s)['local_depth'])
                captions.append(self.get_coordinates(s)['local_depth'])
            elif dim == 'instruments':
                best_instrument=get_best_instrument(s,instruments_markers)
                data.append(instruments_markers[best_instrument])
                captions.append(instruments_captions[best_instrument])
            elif dim == 'orientations':
                best_orientation=get_best_orientation(s,orientations_markers)
                data.append(orientations_markers[best_orientation])
                captions.append(orientations_captions[best_orientation])
            elif dim == 'sample_rates':
                best_samplerate=get_best_samplerate(s,samplerates_markers)
                data.append(samplerates_markers[best_samplerate])
                captions.append(samplerates_captions[best_samplerate])

    if dim == 'instruments':
        for instrument_type in instruments_markers.keys():
            data.append(instruments_markers[instrument_type])
    elif dim == 'orientations':
        for orientation_type in orientations_markers.keys():
            data.append(orientations_markers[orientation_type])

    return data,captions


def scatter6d(self,
              longitudes,
              latitudes,
              sizes,
              colors,
              markers,
              captions,
              captions_dim,
              **kwargs):
    
    norm = matplotlib.colors.Normalize()
    norm.autoscale(colors)
    cm = matplotlib.pyplot.get_cmap('nipy_spectral')
    used=list()
    for i,m in enumerate(markers):
        l=None
        if captions[i] not in used and captions_dim not in ['none']:
            n = sum([1 for c in captions if captions[i]==c])
            l = '%s (%s)' % (captions[i], n)
            used.append(captions[i])
        
        self.scatter(x=longitudes[i],
                     y=latitudes[i],
                     s=sizes[i],
                     c=cm(norm(colors[i])),
                     marker=markers[i],
                     label=l,
                     **kwargs)

def mapstations(self=obspy.core.inventory.inventory.Inventory([],''),
                bmap=None,
                fig=None,
                colors='instruments',
                markers='networks',
                sizes='none',
                titletext='',
                fontsize=8,
                instruments_markers= {'HG':'^', 'HN':'^', 'HH':'s', 'BH':'s', 'EH':'P', 'none':'*'},
                instruments_captions={'HG':'SM','HN':'SM','BH':'BB','HH':'BB','EH':'SP','none':'Other'},
                orientations_markers={'N':'^','2':'s','Z':'P','none':'*'},
                orientations_captions={'N':'triax.','2':'hori.','Z':'vert.','none':'other'},
                samplerates_markers={100:'^',40:'s','none':'*'},
                samplerates_captions={100:'Hi rate',40:'Low rate','none':'other'},
                stations_colors=None,
                stations_markers=None,
                stations_sizes=None,
                stations_colordata=None,
                stations_markerdata=None,
                stations_sizedata=None,
                filled_markers = ('^', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X','o'),
                ):
   

    stations_longitudes, stations_latitudes = obspy_addons.search(self,
                                                                  fields=['longitude','latitude'],
                                                                  levels=['networks','stations'])
    if not stations_sizedata:
        if sizes in ['none']:
            stations_sizedata = numpy.zeros(len(stations_longitudes))+15.
            stations_sizecaptions = numpy.zeros(len(stations_longitudes))+15.
        else:
            stations_sizedata, \
                stations_sizecaptions=make_datacaption(self,
                                                        sizes,
                                                        instruments_markers,
                                                        instruments_captions,
                                                        orientations_markers,
                                                        orientations_captions,
                                                        samplerates_markers,
                                                        samplerates_captions)


    if not stations_colordata:
        if colors in ['none']:
            stations_colordata = numpy.zeros(len(stations_longitudes))
            stations_colorcaptions = numpy.zeros(len(stations_longitudes))
        else:
            stations_colordata, \
                stations_colorcaptions=make_datacaption(self,
                                                        colors,
                                                        instruments_markers,
                                                        instruments_captions,
                                                        orientations_markers,
                                                        orientations_captions,
                                                        samplerates_markers,
                                                        samplerates_captions)

    if not stations_markerdata:
        if markers in ['none']:
            stations_markerdata = numpy.repeat(filled_markers[0],len(stations_longitudes))
            stations_markercaptions = numpy.repeat(filled_markers[0],len(stations_longitudes))
        else:
            stations_markerdata, \
                stations_markercaptions =make_datacaption(self,
                                                          markers,
                                                          instruments_markers,
                                                          instruments_captions,
                                                          orientations_markers,
                                                          orientations_captions,
                                                          samplerates_markers,
                                                          samplerates_captions)

    #make stations_colors colors and stations_markers markers
    if not stations_sizes:
        stations_sizes= obspy_addons.codes2nums(stations_sizedata)
    if not stations_colors:
        stations_colors= obspy_addons.codes2nums(stations_colordata,
                                                 used= [instruments_markers[k] for k in instruments_markers.keys()] )
    if not stations_markers:
        stations_markers = [ filled_markers[min([d,len(filled_markers)])] for d in obspy_addons.codes2nums(stations_markerdata) ]

    scatter6d(bmap,
              stations_longitudes,
              stations_latitudes,
              stations_sizes,
              stations_colors,
              stations_markers,
              stations_sizecaptions,
              sizes,
              facecolor='k',
              edgecolor='w',
              lw=1.5,
              )
    scatter6d(bmap,
              stations_longitudes,
              stations_latitudes,
              stations_sizes,
              stations_colors,
              stations_markers,
              stations_markercaptions,
              markers,
              facecolor='k',
              edgecolor='k',
              lw=1.,
              )
    scatter6d(bmap,
              stations_longitudes,
              stations_latitudes,
              stations_sizes,
              stations_colors,
              stations_markers,
              stations_colorcaptions,
              colors,
              edgecolor='none',
              lw=0,
              )

    times, = obspy_addons.search(self, fields=['start_date'], levels=['networks','stations'])
    if len(times)>0:
        titletext= '\n%s stations (%s to %s)' % (len(times), str(min(times))[:10], str(max(times))[:10])

    return titletext

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
                    plot='none',
                    dt=None,
                    reflevel=10.):


    arrivals,taupmodel,dmax,tmax = traveltimes(tmax=tmax,
                                               depth=depth,
                                               model=model)
        
    darrivals = numpy.asarray([a.distance for a in arrivals])
    tarrivals = numpy.asarray([a.time for a in arrivals])

    if style[0] in ['d','b'] or plot in ['c','chron','h', 'hodo','hodochrone', 's', 'sect', 'section']:
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
                dgrids[i,line,col] = distances[n-1,line,col] + avp*dt[i,line,col]

                if style[0] in ['d','b'] or plot in ['c','chron','h', 'hodo','hodochrone', 's', 'sect', 'section']:
                    closest = 1./(distances[n-1,line,col] - dSarrivals)**2.
                    closest = closest/numpy.nansum(closest)
                    avs = (numpy.nansum(dSarrivals*closest/tSarrivals))
                    
                    avsgrids[i,line,col] = avs
                    if style[0] in ['d','b']:
                        dgrids[i,line,col] = avs*tgrids[i,line,col]
                        tgrids[i,line,col] = (reflevel*avp - tgrids[i,line,col]*avs)/avs

    return tgrids,dgrids,dmax,tmax,avpgrids,avsgrids


def plot_traveltimes(self,
                     tmax=20.,
                     depth=15,
                     model='iasp91',
                     N=range(1,7),
                     fig=None,
                     ax=None,
                     style='p',
                     plot='m',
                     bits=1024,
                     reflevel=7,
                     latencies=None,
                     xpixels=900,
                     resolution='h',
                     fontsize=8,
                     dmin=[6,0.00001,999],
                     sticker_addons=None,
                     event=None,
                     mapbounds=None,
                     clims=None,
                     **kwargs):
    
    if isinstance(dmin[0], list):

        fig, (ax) = matplotlib.pyplot.subplots(len(dmin),1,sharex=True)

        # make a big axe so we have one ylabel for all subplots
        biga = fig.add_subplot(111, frameon=False)
        # turn every element off the big axe so we don't see it
        biga.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        letters='ABCDEFGH'

        for i,d in enumerate(dmin):
            if sticker_addons:
                if isinstance(sticker_addons, list):
                    sa=sticker_addons[i]
                else:
                    sa=letters[i]+'. '+sticker_addons
            else:
                sa=letters[i]+'. '
            plot_traveltimes(self,
                     tmax=tmax,
                     depth=depth,
                     model=model,
                     N=N,
                     ax=ax[i],
                     style=style,
                     plot=plot,
                     bits=bits,
                     reflevel=reflevel,
                     latencies=latencies,
                     xpixels=xpixels,
                     resolution=resolution,
                     fontsize=fontsize,
                     dmin=d,
                     sticker_addons=sa,
                             mapbounds=mapbounds,
                     **kwargs)

            if i==0:
                ax[i].legend().set_visible(False)
                if plot in ['c','chron',]:
                    # add common labels
                    biga.set_ylabel(ax[i].get_ylabel())#'Number of phases')
                    ax[i].set_ylabel('')
                    obspy_addons.adjust_spines(ax[i], ['left', 'top'])
                else:
                    biga.set_ylabel(ax[i].get_ylabel())#'Observed S travel time')
                    obspy_addons.adjust_spines(ax[i], ['left', 'bottom'])
                
            elif i==len(dmin)-1:
                if plot in ['c','chron',]:
                    ax[i].set_ylabel('')
                    obspy_addons.adjust_spines(ax[i], ['left', 'bottom'])
                else:
                    obspy_addons.adjust_spines(ax[i], ['right', 'bottom'])
            else:
                ax[i].legend().set_visible(False)
                if plot in ['c','chron',]:
                    ax[i].set_ylabel('')
                    obspy_addons.adjust_spines(ax[i], ['left'])
                else:
                    obspy_addons.adjust_spines(ax[i], ['bottom'])
            if plot in ['c', 'chron', ]:
                xmax=0
                for a in ax:
                    xmax = max([xmax, max(a.get_xlim())])
                for a in ax:
                    a.set_xlim([0, xmax])
        return fig
            
    if not latencies:
        selffiltered = self
    else:
        selffiltered = delayfilter(self,latencies)
    
    
    statlons, statlats = obspy_addons.search(selffiltered,
                                             fields=['longitude','latitude'],
                                             levels=['networks','stations'])

    names = [re.sub(r' .*', '',x) for x in selffiltered.get_contents()['stations']]

    # generate 2 2d grids for the x & y bounds
    if style[0] in ['b']:
        tmax*=1.7
    dmax=tmax*8./110.
    dlat = ((numpy.nanmax(statlats)+dmax)-(numpy.nanmin(statlats)-dmax))/numpy.sqrt(bits)
    dlon = ((numpy.nanmax(statlons)+dmax)-(numpy.nanmin(statlons)-dmax))/numpy.sqrt(bits)

    if event:
        pref = [o.resource_id for o in event.origins].index(event.preferred_origin_id)
        latitudes, longitudes = numpy.mgrid[slice(event.origins[pref].latitude,
                                                  event.origins[pref].latitude+0.0000001,
                                                  1),
                                            slice(event.origins[pref].longitude,
                                                  event.origins[pref].longitude+0.0000001,
                                                  1)]
    else:
        latitudes, longitudes = numpy.mgrid[slice(numpy.nanmin(statlats)-(dmax),
                                                  numpy.nanmax(statlats)+(dmax),
                                                  dlat),
                                            slice(numpy.nanmin(statlons)-(dmax),
                                                  numpy.nanmax(statlons)+(dmax),
                                                  dlon)]

    kmindeg = obspy_addons.haversine(numpy.nanmin(longitudes),
                               numpy.nanmin(latitudes),
                               numpy.nanmax(longitudes),
                               numpy.nanmax(latitudes))
    kmindeg /= 1000*numpy.sqrt((numpy.nanmin(longitudes)-numpy.nanmax(longitudes))**2+(numpy.nanmin(latitudes)-numpy.nanmax(latitudes))**2)
    
    dsgrid = numpy.zeros([len(statlons),latitudes.shape[0],latitudes.shape[1]])
    for col, lon in enumerate(longitudes[0,:]):
        for line, lat in enumerate(latitudes[:,0]):
            for i,s in enumerate(statlons):
                dsgrid[i,line,col] = numpy.sqrt((statlons[i]-lon)**2 + (statlats[i]-lat)**2.)


    lsgrid = numpy.zeros([len(statlons),latitudes.shape[0],latitudes.shape[1]])
    if  latencies is not None:
        latency = numpy.asarray([ numpy.nanmedian( latencies[n] ) for n in names])
        for col, lon in enumerate(longitudes[0,:]):
            for line, lat in enumerate(latitudes[:,0]):
                for i,s in enumerate(statlons):
                    closest = dsgrid[:,line,col].copy()
                    closest[closest>numpy.nanmin(dsgrid[:,line,col])]=0.
                    closest[closest==numpy.nanmin(dsgrid[:,line,col])]=1.
                    lsgrid[i,line,col] = numpy.nansum(latency*closest)

        lsgrid[lsgrid == numpy.nan] =0

    dsgrid = numpy.sort(dsgrid,axis=0)
    dmingrid = dsgrid[dmin[0]]
    dsgrid = dsgrid[:max(N),:,:]
    lsgrid = lsgrid[:max(N),:,:]

    tgrids,dgrids,dmax,tmaxupdated,avpgrids,avsgrids = traveltimesgrid(longitudes, latitudes,
                                                                dsgrid,
                                                                tmax=tmax,
                                                                depth=depth,
                                                                model=model,
                                                                N=N,
                                                                style=style,
                                                                plot=plot,
                                                                dt=lsgrid,
                                                                reflevel=reflevel)
    if latencies and plot in ['c','chron',]:
        tgrids_nodt,dgrids_nodt,dmax_nodt,tmax_nodt,avpgrids_nodt,avsgrids_nodt = traveltimesgrid(longitudes, latitudes,
                                                                    dsgrid,
                                                                    tmax=tmax,
                                                                    depth=depth,
                                                                    model=model,
                                                                    N=N,
                                                                    style=style,
                                                                    plot=plot,
                                                                    reflevel=reflevel)
    
    
    data=tgrids
    label=str(N[-1])+'$^{th}$ P travel time (s'
    ax2xlabel='Modeled travel time (s'
    if latencies and plot in ['c','chron',]:
        data_nodt=tgrids_nodt

    if style in ['d']:
        ax2xlabel='Modeled S delays (s'
        data=tgrids#+lsgrid#dgrids*110.
        if latencies and plot in ['c','chron',]:
            data_nodt=tgrids_nodt
        label='Modeled S delay at '+str(int(reflevel*numpy.nanmedian(avpgrids)*kmindeg))+'km (s'#S radius at '+str(N[-1])+'$^{th}$ P travel time [km]'
        reflevel=0.
    elif style in ['l']:
        data=lsgrid
        if latencies and plot in ['c','chron',]:
            data_nodt=lsgrids_nodt
        ax2xlabel='Data delays (s'
        label=str(N[-1])+'$^{th}$ data delays (s'
    elif style in ['b']:
        ax2xlabel='Modeled S radius (km'
        tmax=dmax*kmindeg
        reflevel*= numpy.nanmedian(avsgrids)*kmindeg
        data=dgrids*kmindeg
        if latencies and plot in ['c','chron',]:
            data_nodt=dgrids_nodt*kmindeg
        label='Modeled S radius at '+str(N[-1])+'$^{th}$ P travel time (km'#S radius at '+str(N[-1])+'$^{th}$ P travel time [km]'
    
    for i,n in enumerate(data):
        n[dmingrid*kmindeg>dmin[-1]] = numpy.nan
        n[dmingrid*kmindeg<dmin[1]] = numpy.nan

    if latencies and plot in ['c','chron',]:
        for i,n in enumerate(data_nodt):
            n[dmingrid*kmindeg>dmin[-1]] = numpy.nan
            n[dmingrid*kmindeg<dmin[1]] = numpy.nan

    if  latencies is not None :#and plot not in ['c','chron','h', 'hodo','hodochrone', 's', 'sect', 'section']:
        ax2xlabel+= ', data delays incl.)'
        label+= ', data delays incl.)'
    else:
        ax2xlabel+= ')'
        label+= ')'



    if ax :
        fig = ax.get_figure()
        fig.ax = ax
    elif fig:
        fig.ax = fig.add_subplot(111)
    else:
        fig = matplotlib.pyplot.figure()
        fig.ax = fig.add_subplot(111)
    
    if plot in ['m','map']:
        fig, fig.ax, fig.bmap = obspy_addons.mapall(others=[selffiltered],
                                                    xpixels=xpixels,
                                                    resolution=resolution,
                                                    fontsize=fontsize,
                                                    alpha=.5,
                                                    showlegend=False,
                                                    ax=ax,
                                                    mapbounds=mapbounds,
                                                    **kwargs)
        axcb = fig.bmap.ax#matplotlib.pyplot.gca()
        
        if event:
            cf= fig.bmap.tissot(numpy.median(longitudes),
                                numpy.median(latitudes),
                                1.,
                                100)
        else:
            levels = numpy.asarray([0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10,25,50,100,250,500,1000])
            if tmax:
                level = levels[numpy.nanargmin(abs((tmax - 0.)/20 - levels))]
                levels = numpy.arange( 0., tmax+level, level)
            else:
                level = levels[numpy.nanargmin(abs((numpy.nanmax(data[-1]) - numpy.nanmin(data[-1]))/20 - levels))]
                levels = numpy.arange( numpy.nanmin(data[-1])-level, numpy.nanmax(data[-1])+level, level)
            levels += -levels[numpy.nanargmin(abs(-levels))]
            levels = levels[levels>=numpy.nanmin(data[-1]-level)]
            levels = levels[levels<=numpy.nanmax(data[-1]+level)]
            
            cf= fig.bmap.contourf(x=longitudes,
                            y=latitudes,
                            data=data[-1],
                            latlon=True,
                            corner_mask=True,
                            zorder=999,
                            alpha=2/3.,
                            linewidths=0.,
                            levels=levels,
                                  vmax=tmax, vmin=0.)
            CS = fig.bmap.contour(x=longitudes,
                       y=latitudes,
                       data=data[-1],
                       latlon=True,
                                  corner_mask=True,
                       zorder=999,
                                  levels=[numpy.around(reflevel)],
                                  vmax=tmax, vmin=0.)
            matplotlib.pyplot.clabel(CS,
                                     fmt='%1.0f',
                                     inline=1,
                                     fontsize=10)

            if axcb:
               fig.cb = obspy_addons.nicecolorbar(cf,
                                                  axcb=axcb,
                                                  reflevel=reflevel,
                                                  label=label,
                                                  vmax=tmax,
                                                  vmin=0.,
                                                  data=data[-1])

    elif plot in ['h', 'hodo','hodochrone', 's', 'sect', 'section']:
        
        fig.ax.set_alpha(.5)
        fig.ax.set_ylabel('Distance')
        fig.ax.set_xlabel(ax2xlabel)
        fig.ax.grid()
        fig.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        colors='bgrcmykwbgrcmykwbgrcmykwbgrcmykw'
        for i,n in enumerate(N):
            fig.ax.scatter(data[i].flatten(),
                           data[i].flatten()*avpgrids[i].flatten(),
                           10,
                           color=colors[i])
    
    elif plot in ['c','chron',]:
        
        fig.ax.set_alpha(.5)
        fig.ax.set_ylabel('Number of phase')
        fig.ax.set_xlabel(ax2xlabel)
        fig.ax.grid()

        if sticker_addons:
            titletext='%s\ndepth: %s$_{km}$' % (sticker_addons,depth)
        else:
            titletext='Depth: %s$_{km}$'% (depth)
        if dmin[1]<999 and dmin[-1]>=999:
            titletext += '\nd$_{min}$>%s$_{km}$'% (int(dmin[1]))
        elif dmin[1]<.01 and dmin[-1]<999:
            titletext += '\nd$_{min}$<%s$_{km}$'% (int(dmin[-1]))
        elif dmin[-1]<999 and dmin[1]<999:
            titletext += '\n%s<d$_{min}$<%s$_{km}$'% (int(dmin[1]),int(dmin[-1]))
        #fig.ax.set_title(titletext)
        obspy_addons.sticker(titletext, ax, x=0, y=1, ha='left', va='top')#,fontsize='xx-small')

        fig.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.ax.set_xlim([0,min([tmax, 2*dmin[-1]/(kmindeg*numpy.nanmedian(avpgrids))])])
        minorLocator = matplotlib.ticker.MultipleLocator(1)
        fig.ax.yaxis.set_minor_locator(minorLocator)

        labels={'P':'P$_{tt}$',
                'S':'S$_{tt}$',
                's1':'$\sigma1$',
                's2':'$\sigma2$',
                's3':'$\sigma3$',
                'm':'$\widetilde{tt}$'}
        if latencies :
            labels['P']='P$_{tt}^{tt+\delta}$'
            labels['S']= 'S$_{tt}^{tt+\delta}$'
        
        for i,n in enumerate(N):
            tmp=data[i]*avpgrids[i]/avsgrids[i]
            currentSdata = numpy.sort(tmp[tmp<tmax])
            cumS = numpy.linspace(0,1,len(currentSdata))
            cumS[cumS>.5]=0.5-(cumS[cumS>.5]-0.5)
            cumS -= numpy.nanmin(cumS)
            cumS /= numpy.nanmax(cumS)

            currentdata = numpy.sort(data[i][data[i]<tmax])
            cum = numpy.linspace(0,1,len(currentdata))
            cum[cum>.5]=0.5-(cum[cum>.5]-0.5)
            cum -= numpy.nanmin(cum)
            cum /= numpy.nanmax(cum)

            yerr_s2=cum[cum>.136*2]
            yerr_s3=cum[cum<.136*2]
            yerr_s1=cum[cum>.341*2]


            fig.ax.errorbar(currentdata[cum<.136*2],
                            numpy.repeat(n,len(currentdata[cum<.136*2])),
                         yerr=[yerr_s3*0, yerr_s3/2.],
                         color='b',
                         linewidth=0,
                         elinewidth=1,
                            zorder=2,
                            label=labels['s3'])
            fig.ax.errorbar(currentdata[cum>.136*2],
                            numpy.repeat(n,len(currentdata[cum>.136*2])),
                         yerr=[yerr_s2*0, yerr_s2/2.],
                         color='g',
                         linewidth=0,
                         elinewidth=1,
                            zorder=3,
                            label=labels['s2'])
            fig.ax.errorbar(currentdata[cum>.341*2],
                         numpy.repeat(n,len(currentdata[cum>.341*2])),
                         yerr=[yerr_s1*0, yerr_s1/2.],
                         color='r',
                         linewidth=0,
                         elinewidth=1,
                            zorder=4,
                            label=labels['s1'])
            if latencies and plot in ['c','chron',]:
                currentdata_nodt = numpy.sort(data_nodt[i][data_nodt[i]*avpgrids[i]/avsgrids[i]<tmax])
                cum_nodt = numpy.linspace(0,1,len(currentdata_nodt))
                cum_nodt[cum_nodt>.5]=0.5-(cum_nodt[cum_nodt>.5]-0.5)
                cum_nodt -= numpy.nanmin(cum_nodt)
                cum_nodt /= numpy.nanmax(cum_nodt)


                tmp=data_nodt[i]*avpgrids[i]/avsgrids[i]
                currentSdata_nodt = numpy.sort(tmp[tmp<tmax])
                cumS_nodt = numpy.linspace(0,1,len(currentSdata_nodt))
                cumS_nodt[cumS_nodt>.5]=0.5-(cumS_nodt[cumS_nodt>.5]-0.5)
                cumS_nodt -= numpy.nanmin(cumS_nodt)
                cumS_nodt /= numpy.nanmax(cumS_nodt)

                yerr_s2=cum_nodt[cum_nodt>.136*2]
                yerr_s3=cum_nodt[cum_nodt<.136*2]
                yerr_s1=cum_nodt[cum_nodt>.341*2]
            else:
                currentdata_nodt = currentdata
                cum_nodt = cum
                currentSdata_nodt = currentSdata
                cumS_nodt = cumS
                if labels['P'] is not None:
                    labels['P']='tt$_P$'
                    labels['S']='tt$_S$'
            
            
            fig.ax.fill_between(currentSdata,
                                n+cumS/2,
                                n,
                                color='gray',
                                #linewidth=.5,
                                zorder=1)
            fig.ax.fill_between(currentSdata_nodt,
                                n-cumS_nodt/2,
                                n,
                                color='gray',
                                #linewidth=.5,
                                zorder=1,
                                label=labels['S'])
            
            fig.ax.errorbar(currentdata_nodt[cum_nodt<.136*2],
                            numpy.repeat(n,len(currentdata_nodt[cum_nodt<.136*2])),
                         yerr=[yerr_s3/2, yerr_s3*0],
                         color='b',
                         linewidth=0,
                         elinewidth=1,
                            zorder=2)
            fig.ax.errorbar(currentdata_nodt[cum_nodt>.136*2],
                            numpy.repeat(n,len(currentdata_nodt[cum_nodt>.136*2])),
                         yerr=[yerr_s2/2, yerr_s2*0],
                         color='g',
                         linewidth=0,
                         elinewidth=1,
                            zorder=3)
            fig.ax.errorbar(currentdata_nodt[cum_nodt>.341*2],
                         numpy.repeat(n,len(currentdata_nodt[cum_nodt>.341*2])),
                         yerr=[yerr_s1/2, yerr_s1*0],
                         color='r',
                         linewidth=0,
                         elinewidth=1,
                            zorder=4)
            
                            
            fig.ax.plot(currentdata,
                        n+cum/2,
                        color='k',
                        #linewidth=.5,
                        zorder=5,
                        label=labels['P'])
            fig.ax.plot(currentdata_nodt,
                            n-cum_nodt/2,
                        color='k',
                        #linewidth=.5,
                        zorder=5)
            labels={'P':None,
                'S':None,
                'Pdt':None,
                'Sdt':None,
                's1':None,
                's2':None,
                's3':None,
                'm':'$\widetilde{tt}$'}


        medians = [numpy.nanmedian(d[d<tmax]) for d in data]
        if latencies and plot in ['c','chron',]:
            medians_nodt = [numpy.nanmedian(d[d<tmax]) for d in data_nodt]
        else:
            medians_nodt = medians
        
        fig.ax.errorbar(medians_nodt,
                            N,
                            yerr=[numpy.repeat(0.5,len(medians_nodt)),numpy.repeat(0,len(medians_nodt))],
                            color='k',
                            linewidth=0,
                            elinewidth=1,
                        zorder=6,
                        label=labels['m'])
        fig.ax.errorbar(medians,
                        N,
                        yerr=[numpy.repeat(0,len(medians)),numpy.repeat(.5,len(medians))],
                        color='k',
                        linewidth=0,
                        elinewidth=1,
                        zorder=6)
        fig.ax.legend(fancybox=True, framealpha=0.5,ncol=2)

    return fig

def delayfilter(self=obspy.core.inventory.inventory.Inventory([],''),
                delays={'':[]},
               dmax=120,
               dmin=0.,
               save=True,
               out=False):
    d=[]
    for n in self.networks:
        for s in n.stations:

            test = numpy.nanmedian(delays[n.code+'.'+s.code])
            d.append(test)
            
            if test >=dmin and test<=dmax:
                s.code+='_inside'
            else:
                s.code+='_outside'

    inside = self.select(station='*_inside')
    outside = self.select(station='*_outside')
    
    for n in self.networks:
        for s in n.stations:
            s.code = s.code.replace("_inside", '')
            s.code = s.code.replace("_outside", '')

    for n in inside.networks:
        for s in n.stations:
            s.code = s.code.replace('_inside', '')

    for n in outside.networks:
        for s in n.stations:
            s.code = s.code.replace('_outside', '')

    if out in ['d','delay','delays']:
        return d
    else:
        if out:
            return inside, outside
        else:
            return inside


def distfilter(self=obspy.core.inventory.inventory.Inventory([],''),
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
    for n in self.networks:
        d=numpy.nan
        for s in n.stations:
            if (x1 and y1) or (x2 and y2):
                d = 110.*obspy_addons.DistancePointLine(s.longitude, s.latitude, x1, y1, x2, y2)
            
            if z1 is not None:
                d = numpy.sqrt(d**2. + (s.elevation/1000-z1)**2)
            
            distances.append(d)
            
            if d>=dmin and d<=dmax:
                s.code+='_inside'
            else:
                s.code+='_outside'

    inside = self.select(station='*_inside')
    outside = self.select(station='*_outside')
    
    for n in self.networks:
        for s in n.stations:
            s.code = s.code.replace("_inside", '')
            s.code = s.code.replace("_outside", '')

    for n in inside.networks:
        for s in n.stations:
            s.code = s.code.replace('_inside', '')

    for n in outside.networks:
        for s in n.stations:
            s.code = s.code.replace('_outside', '')

    if out in ['d','dist','distance']:
        return distances
    else:
        if out:
            return inside, outside
        else:
            return inside
