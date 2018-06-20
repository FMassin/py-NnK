# -*- coding: utf-8 -*-
"""
source - Addon module for obspy.

This module provides additionnal functionnalities for obspy.
______________________________________________________________________

.. note::

    Functions and classes are ordered from general to specific.

"""
import glob
from collections import defaultdict
import re
import numpy

import warnings
warnings.filterwarnings('ignore')
import matplotlib, obspy

import sys
sys.path.append("/Users/massin/Google Drive/Projects/NaiNo-Kami/Py/NnK")
import eew

def version():
    print('sceewenv_tests','1.2.0')

def plotenvelopediffhist2d(comp,
                           ref,
                           limit=999999999,
                           bins=99,
                           mode='rel',
                           todo = [{'location':'EA','channel':'Z','gm':'Acc','unit':r'm/s$^2$'},
                                   {'location':'EV','channel':'Z','gm':'Vel','unit':r'm/s'},
                                   {'location':'ED','channel':'Z','gm':'Disp','unit':r'm'}]):

    f = matplotlib.pyplot.figure(figsize=(12,12))
    axes = f.subplots(3,2)#,sharex=True)

    locations=[numpy.asarray([str(tr.stats.location) for tr in comp]),
               numpy.asarray([str(tr.stats.location) for tr in ref])]
    orientations=[numpy.asarray([str(tr.stats.channel[-2]) for tr in comp]),
                  numpy.asarray([str(tr.stats.channel[-1]) for tr in ref])]
    outliers=[]
    for itd,td in enumerate(todo):
        
        mask=(locations[0]==td['location'])&(orientations[0]==td['channel'])
        stream = [comp[m] for m in numpy.where(mask)[0] ] #comp.select(location=td['location'])
        
        mask=(locations[1]==td['location'])&(orientations[1]==td['channel'])
        refstream = [ref[m] for m in numpy.where(mask)[0] ] #ref.select(location=td['location'])
        

        n=0
        diffs=[]
        env=[]
        times=[]

        ids=numpy.asarray([str(tr.id) for tr in refstream])
        starttimes=numpy.asarray([tr.stats.starttime.timestamp for tr in refstream])
        endtimes=numpy.asarray([tr.stats.endtime.timestamp for tr in refstream])
        
        for trace in stream:
            mask = (str(trace.id[:-1]) == ids)&(tr.stats.starttime.timestamp<endtimes)&(tr.stats.endtime.timestamp>starttimes)
            reftrace = [refstream[m] for m in numpy.where(mask)[0] ]
            
            if False:
                reftrace = select(refstream,
                                  endafter=trace.stats.starttime,
                                  startbefore=trace.stats.endtime,
                                  id=trace.id[:-1])
                              
            for rtrace in reftrace:
                
                iref, itr, starttime, npts = overlap(rtrace,trace)
                mask= numpy.isnan(trace.data) | numpy.isnan(rtrace.data)
                mask=[not i for i in mask]
                iref=iref&mask
                itr=itr&mask
                [diffs.append(s) for s in trace.data[itr]]
                [env.append(s) for s in rtrace.data[iref]]
                [times.append(s) for s in rtrace.times("utcdatetime")[iref]]

            if len(diffs)>=limit:
                print('REACHED LIMIT!')
                break

        for i in [0,1]:
            axes[itd,i].grid()
            axes[itd,i].set_yscale('log')
            axes[itd,i].set_ylabel('%s scenvelope (%s)'%(td['gm'],td['unit']))
            if i == 1:
                d=100.*numpy.asarray(diffs)/numpy.asarray(env)
                outliers.append(numpy.asarray(times)[(d>90.)&(numpy.asarray(env)>max(env)*.9)])
                h,x,y,im=axes[itd,i].hist2d(d,env,
                                    bins=[numpy.linspace(min(d),max(d),bins),
                                          numpy.logspace(numpy.log10(min(env)),numpy.log10(max(env)),bins)],
                                            norm=matplotlib.colors.LogNorm(),
                                            normed=True)
                im.set_data(x,y,im.get_array()*1./sum(sum(h)))
                im.set_norm(matplotlib.colors.LogNorm())
                axes[itd,i].set_xlabel('%s diff (%s)'%(td['gm'],'%'))

            else:
                h,x,y,im=axes[itd,i].hist2d(diffs,env,
                                    bins=[numpy.linspace(min(diffs),max(diffs),bins),
                                          numpy.logspace(numpy.log10(min(env)),numpy.log10(max(env)),bins)],
                                            norm=matplotlib.colors.LogNorm(),
                                            normed=True)

                im.set_data(x,y,im.get_array()*1./sum(sum(h)))
                im.set_norm(matplotlib.colors.LogNorm())
                axes[itd,i].set_xlabel('%s diff (%s)'%(td['gm'],td['unit']))

            cb=matplotlib.pyplot.colorbar(im,ax=axes[itd,i])
            cb.ax.set_ylabel('Probability')
    f.tight_layout()
    #from obspy.clients.fdsn import Client
    client = Client("http://arclink.ethz.ch:8080")
    events = obspy.core.event.catalog.Catalog()
    for out in outliers:
        for o in out:
            for e in client.get_events(starttime=o-120, endtime=o+120):
                if e not in events:
                    events+=e

    return f,events

def histenvelopediffs(differences,
                      bwidths,
                      axe,
                      units,
                      xlabels,
                      orientations,
                      x,
                      maxy):
    
    ymin=2./len(differences)
    
    # the histogram of the data
    results, edges = numpy.histogram(differences, bwidths, normed=True)
    binWidth = numpy.diff(edges)
    maxy=max([max(results*binWidth), maxy])


    axe.fill_between(edges[:-1],
                       results*binWidth,
                       ymin)
    axe.plot(edges[:-1],
               results*binWidth,
              lw=.5)
    
    axe.set_xlim(left=min(edges[:-1][(results*binWidth) >ymin])-numpy.mean(binWidth))
    axe.set_xlim(right=max(edges[:-1][(results*binWidth)>ymin])+numpy.mean(binWidth))


    results[abs(edges[:-1]-numpy.median(differences))>1.1*numpy.std(differences)]=-10.
    axe.fill_between(edges[:-1],
                       results*binWidth,
                       ymin,
                       label=r'$\sigma$=%.2E%s'%(numpy.std(differences),units))#, binWidth)
    axe.plot(edges[:-1],
               results*binWidth)

    axe.plot(numpy.median(differences),
               max(results*binWidth),
               'x',
               label=r'$\mu$=%.2E%s'%(numpy.median(differences),units))#, binWidth)

    axe.set_xlabel(xlabels)
    axe.set_ylabel('Probability')
    #axe.set_title()
    axe.legend(title=r'%s %s'%(orientations,xlabels))
    axe.grid()
    #axe.set_ylim(top=maxy)
    axe.set_ylim(bottom=ymin*1.1)
    axe.set_yscale('log')

    return maxy



def plotenvelopediffs(stream,
                      splitby='none',
                      channel='*d',
                      excludedorientationcode='C',
                      xlabels={'EA':r'Acc. diff. (m/$s^2$)',
                               'EV':'Vel. diff. (m/s)',
                               'ED':'Disp. diff. (m)',
                               'UTC':'Timestamp diff. (s)'},
                      orientations={'d':'Horizontal',
                                    'Zd':'Vertical'},
                      units={'EA':r'm/$s^2$',
                             'EV':r'm/s',
                             'ED':r'm',
                             'UTC':r's'},
                      bwidths={'EA':999,
                               'EV':999,
                               'ED':999,
                               'UTC':numpy.arange(-20,20,1)},
                      maxy=0):
    differences = {}
    
    if splitby in 'Nonenone':
        for tr in  stream.select(channel=channel):
            if tr.stats.channel[2]  in excludedorientationcode:
                continue
            data=[d for d in tr.data if not isinstance(d,obspy.UTCDateTime) and  not numpy.isnan(d) ]
            if tr.stats.location in differences :
                differences[tr.stats.location] = numpy.append(differences[tr.stats.location],data)
            else:
                differences[tr.stats.location] = data
        f = matplotlib.pyplot.figure(figsize=(12,12))
        ax = f.subplots(int(len(differences)**.5),
                        int(len(differences)/len(differences)**.5),
                       #sharey=True
                       )
        ax=ax.flatten()
        for i,x in enumerate(differences):
            if not isinstance(differences[x][0],obspy.UTCDateTime):
                maxy = histenvelopediffs(differences = differences[x],
                                         bwidths=bwidths[x],
                                         axe=ax[i],
                                         units=units[x],
                                         xlabels=xlabels[x],
                                         orientations='',
                                         x=x,
                                         maxy=maxy)

    elif 'rientation' in splitby:
        for tr in  stream.select(channel=channel):
            data=[d for d in tr.data if not isinstance(d,obspy.UTCDateTime) and not numpy.isnan(d)]
            if not tr.stats.location in differences :
                differences[tr.stats.location]={}
            if tr.stats.channel[2]  in excludedorientationcode:
                continue
            if not tr.stats.channel[2:] in differences[tr.stats.location] :
                differences[tr.stats.location][tr.stats.channel[2:]] = data
            else:
                differences[tr.stats.location][tr.stats.channel[2:]] = numpy.append(differences[tr.stats.location][tr.stats.channel[2:]],data)

        f = matplotlib.pyplot.figure(figsize=(12,15))
        ax = f.subplots(4,2)

        for i,x in enumerate(differences):
            for j,y in enumerate(differences[x]):

                maxy = histenvelopediffs(differences = differences[x][y],
                                         bwidths=bwidths[x],
                                         axe=ax[i][j],
                                         units=units[x],
                                         xlabels=xlabels[x],
                                         orientations=orientations[y],
                                         x=x,
                                         maxy=maxy)


def combinehoriz(EN,
                 horizontal_code='h'):
    #starttime = max([tr.stats.starttime for tr in EN])
    #endtime = min([tr.stats.endtime for tr in  EN])
    #EN = EN.slice(max(starttimes),min(endtimes))
    #npts = [tr.stats.npts for tr in EN]
    iref, itr, starttime, npts = overlap(EN[0],EN[1])
    tr = obspy.core.stream.Trace(header=EN[0].stats) #EN[0].copy()
    tr.stats.channel = tr.stats.channel[:-1]+horizontal_code
    tr.stats.npts = npts
    tr.stats.startime = starttime
    if isinstance(EN[0].data[0] , obspy.UTCDateTime):
        tr.data = numpy.max([EN[0].data[iref],
                             EN[1].data[itr]],
                            axis=0)
    else:
        tr.data = (EN[0].data[iref]**2+EN[1].data[itr]**2)**.5
    return EN, tr

def overlap(reftrace,
            trace):
    starttime = max([reftrace.stats.starttime , trace.stats.starttime])
    endtime =  min([reftrace.stats.endtime , trace.stats.endtime])
    if starttime>=endtime:
        print('WARNING! No overlaping data in')
        print(starttime+starttime)
        return [],[],starttime,0
    iref = (reftrace.times(reftime=starttime)>=0)&(reftrace.times(reftime=endtime)<=0)
    itr = (trace.times(reftime=starttime)>=0)&(trace.times(reftime=endtime)<=0)
    return iref, itr, starttime, sum(iref)

def select(stream,
           startafter=None,
           endbefore=None,
           startbefore=None,
           endafter=None,
           **kargs):
    out = stream.select(**kargs)
    if startafter is not None:
        out = [tr for tr in out if tr.stats.starttime>=startafter]
    if endbefore is not None:
        out = [tr for tr in out if tr.stats.endtime<=endbefore]
    if startbefore is not None:
        out = [tr for tr in out if tr.stats.starttime<startbefore]
    if endafter is not None:
        out = [tr for tr in out if tr.stats.endtime>endafter]
    return obspy.core.Stream(out)

def combinechannels(stream = obspy.core.Stream(),
                    combine = 'all',
                    horizontal_code = 'b',
                    tridimentional_code = 't',
                    difference_code = 'd',
                    ref = obspy.core.Stream(),
                    max_code = 'm',
                    percentile=False,
                    verbose=False,
                    limit=9999999):
    dontdo=[]
    tridim = obspy.core.stream.Stream()
    horiz = obspy.core.stream.Stream()
    diff = obspy.core.stream.Stream()
    for trace in stream:
        if trace.id[:-1] not in dontdo:
            dontdo.append(trace.id[:-1])
            if combine in 'differences':
                if len(diff)>limit:
                    print('REACHED LIMIT!!!')
                    return diff
                #reftrace=ref.select(id=trace.id)
                reftrace = select(ref,
                                  startafter = trace.stats.starttime,
                                  endbefore = trace.stats.endtime,
                                  id=trace.id)
                if (not len(reftrace) and
                    len(trace.stats.channel)==2):
                    E = select(ref,
                               startafter = trace.stats.starttime,
                               endbefore = trace.stats.endtime,
                               id=trace.id+'[E,2]')
                    N = select(ref,
                              startafter = trace.stats.starttime,
                              endbefore = trace.stats.endtime,
                              id=trace.id+'[N,3]')
                    #E = ref.select(id=trace.id+'[E,2]').slice(trace.stats.starttime,trace.stats.endtime)
                    #N = ref.select(id=trace.id+'[N,3]').slice(trace.stats.starttime,trace.stats.endtime)
                    try:
                        EN, reftrace = combinehoriz(obspy.core.stream.Stream([E[0], N[0]]), horizontal_code='')
                    except:
                        if verbose:
                            print('WARNING! No reference streams found in ref for ')
                            print(trace)
                        continue
                else:
                    try:
                        reftrace=reftrace[0]
                    except:
                        if verbose:
                            print('WARNING! No reference stream found in ref for ')
                            print(trace)
                        continue
                if not len(reftrace.data) :
                    if verbose:
                        print('WARNING! No reference trace found in ref for ')
                        print(trace)
                    continue
                        
                iref, itr, starttime, npts = overlap(reftrace,trace)
                tr = obspy.core.stream.Trace(header=trace.stats) #reftrace.copy()
                tr.stats.starttime = starttime
                tr.stats.npts = npts
                tr.stats.channel = tr.stats.channel+difference_code
                
                if isinstance(trace.data[0] , obspy.UTCDateTime):
                    tr.data = numpy.asarray([ reftrace.data[iref][i]-d for i,d in enumerate(trace.data[itr])])
                else:
                    tr.data = reftrace.data[iref]-trace.data[itr]
                    if percentile:
                        tr.data[reftrace.data<numpy.sort(reftrace.data)[int(len(reftrace.data)*percentile)]]=numpy.nan
                diff += tr
            else:
                Z = stream.select(id=trace.id[:-1]+'[Z,1]').slice(trace.stats.starttime,trace.stats.endtime)
                E = stream.select(id=trace.id[:-1]+'[E,2]').slice(trace.stats.starttime,trace.stats.endtime)
                N = stream.select(id=trace.id[:-1]+'[N,3]').slice(trace.stats.starttime,trace.stats.endtime)

                if combine in 'horizontal2dimensionaltwodimensionalboth' and E and N:
                    EN,tr = combinehoriz(E+N,
                                         horizontal_code=horizontal_code)
                    horiz += EN
                    horiz += tr

                if Z and E and N and  combine in 'all3dimensionaltridimensionalboth' :
                    starttimes = [tr.stats.starttime for tr in E+N+Z]
                    endtimes = [tr.stats.endtime for tr in E+N+Z]
                    ZEN = (Z+E+N).slice(max(starttimes),min(endtimes))
                    npts = [tr.stats.npts for tr in ZEN]
                    tridim += ZEN
                    tr = ZEN[0].copy()
                    tr.stats.channel = tr.stats.channel[:-1]+tridimentional_code
                    tr.data = (ZEN[0].data[:min(npts)]**2+ZEN[1].data[:min(npts)]**2+ZEN[2].data[:min(npts)]**2)**.5
                    tridim += tr

    if combine in 'both':
        return tridim , horiz
    elif combine in 'all3dimensionaltridimensional':
        return tridim
    elif combine in 'differences':
        return diff
    return horiz

def remove_response(eventstreams,
                    pre_filt=[0.1, 0.33, 45, 50],
                    water_level=60,
                    **detrend_options):
    # remove responses
    
    if 'type' not in detrend_options:
        detrend_options['type']='polynomial'
    if  detrend_options['type']=='polynomial' and 'order' not in detrend_options:
        detrend_options['order']=3

    eventstreams['acc'] = eventstreams['raw'].copy()
    eventstreams['vel'] = eventstreams['raw'].copy()
    eventstreams['disp'] = eventstreams['raw'].copy()

    for output in ['acc','vel','disp']:
        eventstreams[output].output = output
        eventstreams[output].correction_method='remove_response'
        eventstreams[output].detrend(**detrend_options)#, plot=True)
        eventstreams[output].remove_response(pre_filt=pre_filt,
                                             water_level=water_level,
                                             output=output) # plot=True,
    return eventstreams

def remove_sensitivity(eventstreams,
                       filters={'pre':None,#{'type':'highpass','freq':0.075},
                                'acc':None,
                                'vel':None,
                                'disp':{'type':'highpass','freq':1/3.}},
                       integrate_method='cumtrapz',
                       differentiate_method='gradient',
                       sceewenv=False,
                    **detrend_options):
    # remove sensitivity
    
    
    if 'type' not in detrend_options:
        detrend_options['type']='polynomial'
    if  detrend_options['type']=='polynomial' and 'order' not in detrend_options:
        detrend_options['order']=4
    
    eventstreams['acc'] = eventstreams['raw'].copy()
    eventstreams['vel'] = eventstreams['raw'].copy()
    eventstreams['disp'] = eventstreams['raw'].copy()
    if sceewenv:
        tmp = eventstreams['raw'].copy()
        tmp.detrend(**detrend_options)
        tmp.filter(type='lowpass',freq=1/60.)
    
    for output in ['acc','vel','disp']:

        eventstreams[output].output = output
        eventstreams[output].correction_method='remove_sensitivity'
        if sceewenv:
            for tri,tr in enumerate(eventstreams[output]):
                tr.data = tr.data*1.
                tr.data[:len(tr.data)-2] -= tmp[tri].data[:len(tr.data)-2]
        
        if filters['pre']:
            eventstreams[output].detrend(**detrend_options)#, plot=True)
            eventstreams[output].filter(**filters['pre'])
        eventstreams[output].remove_sensitivity()

        for trace in eventstreams[output]:
            if 'M/S**2' == trace.stats.response.get_paz().input_units:
                if output in ['acc']:
                    pass
                elif output in ['vel']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)
                elif output in ['disp']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)
                    trace.taper(.05,side='left')
                    trace.detrend(**detrend_options)#, plot=True)
                    trace.integrate(method=integrate_method)

            elif 'M/S' == trace.stats.response.get_paz().input_units:
                if output in ['acc']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.differentiate(method=differentiate_method)
                elif output in ['vel']:
                    pass
                elif output in ['disp']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)

            elif 'M' == trace.stats.response.get_paz().input_units:
                if output in ['acc']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)
                elif output in ['vel']:
                    eventstreams[output].detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)
                    trace.detrend(**detrend_options)#, plot=True)
                    trace.taper(.05,side='left')
                    trace.integrate(method=integrate_method)
                elif output in ['disp']:
                    pass
            else:
                print('WARNING: unknown units for trace:')
                print(trace)

        #eventstreams[output].detrend(**detrend_options)#, plot=True)
        if filters[output]:
            eventstreams[output].filter(**filters[output])
            
            
    return eventstreams

def fdsnws(base_url="http://arclink.ethz.ch:8080",
                      endafter=40.,
                      maxradius=.6,
                      location='*',
                      channel='HNZ,HNE,HNN,HGZ,HGE,HGN,HHZ,HHE,HHN,EHZ,EHE,EHN,SHZ,SHE,SHN',
                      stations_base_url=None,
                      waveforms_base_url=None,
                      quality=None,
                      minimumlength=None,
                      longestonly=None,
           correction_method = remove_sensitivity,
         eventid=None,
                      **get_events_options):
    
    
    # First import :
    from obspy.clients.fdsn import Client
    fdsnclient = Client(base_url)
    
    # eventid in URL case
    if eventid is  None:
        eventid = 'smi:ch.ethz.sed/sc3a/2017epaqsp'
        print('Picks default eventid:',eventid)
    elif '#' in eventid :
        eventid = eventid.split('#')[-1]
        print('Picks eventid in URL format:',eventid)
    
    # Special clients systems
    stationsclient = fdsnclient
    waveformsclient = fdsnclient
    if stations_base_url:
        stationsclient = Client(stations_base_url)
        if not waveforms_base_url:
            waveformsclient = Client(stations_base_url)
    if waveforms_base_url:
        waveformsclient = Client(waveforms_base_url)
        if not stations_base_url:
            stationsclient = Client(waveforms_base_url)
    


    # Load event
    fdsnclient.get_events(eventid=eventid,format='sc3ml',filename='events.xml', **get_events_options)
    eventstreams = {'catalog': obspy.read_events('events.xml',format='sc3ml'),
                    'inventory': obspy.core.inventory.Inventory([],None),
                    'raw' : obspy.core.Stream()}
    if eventstreams['catalog'] is None:
        print('catalog is',eventstreams['catalog'])
    for output in ['catalog','inventory','raw']:
        eventstreams[output].output=output
    
    for event in eventstreams['catalog'].events :
        
        # Load stations
        t=event.preferred_origin().time
        try:
            inventory = stationsclient.get_stations(level='station',
                                                startbefore=t,
                                                endafter=t+endafter,
                                                latitude=event.preferred_origin().latitude,
                                                longitude=event.preferred_origin().longitude,
                                                maxradius=maxradius,
                                                location=location,
                                                channel=channel)
        except:
            print('No station found for event:')
            print(event)
            print('Using client:')
            print(stationsclient)
            continue
        # Load waveforms
        addons = [location, channel] + [t,t+endafter]
        bulk = [tuple(station.split()[0].split('.')[:2]+addons) for station in inventory.get_contents()['stations']]
        try:
            waveforms = waveformsclient.get_waveforms_bulk(bulk,
                                                      attach_response=True,
                                                      quality=quality,
                                                      minimumlength=minimumlength,
                                                      longestonly=longestonly)
        except:
            print('No waveform found for request:')
            print(bulk)
            print('Using client:')
            print(waveformsclient)
            continue
        # Improve waveforms attributes
        for trace in waveforms:
            station = inventory.select(network=trace.stats.network,
                                       station=trace.stats.station).networks[0].stations[0]
            trace.stats.coordinates = {'latitude':station.latitude,
                                       'longitude':station.longitude,
                                       'elevation':station.elevation}
            distance = obspy.geodetics.base.gps2dist_azimuth(station.latitude,
                                                             station.longitude,
                                                             event.preferred_origin().latitude,
                                                             event.preferred_origin().longitude)[0]
            distance = ((distance**2+(trace.stats.coordinates['elevation']*-1)**2.)**.5)
            distance = distance/len(eventstreams['catalog'].events)
            if not hasattr(trace.stats, 'distance'):
                trace.stats.distance = 0.
            trace.stats.distance += distance

        eventstreams['inventory'] += inventory
        eventstreams['raw'] += waveforms

    eventstreams['raw'].sort(keys=['distance'])

    if correction_method:
        eventstreams = correction_method(eventstreams)

    return eventstreams



# load envelope reader
def vs_envelope_read(scvsmag_processing_info,
                  maxcount=90000000,
                  sampling_rate=1.,
                 scenvelope=False,
                     realstream = obspy.core.Stream()):
    """
    Evaluate the envelope log file that is produced by scvsmag.
    """
    
    from statistics import mode
    
    stream = defaultdict(dict)
    cnt = 0
    first = 9999999999999999 #None
    last = 0 #None
    for file in glob.glob(scvsmag_processing_info):
        f = open(file)
        while True:
            line = f.readline()
            if not line: break
            if cnt > maxcount:
                print('WARNIING !!! Maximum line count exceeded.')
                break
            
            pat = r'(\S+ \S+) \[envelope/info/VsMagnitude\] Current time: (\S+);'
            pat += r' Envelope: timestamp: (\S+) waveformID: (\S+) (\S+): (\S+) (\S+): (\S+) (\S+): (\S+)'
            match = re.search(pat, line)
            channels = {'vel':'.EV','acc':'.EA','disp':'.ED'}
            if match:
                pass
            else:
                pat = r'(\S+ \S+) \[envelope/info/VsMagnitude\] Current time: (\S+);'
                pat += r' Envelope: timestamp: (\S+) waveformID: (\S+) (\S+): (\S+) (\S+): (\S+)'
                match = re.search(pat, line)
                if match:
                    pass
                else:
                    pat = r'(\S+ \S+) \[envelope/info/VsMagnitude\] Current time: (\S+);'
                    pat += r' Envelope: timestamp: (\S+) waveformID: (\S+) (\S+): (\S+)'
                    match = re.search(pat, line)
                    if match:
                        pass
                    else:
                        print("problem with line %d" % cnt)
                        print( line)
                        continue
        
            ttmp = match.group(1)
            dt, t = ttmp.split()
            year, month, day = map(int, dt.split('/'))
            hour, min, sec = map(int, t.split(':'))
            logtime = obspy.UTCDateTime(year, month, day, hour, min, sec)
            currentTime = obspy.UTCDateTime(match.group(2))
            # the timestamp marks the beginning of the data window
            timestamp = obspy.UTCDateTime(match.group(3))
            if scenvelope:
                timestamp += 1./sampling_rate
            ts_string = timestamp.datetime#strftime("%Y-%m-%dT%H:%M:%S")
            wID = match.group(4)
            station = wID.split('.')[0] + '.' + wID.split('.')[1]
            net = wID.split('.')[0]
            stream[ station+'.UTC'+'.'+wID.split('.')[2] ][ts_string] = currentTime
            for i in [5,7,9]:
                try:
                    stream[ station+channels[match.group(i)]+'.'+wID.split('.')[2] ][ts_string] = float( match.group(i+1) )
                except:
                    pass
            #if cnt == 0:
            first = numpy.min([first, timestamp])
            #if cnt == maxcount-1:
            last = numpy.max([last, timestamp])
            cnt += 1
        f.close()
    print(cnt,'lines read')
    

    for key in stream.keys():
        timekeys = list(stream[key].keys())
        sortstimekeys = numpy.argsort( timekeys )
        data = [ stream[key][timekeys[i]] for i in sortstimekeys ]
        time = [ timekeys[i] for i in sortstimekeys ]
        lasti=0
        timediffs = numpy.diff(time)
        timediffs = [x.total_seconds() for x in timediffs]
        try:
            test = mode(timediffs)
        except:
            continue
        if sampling_rate != test and len(timediffs)>2:
            sampling_rate = mode(timediffs)
            print('WARNING!!! Found sampling_rate: ', sampling_rate)
        
        for i,x in enumerate(timediffs):
            if x != sampling_rate or i+2==len(time):
                header = obspy.core.Stats()
                header.network = key.split('.')[0]
                header.station = key.split('.')[1]
                header.location = key.split('.')[2]
                header.channel = key.split('.')[3]
                header.sampling_rate = sampling_rate
                header.starttime = obspy.UTCDateTime(time[lasti])
                
                header.npts = len(data[lasti:i+1])
                realstream.append(obspy.core.Trace(data = numpy.asarray( data[lasti:i+1] ),
                                                   header = header))
                lasti=i+1
    realstream._cleanup()
    print(numpy.sum([tr.stats.npts for tr in realstream]),'samples in stream')
    return realstream



# load stream and envelope plotter
def plotstreamsenvelopes(eewenvelopes={},
                         acc=obspy.core.Stream(),
                         vel=obspy.core.Stream(),
                         disp=obspy.core.Stream(),
                         spd=[3,3],
                         outputs={'acc':'A','vel':'V','disp':'D'},
                         units={'acc':'m/s/s','vel':'m/s','disp':'m'},
                         raw=obspy.core.Stream(),
                         sensible=obspy.core.Stream(),
                         channel='*Z',
                         catalog=None,
                         inventory=None,
                         names=['current'],
                         style='fast',
                        # style='seaborn-muted',
                        # style='seaborn-poster',
                        # style='seaborn-deep',
                        # style='seaborn-bright',
                        # style='seaborn-dark-palette',
                        # style='seaborn-dark',
                        # style='seaborn-white',
                        # style='seaborn-talk',
                        # style='seaborn-notebook',
                        # style='seaborn-colorblind',
                        # style='seaborn-whitegrid',
                        # style='seaborn-paper',
                        # style='seaborn-ticks',
                        # style='seaborn-pastel',
                        # style='seaborn-darkgrid',
                        # style='seaborn',
                        # style='fivethirtyeight',
                        # style='_classic_test',
                        # style='Solarize_Light2',
                        # style='bmh',
                        # style='classic',
                        # style='ggplot',
                        # style='tableau-colorblind10',
                        # style='grayscale',
                        # style='dark_background',
                        # style='fast'
                         ):
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    fe = obspy.geodetics.flinnengdahl.FlinnEngdahl()
    
    with matplotlib.pyplot.style.context((style)):
        
        for event in catalog.events:
            forlegend={'o':[],'l':[]}
            fig = matplotlib.pyplot.figure(dpi=100)
            fig.set_size_inches([fig.get_size_inches()[0]*1.5, 1.5*fig.get_size_inches()[0]*5*spd[0]/3./8.])
            ppwaveforms=[acc,vel,disp]
            ax=[]
            for si,s in enumerate(ppwaveforms):
                ax.append([])
                for ti,t in enumerate(s.select(channel=channel)):
                    ax[si].append(None)
                    ax[si][ti]=fig.add_subplot(spd[0],
                                       spd[1],
                                       ti*spd[1]+(si+1),
                                              #ylabel='%s (%s)'%(s.output,units[s.output]),
                                              xlabel='Time-Ot (s)'%(),
                                              sharex=ax[0][ti])

                    traces=[t.slice(event.preferred_origin().time+t.stats.distance/6500*.9,
                                    event.preferred_origin().time+t.stats.distance/1700*1.1)]
                    locationsaddons=names

                    for k in reversed(list(eewenvelopes.keys())):
                        envs=eewenvelopes[k]
                        for streami,streams in enumerate([envs.select(network=t.stats.network,
                                                                               station=t.stats.station,
                                                                               location='E'+outputs[s.output],
                                                                               channel=t.stats.channel),
                                                          envs.select(network=t.stats.network,
                                                                           station=t.stats.station,
                                                                           location=t.stats.location,
                                                                           channel=t.stats.channel), ]):

                            if streami==0 and len(streams)==0 and t.stats.channel[-1] in ['E', 'N']:
                                streams = envs.select(network=t.stats.network,
                                                     station=t.stats.station,
                                                     location='E'+outputs[s.output],
                                                     channel=t.stats.channel[:2])

                            if len(streams)>0:
                                tmp=streams.slice(event.preferred_origin().time+t.stats.distance/6500*.9,
                                                  event.preferred_origin().time+t.stats.distance/1700*1.1)

                                if len(tmp)>0:
                                    traces.append(tmp[0])
                                    locationsaddons.append('%s (%s)'%(k, (streams[0].stats.location+'φ')[0]))

                    for tracei,trace in enumerate(traces) :
                        if tracei>0:
                            if (('acc'  in s.output and 'A' not in trace.stats.location and 'HH' in trace.stats.channel ) or
                                ('acc'  in s.output and 'A' not in trace.stats.location and 'EH' in trace.stats.channel ) or
                                ('acc'  in s.output and 'A' not in trace.stats.location and 'SH' in trace.stats.channel ) or
                                ('vel'  in s.output and 'V' not in trace.stats.location and 'HG' in trace.stats.channel ) or
                                ('vel'  in s.output and 'V' not in trace.stats.location and 'HN' in trace.stats.channel ) or
                                ('disp' in s.output and 'D' not in trace.stats.location) or
                                ('disp' in s.output and 'D' not in trace.stats.location)  ):
                                continue

                        time = numpy.arange(0,
                                            trace.stats.npts / trace.stats.sampling_rate,
                                            1 / trace.stats.sampling_rate)
                        time += trace.stats.starttime-(event.preferred_origin().time+t.stats.distance/6500*.9)
                        # abslute UTC time
                        if False:
                            time = [(trace.stats.starttime+e).datetime for e in time]
                        # time after eq
                        time = [t.stats.distance/6500*.9+e for e in time]

                        if len(trace.data):
                            l=locationsaddons[tracei]#+
                               #trace.id.split('.')[2]#+
                               #outputs[s.output]

                            if l  in forlegend['l']:
                                l=None

                            if trace.stats.location not in ['EA','EV','ED']:
                                o=ax[si][ti].plot(time,
                                                trace.data,#abs(trace.data),
                                                  label=l,
                                                  linewidth=len(traces)-tracei,
                                                zorder=9)
                            else:
                                time = list(time)+list(time)
                                o=ax[si][ti].step(time,
                                                numpy.ma.masked_where((time == min(time)),
                                                                   list(-1*trace.data)+list(trace.data)),#abs(trace.data),
                                                  label=l,
                                                  linewidth=len(traces)-tracei,
                                                zorder=9)
                            if l is not None and l not in forlegend['l']:
                                forlegend['o'].append(o[0])
                                forlegend['l'].append(l)


                    eew.obspy_addons.sticker("%s. %s-%s (%s%s, %.2g%s)"%(alphabet[ti*spd[1]+(si+1)-1],
                                                                      t.id, s.output.upper(),
                                                                  int(t.stats.distance/1000.),
                                                                  'km',
                                                                  max(ax[si][ti].get_ylim()),
                                                                  units[s.output]),
                                             ax[si][ti],
                                             y=1.,
                                                     size= 'xx-small',
                                            foregrounds=['None', 'None'])

                    x=.02
                    ha='left'
                    if si==2:
                        x=.98
                        ha='right'
                    eew.obspy_addons.sticker('%s (%s)'%(s.output,units[s.output]),
                                             ax[si][ti],
                                             fontweight='normal',
                                             y=1.,
                                             x=x,
                                             va='top',
                                             ha=ha,
                                                     size= 'xx-small')

                    # todo : add one legend for ground motion codes and replace subplot lgend entries with max values
                    ax[si][ti].grid()
                    if ti==0:
                        ax[si][ti].set_xlabel('')
                    elif ti+1<spd[0]:
                        ax[si][ti].set_xlabel('')
                    if si==2:
                        ax[si][ti].yaxis.tick_right()
                        ax[si][ti].set_xlabel('')
                        ax[si][ti].spines['left'].set_color('none')
                        ax[si][ti].spines['left'].set_smart_bounds(True)
                    elif si==0:
                        ax[si][ti].spines['right'].set_color('none')
                        ax[si][ti].spines['right'].set_smart_bounds(True)
                        ax[si][ti].set_xlabel('')
                    else:
                        ax[si][ti].spines['right'].set_color('none')
                        ax[si][ti].spines['right'].set_smart_bounds(True)

                    #ax[si][ti].yaxis.set_ticklabels([])#, pad=-22)
                    ax[si][ti].tick_params(labelsize='xx-small')
                    ax[si][ti].yaxis.label.set_size('xx-small')
                    ax[si][ti].tick_params(axis="y",direction="in")#, pad=-22)
                    ax[si][ti].tick_params(axis="x",direction="in")#, pad=-15)
                    ax[si][ti].spines['top'].set_color('none')
                    ax[si][ti].spines['bottom'].set_smart_bounds(True)


                    if ti+1>=spd[0]:
                        break
    
            region=fe.get_region(event.preferred_origin().longitude,event.preferred_origin().latitude)
            legend=fig.legend(forlegend['o'],
                              forlegend['l'],
                              ncol=len(eewenvelopes.keys())+2,
                              prop={'size': 'xx-small'},
                              loc=9,
                              title = '%s - %s%1.2g - %s'%(event.preferred_origin().time.strftime('%Y-%m-%d'),
                                                         event.preferred_magnitude().magnitude_type,
                                                         event.preferred_magnitude().mag,
                                                         region))



            legend.get_title().set_fontsize('x-small')




class eventstreams(object):
    def __init__(self,
                 fdsnwsoptions=None,
                 obspyreadoptions=None,
                 vsenvelopereadoptions=None):
        
        if fdsnws:
            fdsnws(**fdsnwsoptions)
        
        if obspyread:
            for key in pathname_or_url.keys():
                if key not in ['catalog','inventory','raw','acc','vel','disp']:
                    self[key] = obspy.read(**obspyreadoptions[key])
                else:
                    print('WARNING: atribute %s already used'%(key))
        
        if vsenveloperead:
            for key in vsenveloperead.keys():
                if key not in ['catalog','inventory','raw','acc','vel','disp']:
                    self[key] = vs_envelope_read(**vsenvelopereadoptions[key])
                else:
                    print('WARNING: atribute %s already used'%(key))

    def plot(self):
        
        plotstreamsenvelopes(eewenvelopes={},
                         acc=obspy.core.Stream(),
                         vel=obspy.core.Stream(),
                         disp=obspy.core.Stream(),
                         spd=[3,3],
                         outputs={'acc':'A','vel':'V','disp':'D'},
                         units={'acc':'m/s/s','vel':'m/s','disp':'m'},
                         raw=obspy.core.Stream(),
                         sensible=obspy.core.Stream(),
                         channel='*Z',
                         catalog=None,
                         inventory=None)
