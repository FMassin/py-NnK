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
sys.path.append("../NnK")
import eew

def version():
    print('sceewenv_tests','1.1.0')


def combinechannels(stream = obspy.core.Stream(),
                    combine = 'all',
                    horizontal_code = 'b',
                    tridimentional_code = 't',
                    dontdo=[],
                    tridim = obspy.core.stream.Stream(),
                    horiz = obspy.core.stream.Stream(),
                    max_code = 'm'):
    
    for trace in stream:
        if trace.id[:-1] not in dontdo:
            dontdo.append(trace.id[:-1])
            Z = stream.select(id=trace.id[:-1]+'Z')
            E = stream.select(id=trace.id[:-1]+'E')
            N = stream.select(id=trace.id[:-1]+'N')

            if combine in 'horizontal2dimensionaltwodimensionalboth' and E and N:
                starttimes = [tr.stats.starttime for tr in E+N]
                endtimes = [tr.stats.endtime for tr in  E+N]
                EN = (N+E).slice(max(starttimes),min(endtimes))
                npts = [tr.stats.npts for tr in EN]
                horiz += EN
                tr = EN[0].copy()
                tr.stats.channel = tr.stats.channel[:-1]+horizontal_code
                tr.data = (EN[0].data[:min(npts)]**2+EN[1].data[:min(npts)]**2)**.5
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

    if tridim and horiz:
        return tridim , horiz
    elif tridim:
        return tridim
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

def fdsn(base_url="http://arclink.ethz.ch:8080",
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
                      **get_events_options):
    
    
    # First import :
    from obspy.clients.fdsn import Client
    fdsnclient = Client(base_url)
    
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
    eventstreams = {'catalog': fdsnclient.get_events(**get_events_options),
                    'inventory': obspy.core.inventory.Inventory([],None),
                    'raw' : obspy.core.Stream()}
    
    for output in ['catalog','inventory','raw']:
        eventstreams[output].output=output
    
    for event in eventstreams['catalog'].events :
        
        # Load stations
        t=event.preferred_origin().time
        inventory = stationsclient.get_stations(level='station',
                                            startbefore=t,
                                            endafter=t+endafter,
                                            latitude=event.preferred_origin().latitude,
                                            longitude=event.preferred_origin().longitude,
                                            maxradius=maxradius,
                                            location=location,
                                            channel=channel)
        # Load waveforms
        addons = [location, channel] + [t,t+endafter]
        bulk = [tuple(station.split()[0].split('.')[:2]+addons) for station in inventory.get_contents()['stations']]
        waveforms = waveformsclient.get_waveforms_bulk(bulk,
                                                  attach_response=True,
                                                  quality=quality,
                                                  minimumlength=minimumlength,
                                                  longestonly=longestonly)

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
                  maxcount=1000000,
                  sampling_rate=1.,
                 scenvelope=False):
    """
    Evaluate the envelope log file that is produced by scvsmag.
    """
    stream = defaultdict(dict)
    cnt = 0
    first = 9999999999999999 #None
    last = 0 #None
    for file in glob.glob(scvsmag_processing_info):
        f = open(file)
        while True:
            line = f.readline()
            if not line: break
            if cnt > maxcount: break
            
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
    
    realstream = obspy.core.Stream()
    for key in stream.keys():
        timekeys = list(stream[key].keys())
        sortstimekeys = numpy.argsort( timekeys )
        data = [ stream[key][timekeys[i]] for i in sortstimekeys ]
        time = [ timekeys[i] for i in sortstimekeys ]
        lasti=0
        for i in  [i+1 for i,x in enumerate(numpy.diff(time)) if x.total_seconds() != sampling_rate or i+2==len(time)] :
            header = obspy.core.Stats()
            header.network = key.split('.')[0]
            header.station = key.split('.')[1]
            header.location = key.split('.')[2]
            header.channel = key.split('.')[3]
            header.sampling_rate = sampling_rate
            header.npts = len(data[lasti:i])
            header.starttime = obspy.UTCDateTime(time[lasti])
            realstream.append(obspy.core.Trace(data = numpy.asarray( data[lasti:i] ),
                                               header = header))
            lasti=i+1
    realstream._cleanup()
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
                         style='fast',
                         names=['current']
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
                 fdsnws=None,
                 obspyread=None,
                 vsenveloperead=None):
        
        if fdsnws:
            fdsnws(**fdsnws)
        
        if obspyread:
            for key in pathname_or_url.keys():
                if key not in ['catalog','inventory','raw','acc','vel','disp']:
                    self[key] = obspy.read(**obspyread[key])
                else:
                    print('WARNING: atribute %s already used'%(key))
        
        if vsenveloperead:
            for key in vsenveloperead.keys():
                if key not in ['catalog','inventory','raw','acc','vel','disp']:
                    self[key] = vs_envelope_read(**vsenveloperead[key])
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
