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
import datetime
from obspy.taup import TauPyModel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


try:
    from . import obspy_addons
except:
    try:
        import eew.obspy_addons as obspy_addons
    except:
        import NnK.eew.obspy_addons as obspy_addons



def delay(self,
          reference_time,
          mode='relative',
          filters=None,
          nan = numpy.nan): # instead make an error?
    """
        Returns the object's delays or its time or the reference time.
        
    ______
    :type:
        - self:
            ObsPy:class:`~obspy.core.event.origin.Pick` or
            ObsPy:class:`~obspy.core.event.origin.Origin` or
            ObsPy:class:`~obspy.core.event.magnitude.Magnitude`
            ObsPy:class:`~obspy.core.event.Event` or
            ObsPy:class:`~obspy.core.event.catalog.Catalog`.
        - reference_time:
            ObsPy:class:`~obspy.core.utcdatetime.UTCDateTime`.
        - filters:
            string, default None.
        - mode:
            string, default 'relative'.
        - nan:
            Default NumPy:class:`~numpy.nan`.
    :param:
        - self:
            Pick, origin, magnitude, event or catalog.
        - reference_time:
            Time to take as a reference for delay estimate.
        - filters:
            A string to be evalutated as a filter. If
            the evaluated string is True the delay is returned. The
            string might test any attribute of the oject (i.e. self, e.g.
            '"P" in self.phase_hint' to select only P picks).
        - mode:
            String controling output type:
            - 'relative': time difference between the object timestamp and
                          the reference time.
            - 'absolute': timestamp of self.
            The following work only with picks:
            - 'travel': time difference between the object time and the
                        reference time.
            - 'time': timestamp of object.
            - 'self': object itself.
        - nan:
            Value to return when filters evaluate as False (e.g. None, [] or
            default's numpy.nan).
    _______
    :rtype:
        - float of datetime:class:`~datetime.datetime`
    :return:
        - seconds or timestamp.
    _________
    .. note::

        Works similarly to ObsPy:meth:`~obspy.core.stream.select`.

    """
    if filters and 'ref' not in mode :
        if not eval(filters):
            return nan
    if 'ref' in mode:
        return reference_time.datetime
    elif 'abs' in mode:
        return self.creation_info.creation_time.datetime
    elif 'trav' in mode:
        if self.time < reference_time:
            print(str(reference_time)+' WARNING self.time < reference_time return numpy.nan' )
            return numpy.nan
        return self.time - reference_time
    elif 'time' in mode:
        return self.time.datetime
    elif 'self' in mode:
        return self

    if hasattr(self,'time'):
        if self.creation_info.creation_time < self.time:
            print(str(reference_time)+' WARNING self.creation_info.creation_time < self.time return numpy.nan' )
            return  numpy.nan #self.time - reference_time

    if self.creation_info.creation_time < reference_time:
        print(str(reference_time)+' WARNING self.creation_info.creation_time < reference_time return numpy.nan' )
        return  numpy.nan #self.time - reference_time

    return self.creation_info.creation_time - reference_time

def origin_delays(self,
                  reference_time,
                  rank=None,
                  filters=None,
                 mode='relative'):
    tmp=[a.pick_id.get_referred_object().delay(reference_time,filters=filters,mode=mode)  for a in self.arrivals ]
    
    if len(tmp)-1>=rank and rank is not None:
        return numpy.sort(tmp)[min([len(tmp)-1,rank])]
    elif len(tmp)==0:
        return numpy.nan
    return tmp

def magnitude_delays(self,
                     reference_time,
                     rank=None,
                     filters=None,
                     mode='relative'):
    return origin_delays(self.origin_id.get_referred_object(),
                         reference_time,
                         rank=rank,
                         filters=filters,
                         mode=mode)

def event_delay(self,
                mode='relative',
                filters=None):
    return  delay(self,
                  self.preferred_origin_id.get_referred_object().time,
                  mode=mode,
                  filters=filters)

def event_delays(self,
                 field='magnitudes',
                 function='delay',
                 rank=None,
                 ref=None,
                 filters=None,
                 mode='relative'):
    if not ref:
        ref=self.preferred_origin_id.get_referred_object().time
    
    if function == 'delay':
        tmp=[ o.delay(ref,filters=filters,mode=mode) for o in self[field]]
        if rank is not None:
            return numpy.sort(tmp)[min([len(tmp)-1,rank])]
        return tmp
    elif function == 'delays':
        if 'mag' in field:
            o = self.preferred_magnitude_id.get_referred_object()
        else:
            o = self.preferred_origin_id.get_referred_object()
        
        tmp=[ o.delays(ref,rank=rank,filters=filters,mode=mode) ]#for o in self[field]]
        if rank is not None:
            try:
                return numpy.nanmin(tmp)
            except:
                try:
                    return sorted([t for t in tmp if isinstance(t, datetime.datetime) ])[0]
                except:
                    return None
        return tmp
    elif function == 'traveltimes':
        tmp=[ o.traveltimes(ref,rank=rank,filters=filters,mode=mode) for o in self[field]]
        if rank is not None:
            try:
                return numpy.nanmin(tmp)
            except:
                try:
                    return sorted([t for t in tmp if isinstance(t, datetime.datetime) ])[0]
                except:
                    return None
        return tmp

def catalog_delays(self,
                 field='magnitudes',
                 function='delay',
                   rank=None,
                  filters=None,
                 mode='relative'):

    return [ e.delays(field,function,rank=rank,filters=filters,mode=mode) for e in self.events ]

obspy.core.event.origin.Pick.delay = delay
obspy.core.event.origin.Origin.delay = delay
obspy.core.event.magnitude.Magnitude.delay = delay
obspy.core.event.origin.Origin.delays = origin_delays
obspy.core.event.magnitude.Magnitude.delays = magnitude_delays
obspy.core.event.Event.delay = event_delay
obspy.core.event.Event.delays = event_delays
obspy.core.event.catalog.Catalog.delays = catalog_delays
## obspy integration
#class Origin(...):
#
#    def delay(self, ...):
#        return utils._delay(self, ...)


def mag_norm(mags):
    n = matplotlib.colors.PowerNorm(1,
                                    vmin=numpy.max([numpy.min(mags),0.01]),
                                    vmax=numpy.max(mags))
    return n
def depth_norm(depths):
    n = matplotlib.colors.LogNorm(vmin=numpy.max([numpy.min(depths),0.1]),
                                  vmax=numpy.max(depths))
    return n

def nsta2msize(n,nref=None):
    n = numpy.asarray(n)
    if nref:
        nref = numpy.asarray(nref)
    else:
        nref = numpy.asarray(n)
    size = (n-min(nref)+1)/(max(nref)-min(nref))*500
    
    return size


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



def get(self, lst, att=None, types=[] , full=False, pref=False, last=False, first=False, nan=True, ct=False, subatt=None):
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
                                if subatt and hasattr(elt[att],subatt) :
                                    val = elt[att][subatt]
            if ct:
                out.append(mem)
            else:
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
                                if subatt and hasattr(elt[att],subatt) :
                                    val = elt[att][subatt]
            if ct:
                out.append(mem)
            else:
                out.append(val)
        else:
            if nan or set(['n','nan']) & set(types) : #:
                out.append(numpy.nan)
                
    return out


def distfilter(self=obspy.core.event.catalog.Catalog(),
               dmax=None,
               dmin=0.,
               x1=None,
               y1=None,
               x2=None,
               y2=None,
               z1=None,
               out=False):
    """
        Filter the events in self (obspy:Catalog) with specified distance range form specified point or line.
        
    """
    
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
        elif o.quality.minimum_distance :
            d = o.quality.minimum_distance*110.
        else:
            print('warning: no minimum_distance in origin, excluded')
            print(o)
            d=999999999999.
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




def match_events(self=obspy.core.event.catalog.Catalog(),
           toadd=obspy.core.event.catalog.Catalog(),
           client=None,
           v=False,
           x=True,
                 **get_events_args):
    """
        Matchs the events in self (obspy:Catalog) with specified catalog.
        
        Matchs with catalog available in webservice if client is specified.
    """
    
    tofind = toadd.copy()
    matchs = obspy.core.event.catalog.Catalog()
    extras = obspy.core.event.catalog.Catalog()
    matchs.description = 'Intersections of '+str(self.description)+' and '+str(tofind.description)
    missed = obspy.core.event.catalog.Catalog()
    missed.description = 'Part of '+str(self.description)+' not in '+str(tofind.description)
    listfound=list()
    
    if v:
        if client:
            print('Reference client')
            print(client)
    
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
                eq_specs = {'starttime':str(o.time-40),
                            'endtime':str(o.time+40),
                            'minlatitude':o.latitude-1,
                            'maxlatitude':o.latitude+1,
                            'minlongitude':o.longitude-1,
                            'maxlongitude':o.longitude+1,
                            'includearrivals':True}

                try:
                    tofind = client.get_events( **eq_specs, **get_events_args )
                    if v:
                        print('Event in reference client')
                except:
                    if v:
                        print('No event in reference client')
                    continue
                try:
                    tofind = obspy.read_events(get_events_args['filename'],
                                               format=get_events_args['format'])
                except:
                    pass
            
            for candidate in tofind.filter( *filter ).events:
                candidateo = candidate.preferred_origin() or candidate.origins[-1]
                if candidateo.time not in listfound:
                
                    dt = abs(o.time-candidateo.time)
                    dl = numpy.sqrt((o.latitude-candidateo.latitude)**2+(o.longitude-candidateo.longitude)**2)
                
                    if (dt < 10 and dl <=.1 and
                        (dl<memdl or dt<memdt)):
                    
                        found = True
                        memi = str(candidateo.time)
                        memdl = dl
                        memdt = dt
                        meme = candidate
                    
                        if v:
                            print('fits nicely input catalog ',tofind.description,':\n  ', candidate.short_str())
                        break
            
                    elif (dt < 20 and dl <=.4 and
                          (dl<memdl or dt<memdt)):
                    
                        found = True
                        memi = str(candidateo.time)
                        memdl = dl
                        memdt = dt
                        meme = candidate
                        if v:
                            print('fits input catalog ',tofind.description,':\n  ', candidate.short_str())

                    elif (dt < 40 and dl <.8 and
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
        matchs_otherversion, extras, trash = match_events(self=tofind,
                                                          toadd=self,
                                                          v=v,
                                                          x=False)
    
    return matchs, missed, extras


def map_events(self=obspy.core.event.catalog.Catalog(),
               bmap=None,
               fig=None,
               titletext='',
               eqcolorfield = 'depth',
               colorbar=None,
               fontsize=8,
              cmap = None,
               vmax=40.,
              vmin=0.,
               prospective_inventory=None,
               latencies=None,
               flat_delay=0.,
               VpVsRatio=1.75,
               model_correction='iasp91',
               Vs=3.428,
               extra=None,
               extramarker='1',
               extraname='',
               fp=False,
               magnitude_types=['MVS','Mfd']):
    """
        Map event locations on given basemap and colorcoded with depth or times or delays.
        
    """


    cf=[]
    mags = get(self,'magnitudes','mag',['b'] )
    times = get(self, 'origins','time', types=['b'])
    stndrdth = ['','st','nd','rd','th']
    for i in range(10):
        stndrdth.append('th')
    
    if prospective_inventory:
        
        if eqcolorfield[0] == 'P':
            model = TauPyModel(model=model_correction)
        stations_longitudes, stations_latitudes = obspy_addons.search(prospective_inventory,
                                                                  fields=['longitude','latitude'],
                                                                  levels=['networks','stations'])
    
    if len(mags) >0:
        longitudes = get(self, 'origins','longitude', types=['b'])
        latitudes = get(self, 'origins','latitude', types=['b'])


        if eqcolorfield == 'depth':
            eqcolor = get(self, 'origins',eqcolorfield, types=['b'])
            eqcolor = numpy.asarray(eqcolor)/-1000.
            eqcolorlabel = 'Event depth (km) and magnitude'
        
        elif eqcolorfield[0] in ['t', 'P'] and (eqcolorfield[-1] in [str(int(n)) for n in range(9)] or eqcolorfield[1] in ['M', 'O']):
            eqcolorlabel = 'Travel time to %s$^{th}$ station (s'%(eqcolorfield[1:])
            if eqcolorfield[1] in ['M', 'O']:
                if eqcolorfield[-1] is 'l':
                    eqcolorlabel = 'last %s (s after Ot)'%(eqcolorfield[-2].upper())
                else :
                    eqcolorlabel = '%s$^{%s}$ %s (s after Ot)'%(eqcolorfield[-1],stndrdth[int(eqcolorfield[-1])],eqcolorfield[1:-1])
            
            if  latencies is None and ( 'P' in eqcolorfield ):
                eqcolorlabel = 'Travel time for '+eqcolorlabel
            else:
                eqcolorlabel = 'Delay at '+eqcolorlabel
            
            eqcolorlabel = eqcolorlabel.replace(' O (',' origin (')
            eqcolorlabel = eqcolorlabel.replace(' M (',' magnitude (')
            eqcolorlabel = eqcolorlabel.replace(' P (',' P trigger (')
            eqcolorlabel = eqcolorlabel.replace(' OP (',' P trigger (')
            eqcolorlabel = eqcolorlabel.replace(' Op (',' P trigger (')
            
            ntht=list()
            if prospective_inventory:
                print('Using travel time modeling')
            else:
                print('Using earthquake location travel time')

            for ie,e in enumerate(self.events):
                t=[]
                origin=e.preferred_origin_id.get_referred_object()
                #magnitude=e.preferred_magnitude_id.get_referred_object()
                
                if eqcolorfield[1] in ['O']:
                    o=e.origins[0]
                    for o in e.origins:
                        if eqcolorfield[2] not in ['p','P','s','S']:
                            if eqcolorfield[2:-1] in ['',str(o.method_id),str(o.creation_info.author)] : #or eqcolorfield[1] in ['M']:
                                if o.creation_info.creation_time-origin.time>0:
                                    t.append(o.creation_info.creation_time-origin.time)
                    
                        else:
                            if eqcolorfield[-1] == 'l':
                                t.append(numpy.nan)
                            for a in o.arrivals:
                                #if a.phase[0] in [ eqcolorfield[2].upper() , eqcolorfield[2].lower() ]:
                                p = a.pick_id.get_referred_object()
                                if p.time - origin.time>0 and p.creation_info.creation_time - origin.time>0:
                                    if eqcolorfield[2] in ['s','p']:
                                        if eqcolorfield[-1] == 'l' :#and t[-1]<9999999999999999.:
                                            t[-1] = numpy.nanmax([p.creation_info.creation_time - origin.time, t[-1]])
                                        else:
                                            t.append(p.creation_info.creation_time - origin.time)
                                    else:
                                        if eqcolorfield[-1] == 'l' :#and t[-1]<9999999999999999.:
                                            t[-1] = numpy.nanmax([p.time - origin.time, t[-1]])
                                        else:
                                            t.append(p.time - origin.time)
                                        
                                        mlatency=0
                                        if latencies:
                                            mlatency = numpy.median(latencies[p.waveform_id.network_code+'.'+p.waveform_id.station_code])
                                        t[-1] += mlatency+flat_delay

                                #if eqcolorfield[-1] =='l':
                                #    pass#    break
                                #elif len([tmp for tmp in t if tmp<vmax]) > int(eqcolorfield[-1])-1:
                                #    break
                        #if eqcolorfield[-1] == 'l':
                        #    print(t)
                elif eqcolorfield[1] in ['M']:
                    for m in e.magnitudes:
                        #test = re.sub('.*rigin#','', str(m.resource_id ))
                        #test = re.sub('#.*','', str(test))
                        if eqcolorfield[1:-1] in str(m.magnitude_type) or str(m.magnitude_type) in eqcolorfield[1:-1] : # and str( test )  in  str( o.resource_id ) :
                            if m.creation_info.creation_time-origin.time>0:
                                t.append(m.creation_info.creation_time-origin.time)
                                #break
                                    
                elif eqcolorfield[0] in ['p','P'] :
                    d=[9999999999999999 for d in range(100)]
                    if not prospective_inventory:
                        for a in origin.arrivals:
                            if a.phase[0] in [ eqcolorfield[0].upper() , eqcolorfield[0].lower() ]:
                                p = a.pick_id.get_referred_object()
                                if eqcolorfield[0] in ['s','p']:
                                    t.append(p.creation_info.creation_time - origin.time)
                                else:
                                    t.append(p.time - origin.time)
                                    if latencies:
                                        mlatency = numpy.median(latencies[p.waveform_id.network_code+'.'+p.waveform_id.station_code])
                                        t[-1] += mlatency+flat_delay

                            if eqcolorfield[-1] =='l':
                                pass#    break
                            elif len([tmp for tmp in t if tmp<vmax])>int(eqcolorfield[-1])-1:
                                break
            

                    else:
                        for istation,lonstation in enumerate(stations_longitudes):
                            latstation =  stations_latitudes[istation]
                            ep_d =  numpy.sqrt((lonstation-origin.longitude)**2+(latstation-origin.latitude)**2)
                            d.append(ep_d)
                        d = numpy.sort(d)
                        for ep_d in  d[:int(eqcolorfield[1:])]:
                            arrivals = model.get_travel_times(origin.depth/1000.,
                                                              distance_in_degree=ep_d,
                                                              phase_list=['ttp'],
                                                              receiver_depth_in_km=0.0)
                            try:
                                t.append( numpy.nanmin([ a.time for a in arrivals ]))
                            except:
                                print('No phase for depth',origin.depth/1000.,'and distance',ep_d)
                                pass
                for tmp in range(100):
                    t.append(99999999999999.)
                t = numpy.sort(t)
                if eqcolorfield[-1] =='l':
                    ntht.append( numpy.nanmin(t) )
                else:
                    ntht.append( t[int(eqcolorfield[-1])-1] )
                
            
            ntht = numpy.asarray(ntht)
            ntht[ntht<0.]=0.
            eqcolor = ntht

        elif eqcolorfield[0] == 'I' and eqcolorfield[-1] in [str(int(n)) for n in range(9)]:
            
            if eqcolorfield[1] in ['o','O','m','M','p', 'P']:
                toadd=eqcolorfield[1:-1]
                if toadd in ['P']:
                    toadd='trigger'
                eqcolorlabel = 'MMI at %s$^{%s}$ %s '%(eqcolorfield[-1],stndrdth[int(eqcolorfield[-1])],toadd)
            else:
                eqcolorlabel = 'MMI at %s$^{%s}$ travel time'%(eqcolorfield[-1],stndrdth[int(eqcolorfield[-1])])
            if latencies:
                eqcolorlabel += ' (delays incl.)'
            if eqcolorfield[1]=='0':
                eqcolorlabel = 'Epicentral MMI'
            
            eqcolorlabel = eqcolorlabel.replace(' O ',' origin ')
            eqcolorlabel = eqcolorlabel.replace(' M ',' magnitude ')
            eqcolorlabel = eqcolorlabel.replace(' P ',' P trigger ')
            eqcolorlabel = eqcolorlabel.replace(' p ',' P trigger ')
            
            
            r=list()
            error=list()
            for ie,e in enumerate(self.events):
                d=[9999999999999999. for d in range(100)]
                preferred_origin = e.preferred_origin_id.get_referred_object()
                if eqcolorfield[1] in ['M', 'm']:
                    for m in e.magnitudes:
                        if eqcolorfield[1:-1] in str(m.magnitude_type) or str(m.magnitude_type) in eqcolorfield[1:-1] :
                            if m.creation_info.creation_time-preferred_origin.time <0 :
                                print('WARNING magnitude at', preferred_origin.time, 'created on', m.creation_info.creation_time )
                            else:
                                d.append((m.creation_info.creation_time-preferred_origin.time)*Vs)

                elif eqcolorfield[1] in ['O', 'o']:
                    for o in e.origins:
                        if o.creation_info.creation_time-preferred_origin.time <0 :
                            print('WARNING origin at', preferred_origin.time, 'created on',  o.creation_info.creation_time)
                        else:
                            d.append((o.creation_info.creation_time-preferred_origin.time)*Vs)

                else:
                    if not prospective_inventory:
                        for a in preferred_origin.arrivals:
                            if 'p' in str(a.phase).lower():
                                p = a.pick_id.get_referred_object()
                                for testpick in e.picks:
                                   if (testpick.waveform_id == p.waveform_id and
                                       testpick.phase_hint == p.phase_hint and
                                       testpick.creation_info.creation_time<p.creation_info.creation_time ):
                                       p=testpick
                            
                                if eqcolorfield[1] in ['p', 'P']:
                                    if max([p.creation_info.creation_time, p.time]) < preferred_origin.time:
                                        print('WARNING', 'P at',p.time, ' created on', p.creation_info.creation_time, 'while origin at', preferred_origin.time)
                                    else:
                                        d.append((max([p.creation_info.creation_time, p.time])-preferred_origin.time)*Vs)
                                else:
                                    testd = numpy.sqrt((a.distance*110.)**2+(preferred_origin.depth/1000.)**2)
                                    if testd <= preferred_origin.depth/1000. :
                                        print('WARNING','P at ',testd,'km while origin at ',preferred_origin.depth/1000. ,'km deep' )
                                    d.append( testd/VpVsRatio )

                                if latencies:
                                    mlatency = numpy.median(latencies[p.waveform_id.network_code+'.'+p.waveform_id.station_code])
                                    d[-1] += (mlatency+flat_delay)*Vs
                    else:
                        for istation,lonstation in enumerate(stations_longitudes):
                            latstation =  stations_latitudes[istation]
                            ep_d = obspy_addons.haversine(lonstation,
                                                        latstation,
                                                        o.longitude,
                                                        o.latitude)
                            d.append( numpy.sqrt((ep_d/1000.)**2+(o.depth/1000.)**2)/VpVsRatio)

                d = numpy.sort(d)
                if eqcolorfield[1]=='0':
                    r.append( preferred_origin.depth/1000. )
                else:
                    r.append( d[int(eqcolorfield[-1])-1] )
            r = numpy.asarray(r)
            r[r<1.]=1.
            eqcolor = obspy_addons.ipe_allen2012_hyp(r,
                                                     numpy.asarray(mags))
            #eqcolor[numpy.isnan(eqcolor)] = numpy.nanmin(eqcolor[eqcolor>.5])
            #eqcolor[eqcolor<1.] =  numpy.nan #numpy.nanmin(eqcolor[eqcolor>.5])
            cmap='nipy_spectral'
            vmin=1.
            vmax=8.5

        sizes, sizesscale, labelsscale, x = eventsize( mags = get(self,'magnitudes','mag',['b'] ))
        
        bmap.scatter(longitudes,
                         latitudes,
                         sizes,
                         edgecolor='w',
                         lw=1,
                        zorder=992)
        bmap.scatter(longitudes,
                         latitudes,
                         sizes,
                         edgecolor='k',
                         lw=.5,
                        zorder=992)
        bmap.scatter(longitudes,
                     latitudes,
                     sizes,
                     color='w',
                     edgecolor='none',
                     lw=0.,
                     zorder=992)
        
        sortorder =  numpy.argsort(eqcolor)
        if eqcolorfield[0] in [ 't' , 'P' , 'p' ]:
            sortorder =  sortorder[::-1]

        eqcolor[eqcolor>vmax]=numpy.nan
        cf = bmap.scatter([longitudes[i] for i in sortorder],
                          [latitudes[i] for i in sortorder],
                          [sizes[i] for i in sortorder],
                          [eqcolor[i] for i in sortorder ],
                          edgecolor='None',
                          cmap=cmap,
                          norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax),
                          zorder=995)
        
        
        nextra=''
        if extra:
            nextra=0
            for i in numpy.argsort(eqcolor):
                test = [((e[10]-longitudes[i])**2+(e[9]-latitudes[i])**2+((obspy.UTCDateTime(e[0],e[1],e[2],e[3],e[4],e[5])-times[i])/1200)**2)**.5 for e in extra]
                
                
                if numpy.nanmin(test)<.1 :
                    
                    t = extra[numpy.argmin(test)]
                    nextra+=1
                    bmap.scatter(x=longitudes[i],
                                  y=latitudes[i],
                                  s=sizes[i],
                                  c='w', 
                                  marker=extramarker,
                                  latlon=True,
                                  zorder=997)
                    bmap.ax.text(bmap(longitudes[i], latitudes[i])[0]+t[12]/2,
                                  bmap(longitudes[i], latitudes[i])[1]-.1,
                                  s='%s'%(t[0]),#,t[1],t[2]), 
                                  va='top',
                                  ha=t[11],
                                  alpha=.5,
                                  fontsize='x-small',
                                  path_effects=[
                                      matplotlib.patheffects.withStroke(linewidth=3,
                                                       foreground="white")],
                                 zorder=1000)
            extraname='and %s %s ' % (str(nextra), extraname)

        if fp:
            #from obspy.clients.fdsn import Client
            from obspy.imaging.beachball import beach
            cmap=cf.get_cmap()
            if vmin and vmax:
                norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            else:
                norm=matplotlib.colors.Normalize(vmin=min(eqcolor),vmax=max(eqcolor))
            
            #client = Client("USGS")
            for i in numpy.argsort(sizes)[max([-100,-1*len(sizes)]):]:
                if sizes[i]>=fp:
                    id = str(self.events[i].resource_id).split('id=')[-1].split('&')[0]
                    #try:
                    e = self.events[i] #client.get_events(eventid=id,producttype='moment-tensor').events[0]
                    o = e.preferred_origin()
                    for f in e.focal_mechanisms[::-1]:
                        #print(f)
                        t=f.moment_tensor.tensor
                        xy=bmap(longitudes[i],latitudes[i])
                        try:
                            b = beach([t.m_rr,t.m_tt,t.m_pp,t.m_rt,t.m_rp,t.m_tp],
                                      xy=(xy[0],xy[1]),
                                      width=.028*sizes[i]**.525,
                                      linewidth=0,
                                      alpha=1,
                                      facecolor=cmap(norm(eqcolor[i])),
                                      #edgecolor=cmap(norm(eqcolor[i]))
                                      )
                            b.set_zorder(999)
                            bmap.ax.add_collection(b)
                            break
                        except:
                            pass
                        #except:
                        #pass

        titletext += '\n%s events %s(%s' % (len(times), extraname, str(min(times))[:10])
        if str(max(times))[:10] > str(min(times))[:10]:
            titletext += ' to %s)' % str(max(times))[:10]
        else:
            titletext += '%s)' % titletext
        
        if colorbar and len(self.events)>0 :
            fig.cb = obspy_addons.nicecolorbar(cf,
                                  axcb = bmap.ax,
                                  label = eqcolorlabel,
                                  data = eqcolor,
                                  fontsize=fontsize)
            fig.cb.ax.scatter(numpy.repeat(.1,len(sizesscale)),x,s=sizesscale, facecolor='None', linewidth=3, edgecolor='w')
            fig.cb.ax.scatter(numpy.repeat(.1,len(sizesscale)),x,s=sizesscale, facecolor='None', edgecolor='k')
            for i,v in enumerate(labelsscale):
                fig.cb.ax.text(-.2,x[i],'M'+str(v), va='center',ha='right', rotation='vertical',fontsize=fontsize)
            if eqcolorfield == 'depth':
                fig.cb.ax.set_yticklabels([ l.get_text().split('−')[-1] for l in fig.cb.ax.get_yticklabels()])

            if eqcolorfield[0] == 'I' and eqcolorfield[-1] in [str(int(n)) for n in range(100)]:
                fig.cb.ax.set_yticklabels(obspy_addons.num2roman(list(range(1,15))))

    return titletext


def plot_traveltime(self=obspy.core.event.catalog.Catalog(),
                    NumbersofP=[6,4],
                    NumbersofS=[1],
                    ax=None,
                    plots=True,
                    style='c',
                    iplot=None,
                    sticker_addons=None,
                    depth_correction=5,
                    model_correction='iasp91',
                    latencies=None,
                    flat_latency=0.,
                    latencies_type=numpy.median):

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
                            sticker_addons=sa,
                            depth_correction=depth_correction,
                            model_correction=model_correction,
                            latencies=latencies,
                            flat_latency=flat_latency,
                            latencies_type=latencies_type)
            #ax[i].grid()
            if i==0:
                ax[i].legend().set_visible(False)
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
                ax[i].legend().set_visible(False)
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

    Ndt=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    ndt=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    
    mags=numpy.zeros((maxarr,len(self.events)))*numpy.nan
    
    
    if depth_correction:
        model = TauPyModel(model=model_correction)
        arrivals = model.get_travel_times(depth_correction,
                                          distance_in_degree=0.0001,
                                          phase_list=['p','s'],
                                          receiver_depth_in_km=0.0)
        tpz_correction = arrivals[0].time
        tsz_correction = arrivals[1].time
    

    if not latencies:
        latencies = {}
        for ie,e in enumerate(self.events):
            for p in e.picks:
                k = '%s.%s'%(p.waveform_id.network_code,p.waveform_id.station_code)
                dt = p.creation_info.creation_time-p.time
                if dt<60*3 and dt>0:
                    if k not in latencies:
                        latencies[k] = [ dt ]
                    else:
                        latencies[k].append( dt )
        for k in latencies:
            latencies[k] = numpy.sort(latencies[k])[:int(len(latencies[k])/100*16)+1]

    dt=[0.]
    for k in latencies:
        dt.extend(latencies[k])
    dt = numpy.sort(dt)[:int(len(dt)/100*16)+1]
    for ie,e in enumerate(self.events):
        for p in e.picks:
            k = '%s.%s'%(p.waveform_id.network_code,p.waveform_id.station_code)
            if k not in latencies:
                latencies[k] = dt

    for ie,e in enumerate(self.events):
        ns=-1
        np=-1
        VpVsRatio = 5./3.
        mag=e.magnitudes[0].mag
        for m in e.magnitudes:
            if str(e.preferred_magnitude_id) == str(m.resource_id):
                mag=m.mag

        o = e.preferred_origin_id.get_referred_object()
        if depth_correction:
            z=o.depth/1000.
            if o.depth/1000. < .1:
                z=.1
            arrivals_correction = model.get_travel_times(z,
                                                         distance_in_degree=0.,#0001,
                                              phase_list=['p','s'],
                                              receiver_depth_in_km=0.0)

        for a in o.arrivals:
            p = a.pick_id.get_referred_object()
            if p.time-o.time>0:
                k = '%s.%s'%(p.waveform_id.network_code,p.waveform_id.station_code)
                if 'S' in str(a.phase[0])  :
                    ns+=1
                    n[ns][ie] = p.time-o.time
                    ndt[ns][ie] = p.time-o.time+flat_latency+latencies_type(latencies[k])
                    mags[np][ie] = mag
                    if depth_correction:
                        tnadir = arrivals_correction[1].time
                        thoriz = ((p.time-o.time)**2-tnadir**2)**.5
                        n[ns][ie] = (thoriz**2+tsz_correction**2)**.5
                        ndt[ns][ie] = (thoriz**2+tsz_correction**2)**.5+flat_latency+latencies_type(latencies[k])

                elif 'P' in str(a.phase[0])  :
                    np+=1
                    N[np][ie] = p.time-o.time
                    Ndt[np][ie] = p.time-o.time+flat_latency+latencies_type(latencies[k])
                    mags[np][ie] = mag
                    if depth_correction:
                        tnadir = arrivals_correction[0].time
                        thoriz = ((p.time-o.time)**2-tnadir**2)**.5
                        N[np][ie] = (thoriz**2+tpz_correction**2)**.5
                        Ndt[np][ie] = (thoriz**2+tpz_correction**2)**.5+flat_latency+latencies_type(latencies[k])
        AvVsVpRatio=0.
        AvN=0
        VpVsRatio=1.8
        for p in range(np):
            if n[p][ie] >0 and N[p][ie] >0:
                AvVsVpRatio += n[p][ie]/N[p][ie]
                AvN+=1
                VpVsRatio = AvVsVpRatio/AvN
        for p in range(np):
            if not n[p][ie] >0:
                n[p][ie] = N[p][ie]*VpVsRatio
                ndt[p][ie] = Ndt[p][ie]*VpVsRatio


    
    if not ax:
        f, (ax) = matplotlib.pyplot.subplots(1, 1)
        obspy_addons.adjust_spines(ax, ['left', 'bottom'])
    if style in ['vs','v','versus']:
        ax.set_ylabel('Observed S travel time')
        ax.set_xlabel('Observed P travel time')
        if depth_correction:
            ax.set_xlabel('Observed P travel Time (corr. to depth %s km)'%(depth_correction))
            ax.set_ylabel('Observed S travel Time (corr. to depth %s km)'%(depth_correction))
        ax.set_aspect('equal')
    elif style in  ['cum','c','cumulate']:
        ax.set_ylabel('Phase arrival order')
        ax.set_xlabel('Travel Time')
        if depth_correction:
            ax.set_xlabel('Travel Time (corr. to depth %s km)'%(depth_correction))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        minorLocator = matplotlib.ticker.MultipleLocator(1)
        ax.yaxis.set_minor_locator(minorLocator)
    ax.grid()
    if sticker_addons:
        obspy_addons.sticker(sticker_addons,
                             ax,x=0, y=1,
                             ha='left', va='top')  # ,fontsize='xx-small')

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


        labels={'P': 'P$_{tt}$',
                'S': 'S$_{tt}$',
                'Ps1': 'tt $\in\sigma_1$',
                'Ps2': 'tt $\in\sigma_2$',
                'Ps3': 'tt $\in\sigma_3$',
                'Pm': '$\widetilde{tt}$',
                'Ss1': None,
                'Ss2': None,
                'Ss3': None,
                'Sm': None}
        
        if latencies :
            labels['P']='P$_{tt}^{tt+\delta}$'
            labels['S']= 'S$_{tt}^{tt+\delta}$'

        l=['P','S']
        p=max(NumbersofP)+2
        b = max([b,numpy.nanmax(n[p])])
        b = max([b,numpy.nanmax(N[p])])

        for i in range(max(NumbersofP)):
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i+1,
                                                    ndt[i],
                                                    PorS='S',
                                                    upordown=1,
                                                    labels=labels)
            
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i+1,
                                                    n[i],
                                                    PorS='S',
                                                    upordown=-1,
                                                    labels=labels)
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i+1,
                                                    Ndt[i],
                                                    PorS='P',
                                                    upordown=1,
                                                    labels=labels)
            
            obspy_addons.plot_arrivals_chronologies(ax,
                                                    i+1,
                                                    N[i],
                                                    PorS='P',
                                                    upordown=-1,
                                                    labels=labels)

    if plots:
        if not iplot:
            ax.legend(#ncol=2,
                      loc=4,
                      framealpha=1)
        if style in ['vs','v','versus']:
            ax.plot([0,b],[0,b],color='grey')
    return ax, b

def plot_Mfirst(self=obspy.core.event.catalog.Catalog(),last=0, agency_id=['*'],minlikelyhood=None):
    
    #solutions = Solutions(catalog=self,last=last, agency_id=agency_id, minlikelyhood = minlikelyhood)
    #mags = solutions.mags
    mags = get(self,'magnitudes','mag',['f'] )
    pref_mags = get(self,'magnitudes','mag',['b'] )
    #profs = solutions.depths
    profs = get(self, 'origins','depth', types=['b'])
    lats = get(self,'origins','latitude',['b'] )
    lons = get(self,'origins','longitude',['b'] )
    
    mags_profs = get(self, 'origins','depth', types=['f'])
    mags_lats = get(self,'origins','latitude',['f'] )
    mags_lons = get(self,'origins','longitude',['f'] )

    mags_orig_errors=[obspy_addons.haversine(lons[i],lats[i],mags_lons[i],mags_lats[i])/1000. for i in range(len(mags))]
                             
    #mags_stations = solutions.mags_station_counts
    mags_stations = get(self,'magnitudes','station_count',['f'] )
    mags_orig_station_counts = get(self,'origins','quality',['f'] ,subatt='used_station_count')
    #m1_errors = solutions.mags_errors
    m1_errors = [ pref_mags[i] -mags[i] for i in range(len(mags))]
    #m1_types = solutions.mags_types
    m1_types = get(self,'magnitudes','magnitude_type',['f'] )
    #m1_times = solutions.mags_creation_times
    m1_times = get(self,'magnitudes','mag',['f'],ct=True)
    #m1_origint = solutions.origins_times
    m1_origint = get(self,'origin','time',['b'])
    #m1_delays = solutions.mags_delays
    m1_delays = [ m1_times[i] - m1_origint[i] for i in range(len(mags))]
    
    print('origin and mag may not match')
    
    
    for n in range(len(mags_stations)):
        if  mags_stations[n] is None:
            mags_stations[n]=1


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
                            obspy_addons.nsta2msize(m,[0,32]),#mags_stations),
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
                            obspy_addons.nsta2msize([mags_orig_station_counts[j] for j in matches],[0,32]),# mags_stations)
                           [mags_orig_errors[j] for j in matches], #[profs[j] for j in matches],
                            m,
                            facecolor='w',alpha=1.,zorder=100,edgecolors='k')
                sc = ax.scatter([mags[j] for j in matches] ,
                                [m1_errors[j] for j in matches] ,
                                obspy_addons.nsta2msize([mags_orig_station_counts[j] for j in matches],[0,32]),# mags_stations),
                                [mags_orig_errors[j] for j in matches],
                                m,
                                norm=obspy_addons.depth_norm([0,100]),#profs),
                                label=types[i],alpha=.8,zorder=150,edgecolors='None')
        cb=matplotlib.pyplot.colorbar(sc)
        cb.set_label('Location error (km)')
        lg = matplotlib.pyplot.legend(loc=1, numpoints=1, scatterpoints=1, fancybox=True)
        lg.set_title('Marker & Sizes')
        lg.get_frame().set_alpha(0.1)
        lg.get_frame().set_color('k')
        matplotlib.pyplot.axis('equal')

    return f


def delays(self=obspy.core.event.catalog.Catalog(),
           element = [],
           target=['creation_info','creation_time'],
           ref = 'preferred_origin().time',
           rank=0):
    
    if target[-1][-1].isdigit() :
        rank=int(target[-1][-1])-1
        target[-1] = target[-1][:-1]
    data = []
    references=[]

    for e in self.events:
        reftime=0
        if 'preferred_origin().time' in ref:
            for o in e.origins:
                if str(o.resource_id) == str(e.preferred_origin_id):
                    reftime = o.time

        if reftime:
            lasti = len(references)
            if len(element) >0:
                for sub in e[element[0]]:
                    
                    if len(element) >1:
                        
                        lasti = len(references)
                        for subsub in sub[element[1]]:
                            if len(element) >2:
                                pass
                            else:
                                references.append(reftime.datetime)
                                o=subsub
                                for t in target:
                                    o = o[t]
                                if 'time' in target[-1]:
                                    data.append(o-reftime)
                                else:
                                    data.append(o.datetime)

                        data[lasti:] = numpy.sort(data[lasti:])[-1]
                    else:
                        references.append(reftime.datetime)
                        o=sub
                        for t in target:
                            o = o[t]
                        if 'time' in target[-1]:
                            data.append(o-reftime)
                        else:
                            data.append(o.datetime)
            else:
                references.append(reftime.datetime)
                o=e
                for t in target:
                    o = o[t]
                if 'time' in target[-1]:
                    data.append(o-reftime)
                else:
                    data.append(o.datetime)
        if len(data[lasti:])>0:
            tmp = numpy.sort(data[lasti:])[min([rank,len(data[lasti:])-1])]
            data[lasti:] = [ tmp for d in data[lasti:] ]
            data = data[:lasti+1]
            references = references[:lasti+1]
            
    return data,references

def history(self=obspy.core.event.catalog.Catalog(),
            fig=False,
            ax=False,
            fields = ['magnitudes',
                      'magnitudes',
                      'magnitudes',
                      'origins',
                      'origins',
                      'origins'],
            filters = ['self.magnitude_type not in ["MVS" , "Mfd"]',
                       'self.magnitude_type  in ["MVS"]',
                       'self.magnitude_type  in ["Mfd"]',
                       'True',
                       '"P" in self.phase_hint',
                       '"P" in self.phase_hint'],
            functions = ['delay',
                         'delay',
                         'delay',
                         'delay',
                         'delays',
                         'delays'],
            modes = ['rel',
                     'rel',
                     'rel',
                     'rel',
                     'rel',
                     'trav'],
            ranks = [0,0,0,0,5,5],
            legendlabels=['Traditional M$_{L}$ delay',
                          'Actual EEW delay (first M$_{VS}$)',
                          'Actual EEW delay (first M$_{fd}$)',
                          'Actual first origin delay',
                          'Actual data delay (6$^{th}$ trigger)',
                          'Theoritical delay (6$^{th}$ travel time)'],
            markers='.',
            eew=True,
            titleaddons=None):
    """
    Plots each event's time delays as a function of time
        - Magnitude delays
        - origin delays
        - pick delays
        - data delays.
    ______
    :type:
        - self:
            ObsPy:class:`~obspy.core.events.Events`.
    :param:
        - self: event catalog
            ObsPy:class:`~obspy.core.events.Events`.
    _______
    :rtype:
        - matplotlib:class:`~matplotlib.pyplot.figure`
    :return:
        - figure with plot.
    _________
    .. note::

        Works similarly to ObsPy:meth:`~obspy.core.stream.select`.

    """
    import datetime
    from operator import sub

    if not fig and not ax:
        fig = matplotlib.pyplot.figure()
    if not ax:
        ax = fig.gca()

    handles=[]
    for index,field in enumerate(fields):

        test = [ self.delays(field=field,
                        function=functions[index],
                             mode=modes[index],
                        rank=ranks[index],
                             filters=filters[index]),
                 self.delays(field=field,
                        function=functions[index],
                        mode='ref',
                             rank=ranks[index],
                             filters=filters[index]) ]

        ok = [isinstance(d, datetime.datetime)  for d in test[-1] ]
        test[-1] = [d for i,d in enumerate(test[-1]) if ok[i]]
        test[0] = [d for i,d in enumerate(test[0]) if ok[i]]
        ok = [d>0 for d in test[0] ]
        test[-1] = [d for i,d in enumerate(test[-1]) if ok[i]]
        test[0] = [d for i,d in enumerate(test[0]) if ok[i]]
        
        if False:
            ax.plot(test[-1],
                      test[0],
                      '.',
                      markersize=3+(len(fields)-index)**1.5,
                      markeredgewidth=2.+(len(fields)-index)/3.,
                      zorder=-1*index+len(fields),
                    color='k',
                      )
        ax.plot(test[-1],
                test[0],
                '.',
                label='%s (%s)'%((field or ['event'])[-1] , fields[index][-1] ) ,
                markersize=3+(len(fields)-index)**1.5,
                markeredgewidth=.5+(len(fields)-index)/3.,
                zorder=index+len(fields),
                )


    ax.legend(legendlabels,
              prop={'size':'small'},
              loc='lower left')
    ax.set_yscale('log')
    ax.set_ylabel('Delays after origin time (s)')
    ax.text(0,1,titleaddons,
            ha='left',
            va='bottom',
            transform=ax.transAxes)
    ax.grid()
    ax.xaxis_date()
    ax.get_figure().autofmt_xdate()
    matplotlib.pyplot.xticks(rotation=20,
                                 ha='right')
    if eew:
        ax.set_ylim([1., 100.])
    return fig









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
                        d=obspy_addons.haversine(preferred_origin_longitude,
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

def eewtlines(self=obspy.core.event.catalog.Catalog(),
              last=0,
              xlims=[3,30],
              agency_id=['*'],
              magnitude_type=['MVS','Mfd'],
              log=None,
              minlikelyhood=None,
              ax1=None,
              f=None):
    
    solutions_first = Solutions(catalog=self,last=0, arrivals=0, magnitude_type=magnitude_type, agency_id=agency_id,minlikelyhood=minlikelyhood)
    solutions_all = Solutions(catalog=self,last='*', arrivals=0, magnitude_type=magnitude_type, agency_id=agency_id,minlikelyhood=minlikelyhood)
    
    if ax1:
        if not f:
            f = ax1.get_figure()
    else:
        if not f:
            f = matplotlib.pyplot.figure()
        ax1 = f.add_subplot(111)
    ax2=ax1.twinx()

    pos1 = ax1.get_position() # get the original position
    pos2 = [pos1.x0 , pos1.y0+pos1.height*.5,  pos1.width, pos1.height*.5]
    ax1.set_position(pos2)

    pos2 = [pos1.x0 , pos1.y0,  pos1.width, pos1.height*.5]
    ax2.set_position(pos2)
    
    
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
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
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
        ax2.set_xlim(xlims)
        ax2.set_ylim([-1.1,1.1])
        ax1.set_ylim([1,100])


            #lg = matplotlib.pyplot.legend((ob1, ob2),
            #                          ('Solutions (loc. or M)', 'Picks (t or A)'),
            #                          numpoints=1,
            #                          scatterpoints=1,
            #                          fancybox=True,
            #                          loc=4)
    return f


