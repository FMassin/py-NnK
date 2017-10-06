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
    from . import obspy_addons
except:
    try:
        import eew.obspy_addons as obspy_addons
    except:
        import NnK.eew.obspy_addons as obspy_addons




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
                eq_specs = {'starttime':str(o.time-30),
                            'endtime':str(o.time+30),
                            'minlatitude':o.latitude-1,
                            'maxlatitude':o.latitude+1,
                            'minlongitude':o.longitude-1,
                            'maxlongitude':o.longitude+1,
                            'includearrivals':True}
                try:
                    tofind = client.get_events( **eq_specs, **get_events_args )
                except:
                    if v:
                        print('No event in reference client')
                    continue
            
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
              vmax=None,
              vmin=None,
               prospective_inventory=None,
               latencies=None,
               flat_delay=2,
               VpVsRatio=1.75,
               model_correction='iasp91',
               Vs=3.428,
               extra=None,
               extramarker='1',
               extraname='',
               fp=False):


    cf=[]
    mags = get(self,'magnitudes','mag',['b'] )
    times = get(self, 'origins','time', types=['b'])
    
    if prospective_inventory:
        
        if eqcolorfield[0] == 't':
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
        
        elif eqcolorfield[0] == 't' and eqcolorfield[1:] in [str(int(n)) for n in range(100)]:
            eqcolorlabel = 'Travel time to %s$^{th}$ station (s'%(eqcolorfield[1:])
            if latencies:
                eqcolorlabel += 'delays incl.)'
            else:
                eqcolorlabel += ')'
            ntht=list()
            for ie,e in enumerate(self.events):
                t=[9999999999999999 for d in range(100)]
                for o in e.origins:
                    if str(e.preferred_origin_id) == str(o.resource_id):
                        d=[9999999999999999 for d in range(100)]
                        if not prospective_inventory:
                            for a in o.arrivals:
                                if a.phase[0] == 'P':
                                    for p in e.picks:
                                        if str(a.pick_id) == str(p.resource_id) :
                                            t.append(p.time - o.time)
                                            if latencies:
                                                mlatency = numpy.median(latencies[p.waveform_id.network_code+'.'+p.waveform_id.station_code])
                                                t[-1] += mlatency+flat_delay
                    

                        else:
                            for istation,lonstation in enumerate(stations_longitudes):
                                latstation =  stations_latitudes[istation]
                                ep_d =  numpy.sqrt((lonstation-o.longitude)**2+(latstation-o.latitude)**2)
                                d.append(ep_d)
                            d = numpy.sort(d)
                            for ep_d in  d[:int(eqcolorfield[1:])]:
                                
                                arrivals = model.get_travel_times(o.depth/1000.,
                                                                  distance_in_degree=ep_d,
                                                                  phase_list=['ttp'],
                                                                  receiver_depth_in_km=0.0)
                                try:
                                    t.append( numpy.nanmin([ a.time for a in arrivals ]))
                                except:
                                    print('No phase for depth',o.depth/1000.,'and distance',ep_d)
                                    pass
            
                t = numpy.sort(t)
                ntht.append( t[int(eqcolorfield[1:])-1] )
                
            
            ntht = numpy.asarray(ntht)
            ntht[ntht<0.]=0.
            eqcolor = ntht
            vmax=40.
            vmin=0.

        elif eqcolorfield[0] == 'I' and eqcolorfield[1:] in [str(int(n)) for n in range(100)]:
            eqcolorlabel = 'Intensity at %s$^{th}$ station'%(eqcolorfield[1:])
            if latencies:
                eqcolorlabel += ' (delays incl.)'
            if int(eqcolorfield[1:])==0:
                eqcolorlabel = 'Epicentral intensity'
            r=list()
            error=list()
            for ie,e in enumerate(self.events):
                d=[9999999999999999 for d in range(100)]
                for o in e.origins:
                    if str(e.preferred_origin_id) == str(o.resource_id):
                        if not prospective_inventory:
                            for a in o.arrivals:
                                if 'p' in str(a.phase).lower():
                                    for p in e.picks:
                                        if str(a.pick_id) == str(p.resource_id) :
                                            d.append( numpy.sqrt((a.distance*110.)**2+(o.depth/1000.)**2)/VpVsRatio )
                                            
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
                if int(eqcolorfield[1:])==0:
                    r.append( o.depth/1000. )
                else:
                    r.append( d[int(eqcolorfield[1:])-1] )
            r = numpy.asarray(r)
            r[r<1.]=1.
            eqcolor = obspy_addons.ipe_allen2012_hyp(r,
                                                     numpy.asarray(mags))
            
            eqcolor[numpy.isnan(eqcolor)] = numpy.nanmin(eqcolor[eqcolor>.5])
            eqcolor[eqcolor<.5] =  numpy.nanmin(eqcolor[eqcolor>.5])
            cmap='jet'
            vmin=.5
            vmax=9.

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
        
        cf = bmap.scatter([longitudes[i] for i in numpy.argsort(eqcolor)],
                              [latitudes[i] for i in numpy.argsort(eqcolor)],
                              [sizes[i] for i in numpy.argsort(eqcolor)],
                                [eqcolor[i] for i in numpy.argsort(eqcolor)],
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
                                      width=.0182*sizes[i]**.525,
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

            if eqcolorfield[0] == 'I' and eqcolorfield[1:] in [str(int(n)) for n in range(100)]:
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
        for o in e.origins:
            if str(e.preferred_origin_id) == str(o.resource_id):
                if depth_correction:
                    z=o.depth/1000.
                    if o.depth/1000. < .1:
                        z=.1
                    arrivals_correction = model.get_travel_times(z,
                                                                 distance_in_degree=0.,#0001,
                                                      phase_list=['p','s'],
                                                      receiver_depth_in_km=0.0)
                t=list()
                for a in o.arrivals:
                    for p in e.picks:
                        if str(a.pick_id) == str(p.resource_id) :
                            t.append(p.time)
                
                for ia in numpy.argsort(t):
                    a=o.arrivals[ia]
                    for p in e.picks:
                        if str(a.pick_id) == str(p.resource_id) and p.time-o.time>0:
                            k = '%s.%s'%(p.waveform_id.network_code,p.waveform_id.station_code)
                            if a.phase[0] == 'S':
                                ns+=1
                                n[ns][ie] = p.time-o.time
                                ndt[ns][ie] = p.time-o.time+flat_latency+latencies_type(latencies[k])
                                mags[np][ie] = mag
                                if depth_correction:
                                    tnadir = arrivals_correction[1].time
                                    thoriz = ((p.time-o.time)**2-tnadir**2)**.5
                                    n[ns][ie] = (thoriz**2+tsz_correction**2)**.5
                                    ndt[ns][ie] = (thoriz**2+tsz_correction**2)**.5+flat_latency+latencies_type(latencies[k])
        
                            elif a.phase[0] == 'P':
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
                        
                break
    
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
        ax.set_xlabel('Observed travel Time')
        if depth_correction:
            ax.set_xlabel('Observed travel Time (corr. to depth %s km)'%(depth_correction))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        minorLocator = matplotlib.ticker.MultipleLocator(1)
        ax.yaxis.set_minor_locator(minorLocator)
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
                      loc=4)
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
