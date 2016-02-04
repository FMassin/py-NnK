# -*- coding: utf-8 -*-
"""
trigger - Module to improve obspy.signal.trigger
======================================================
This module ...

.. figure:: /_images/Event.png

.. note::

    For ...

:copyright:
    The ...
:license:
    ...
"""

# def ggg(...):
#         """
#         Run a ...
#
#         :param type: String that specifies which trigger is applied (e.g.
#             ``'recstalta'``).
#         :param options: Necessary keyword arguments for the respective
#             trigger.
#
#         .. note::
#
#             The raw data is not accessible anymore afterwards.
#
#         .. rubric:: _`Supported Trigger`
#
#         ``'classicstalta'``
#             Computes the classic STA/LTA characteristic function (uses
#             :func:`obspy.signal.trigger.classicSTALTA`).
#
#         .. rubric:: Example
#
#         >>> ss.ssss('sss', ss=1, sss=4)  # doctest: +ELLIPSIS
#         <...aaa...>
#         >>> aaa.aaa()  # aaa
#
#         .. plot::
#
#             from ggg import ggg
#             gg = ggg()
#             gg.ggggg("ggggg", ggggg=3456)
#             gg.ggg()
#         """


import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Trace, Stream




def streamdatadim(a):
    """
    Calculate the dimensions of all data in stream.

    Given a stream (obspy.core.stream) calculate the minimum dimensions 
    of the array that contents all data from all traces.

    This method is written in pure Python and gets slow as soon as there
    are more then ... in ...  --- normally
    this does not happen.

    :type a: ObsPy :class:`~obspy.core.stream`
    :param a: datastream of e.g. seismogrammes.
    :rtype: array
    :return: array of int corresponding to the dimensions of stream.
    """
    # 1) Does this
    # 2) Then does 
    #    that

    nmax=0
    for t, tr in enumerate(a):
        nmax = max((tr.stats.npts, nmax))

    return (t+1, nmax)


def recursive(a, scales=None, operation=None):
    """
    recursive (sum|average|rms) performs calculation by 
    creating series of operations of different subsets of 
    the full data set. This is also called rolling 
    operation.

    :type a: ObsPy :class:`~obspy.core.stream`
    :param a: datastream of e.g. seismogrammes.
    :type scales: vector
    :param scales: scale(s) of timeseries operation.
    :type operation: string
    :param operation: type of operation.
    :rtype: array
    :return: array of root mean square time series, scale 
             in r.stats._format (samples scale unit).
    """
    # 1) Iterate on channels
    # 2) Pre calculate the common part of all scales
    # 3) Perform channel calculation 

    import copy
    from obspy.core.stream import Stream
    import numpy as np
    
    # Initialize multiscale if undefined
    if operation is None:
        operation = 'rms'
    (tmax,nmax) = streamdatadim(a)
    if scales is None:
        scales = [2**i for i in range(3,999) if 2**i < (nmax - 2**i)]
        scales = np.require(scales, dtype=np.int) 

    # Initialize results at the minimal size
    timeseries = np.zeros(( tmax, len(scales), nmax )) 

    for t, tr in enumerate(a) : # the channel-wise calculations         
            
        # Avoid clock channels 
        if not tr.stats.channel == 'YH':
              
            if operation is 'rms':  
                # The cumulative sum can be exploited to calculate a moving average (the
                # cumsum function is quite efficient)
                csqr = np.cumsum(tr.detrend('linear').data ** 2)        

            if (operation is 'average') or  (operation is 'sum'):  
                # The cumulative sum can be exploited to calculate a moving average (the
                # cumsum function is quite efficient)
                csqr = np.cumsum(tr.detrend('linear').data)        

            # Convert to float
            csqr = np.require(csqr, dtype=np.float)

            for n, s in enumerate(scales) :
                
                # Avoid scales when too wide for the current channel
                if (s < (tr.stats.npts - s)) :    
                    
                    # Compute the sliding window
                    if (operation is 'rms') or (operation is 'average') or (operation is 'sum'):  
                        timeseries[t][n][s:tr.stats.npts] = csqr[s:] - csqr[:-s]

                        # for average and rms only 
                        if operation is not 'sum':
                            timeseries[t][n][:] /= s                    

                        # Pad with modified scale definitions
                        timeseries[t][n][0] = csqr[0]
                        for x in range(1, s-1):
                            timeseries[t][n][x] = (csqr[x] - csqr[0])

                            # for average and rms only
                            if operation is not 'sum':
                                timeseries[t][n][x] = timeseries[t][n][x]/(1+x)

                    # Avoid division by zero by setting zero values to tiny float
                    dtiny = np.finfo(0.0).tiny
                    idx = timeseries[t][n] < dtiny
                    timeseries[t][n][idx] = dtiny 
    
    return timeseries, scales 





def multiscale(stream, scales=None, cf_low_type=None, cf_algo=None, corr_scales=None):
    """
    Performs correlation between the components of the
    same seismometers by rolling operation.
    ______________________________________________________________________


    :type a: ObsPy :class:`~obspy.core.stream`
    :param a: datastream of e.g. seismogrammes.
    :type scales: vector
    :param scales: scale(s) of correlation operation.
    :type operation: string
    :param operation: type of operation.
    :rtype: array
    :return: array of root mean square time series, scale 
        in r.stats._format (samples scale unit).

    .. note::

        This trigger has been concepted to stick with obspy.signal.trigger.coincidenceTrigger

    .. rubric:: _`Supported Trigger`

    ``'classicstalta'``
        Computes the classic STA/LTA characteristic function (uses
        :func:`obspy.signal.trigger.classicSTALTA`).

    .. rubric:: Example

    >>> ss.ssss('sss', ss=1, sss=4)  # doctest: +ELLIPSIS
    <...aaa...>
    >>> aaa.aaa()  # aaa

    .. plot::

        from ggg import ggg
        gg = ggg()
        gg.ggggg("ggggg", ggggg=3456)
        gg.ggg()
    """
    # 1) Calculate moving rms and average, see moving()
    # 2) ...
    # 3) ... 

    # Initialize multiscale if undefined
    if cf_algo is None:
        cf_algo = 'l'
    if cf_low_type is None:
        cf_low_type='rms'

    (tmax,nmax) = streamdatadim(stream)
    if scales is None:
        scales = [2**i for i in range(3,999) if 2**i < (nmax - 2**i)]
        scales = np.require(scales, dtype=np.int) 
    
    # Initialize results at the minimal size
    multiscale_cf = np.ones(( tmax, nmax ))  # np.zeros(( tmax, nmax )) 
    stream_white = stream.copy()
    for t, trace in enumerate(stream_white):
        stream_white[t].data = np.random.normal(0., 0.00000000001+np.median(np.abs(trace.data)), (trace.data).shape )  # np.ones((trace.data).shape)

    # Pre-calculation
    white, white_scale = recursive(stream_white, scales=scales, operation=cf_low_type) #
    cf_low, cf_low_scale = recursive(stream, scales=scales, operation=cf_low_type) #rms|sum|average
    cf_low_scale = np.asarray(cf_low_scale)
    
    if corr_scales is None:
        corr_scales = cf_low_scale

    for t, trace in enumerate(stream):

        df = trace.stats.sampling_rate
        npts = trace.stats.npts
        time = np.arange(npts, dtype=np.float32) / df


        # multiscale standart STrms/LTrms
        if cf_algo in ('l', 'ST LT') :
             
            ## calc
            n_scale = 0.
            for scale, cf in enumerate(cf_low[t]):
                for largerscale, larger_cf in enumerate(cf_low[t]):
                    if scale**1.5 < largerscale:
                        n_scale +=1.
                        
                        # white_cf =  white[t][scale] / white[t][largerscale]
                        # white_cf[ ~np.isfinite(white_cf) ] = 0 
                        # processed_cf[processed_cf <= np.max(white_cf)] = 0.

                        # processed_cf = cf / larger_cf
                        # processed_cf = processed_cf**2
                        # multiscale_cf[t][np.isfinite(processed_cf)] += processed_cf[np.isfinite(processed_cf)] 

                        #print corr_scales[largerscale], cf_low_scale[scale], cf_low_scale[largerscale]
                        processed_cf = correlationcoef(cf, larger_cf, scales=[2*corr_scales[scale]])
                        multiscale_cf[t][np.isfinite(processed_cf)] *= processed_cf[np.isfinite(processed_cf)]  

            
            #multiscale_cf[t][0:npts] = np.sqrt(multiscale_cf[t][0:npts])

        # multiscale RTrms/LTrms
        elif cf_algo in ('r', 'RW LW') :
             
            ## calc
            n_scale = 0.
            for scale, cf in enumerate(cf_low[t]):
                n_scale +=1.
                
                # white_cf =  white[t][scale] / white[t][largerscale]
                # white_cf[ ~np.isfinite(white_cf) ] = 0 
                # processed_cf[processed_cf <= np.max(white_cf)] = 0.

                processed_cf = np.zeros((cf_low[t][scale]).shape)
                processed_cf[cf_low_scale[scale]:] = cf_low[t][scale][cf_low_scale[scale]:] / cf_low[t][scale][0:-cf_low_scale[scale]]
                processed_cf = processed_cf**2
                multiscale_cf[t][np.isfinite(processed_cf)] += processed_cf[np.isfinite(processed_cf)] 
            
            multiscale_cf[t][0:npts] = np.sqrt(multiscale_cf[t][0:npts])
            

        # station_traces = stream.select(network=trace.stats.network, 
        #     station=trace.stats.station, 
        #     location=trace.stats.location, 
        #     channel=(trace.stats.channel)[:-1]+'*')
        

        # for tc, trace_component in enumerate(station_traces):

    return multiscale_cf
    
            
def correlationcoef(a, b, scales=None, maxscales=None):

    na = len(a)
    if maxscales is None:
        maxscales = na

    if scales is None:
        scales = [2**i for i in range(2,999) if ((2**i <= (maxscales - 2**i)) and (2**i <= (na - 2**i)))]
    
    scales = np.require(scales, dtype=np.int) 
    scales = np.asarray(scales)
    nscale = len(scales)

    cc = np.ones(a.shape)
    prod_cumsum = np.cumsum( a * b )
    a_squarecumsum = np.cumsum( a**2 )
    b_squarecumsum = np.cumsum( b**2 )

    for s in scales :

        scaled_prod_cumsum = prod_cumsum[s:] - prod_cumsum[:-s]
        scaled_a_squarecumsum = a_squarecumsum[s:] - a_squarecumsum[:-s]
        scaled_b_squarecumsum = b_squarecumsum[s:] - b_squarecumsum[:-s]
        
        scaled_prod_cumsum[scaled_prod_cumsum == 0 ] = 1
        scaled_a_squarecumsum[scaled_a_squarecumsum == 0 ] = 1
        scaled_b_squarecumsum[scaled_b_squarecumsum == 0 ] = 1

        cc[:-s] *= (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

        # pading with modified def
        scaled_prod_cumsum = prod_cumsum[na-1] - prod_cumsum[-s+1:]
        scaled_a_squarecumsum = a_squarecumsum[na-1] - a_squarecumsum[-s+1:]
        scaled_b_squarecumsum = b_squarecumsum[na-1] - b_squarecumsum[-s+1:]
        
        scaled_prod_cumsum[scaled_prod_cumsum == 0 ] = 1
        scaled_a_squarecumsum[scaled_a_squarecumsum == 0 ] = 1
        scaled_b_squarecumsum[scaled_b_squarecumsum == 0 ] = 1

        cc[-s+1:] *= (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

    return cc**(1./nscale)


def stream_cf_plot(stream,cf):
    
    fontsize = 12

    fig = plt.figure()#figsize=plt.figaspect(1.2))
    ax = fig.gca() 
    (tmax,nmax) = streamdatadim(stream)
    labels = ["" for x in range(tmax)]
    for t, trace in enumerate(stream):
        df = trace.stats.sampling_rate
        npts = trace.stats.npts
        time = np.arange(npts, dtype=np.float32) / df
        ## plot
        #fig.suptitle(trace.id)
        #ax.annotate(trace.id, xy=(0, t), xytext=(0, t+1./6.))
        ax.plot(time, t+trace.data/(2*np.max(np.abs(trace.data))), 'k')
        #ax.plot(time, t+stream_white[t].data/(2*np.max(np.abs(trace.data))), 'g')
        ax.plot(time, t-.5+cf[t][0:npts]/(np.max(np.abs(cf[t][0:npts]))), 'r') # 
        labels[t] = trace.id

    plt.yticks(np.arange(0, tmax, 1.0))
    ax.set_yticklabels(labels)


    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_ylabel('Channel', fontsize=fontsize)
    plt.axis('tight')
    plt.ylim( -0.5, t+0.5 ) 
    plt.tight_layout()

    return ax


class StLt():
    # short long terms 
    def __init__(self):
        pass

    def RMS(self):
        # return rec rms

    def Average(self):
        # return rec average

class LtRt():
    # right left terms multiplexing
    def __init__(self):
        pass

    def RMS(self):
        # return rec rms

    def Average(self):
        # return rec average

class Components():
    # components terms multiplexing
    def __init__(self):
        pass

    def RMS(self):
        # return rec rms

    def Average(self):
        # return rec average

class Ratio():
    # ratio proc
    def __init__(self):
        self.StLt = StLt(self)
        self.LtRt = LtRt(self)
        self.Components = Components(self)
        # return ratio

class Correlate():
    # ratio proc
    def __init__(self):
        self.StLt = StLt(self)
        self.LtRt = LtRt(self)
        self.Components = Components(self)
        # return correlation

class Multiscale():
    # scaling
    def __init__(self):
        self.Ratio = StLt(self)
        self.Correlation = LtRt(self)
        # return muliscaled proc

class Derivate():
    # post-proc
    def __init__(self):
        self.Multiscale = Multiscale(self)
        # return derivative

class Kurtosis():
    # post-proc
    def __init__(self):
        self.Multiscale = Multiscale(self)
        # return Kurtosis

