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


def trace2stream(trace_or_stream):

    if isinstance(trace_or_stream, Stream):
        newstream = trace_or_stream
    elif isinstance(trace_or_stream, Trace):
        newstream = Stream()
        newstream.append(trace_or_stream)
    else:
        try:
            dims = trace_or_stream.shape
            if len(dims) == 3:
                newstream = trace_or_stream
            elif len(dims) == 2  :
                newstream = np.zeros(( 1, dims[0], dims[1] )) 
                newstream[0] = trace_or_stream
        except:
            raise Exception('I/O dimensions: only obspy.Stream or obspy.Trace input supported.')

    return newstream, trace_or_stream 


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
        
    # data: obspy.stream (|obspy.trace ####################################################################### TODO)
    # si c'est une trace revoyer un truc en (1, nscale, npoints)
    # si c'est un stream revoyer un truc en (nchan, nscale, npoints)

    a, input_a = trace2stream(a)
    
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
                csqr = np.cumsum(np.abs(tr.detrend('linear').data))        

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
                        # (vectorization ####################################################################### TODO)
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


def stream_processor_plot(stream,cf):
    
    fig = plt.figure()#figsize=plt.figaspect(1.2))
    ax = fig.gca() 
    (tmax,nmax) = streamdatadim(stream)
    labels = ["" for x in range(tmax)]
    for t, trace in enumerate(stream):
        df = trace.stats.sampling_rate
        npts = trace.stats.npts
        time = np.arange(npts, dtype=np.float32) / df
        labels[t] = trace.id
        ax.plot(time, t+trace.data/(2*np.max(np.abs(trace.data))), 'k')
        ax.plot(time, t-.5+cf[t][0:npts]/(np.max(np.abs(cf[t][0:npts]))), 'r')         

    plt.yticks(np.arange(0, tmax, 1.0))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    plt.axis('tight')
    #plt.ylim( -0.5, t+0.5 ) 
    plt.tight_layout()

    return ax


def stream_multiplexor_plot(stream,cf):
    
    fig = plt.figure()#figsize=plt.figaspect(1.2))
    ax = fig.gca() 
    (tmax,nmax) = streamdatadim(stream)
    labels = ["" for x in range(tmax)]
    for t, trace in enumerate(stream):
        df = trace.stats.sampling_rate
        npts = trace.stats.npts
        time = np.arange(npts, dtype=np.float32) / df
        labels[t] = trace.id
        ax.plot(time, t+trace.data/(2*np.max(np.abs(trace.data))), 'k')

        for c, channel in enumerate(cf[0][t]):
            if np.sum(cf[1][t][c][0:npts]) != 0 :
                ax.plot(time, t-.5+cf[0][t][c][0:npts]/(np.max(np.abs(cf[0][t][c][0:npts]))), 'r')        
                ax.plot(time, t-.5+cf[1][t][c][0:npts]/(np.max(np.abs(cf[1][t][c][0:npts]))), 'b')       

    plt.yticks(np.arange(0, tmax, 1.0))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    plt.axis('tight')
    #plt.ylim( -0.5, t+0.5 ) 
    plt.tight_layout()

    return ax


class Ratio(object):
    def __init__(self, pre_processed_data, data=None):

        self.data = data
        self.pre_processed_data = pre_processed_data[0]
        self.enhancement_factor = pre_processed_data[1]

    def output(self):

        dtiny = np.finfo(0.0).tiny
        (tmax,nmax) = streamdatadim(self.data)
        cf = np.ones(( tmax, nmax ))  

        for station_i, station_data in enumerate(self.pre_processed_data[0]):
            for enhancement_i, enhancement_data in enumerate(self.pre_processed_data[0][station_i]):

                buf = self.pre_processed_data[0][station_i][enhancement_i] 
                buf /= self.pre_processed_data[1][station_i][enhancement_i]
                # no ~zeros
                buf[buf < dtiny] = dtiny
                # product enhancement (no nans, no infs)
                cf[station_i][np.isfinite(buf)] *= buf[np.isfinite(buf)]

            # rescaling 
            cf[station_i] **= (1./self.enhancement_factor) 

        # ###################################################################################### why is this not ok ??????
        # ratio = self.pre_processed_data[0] / self.pre_processed_data[1]
        
        # # no ~zeros
        # ratio[ ratio < dtiny ] = dtiny
        # # no nans, no infs
        # ratio[ ~np.isfinite(ratio) ] = 1.

        # # returns product enhanced and rescaled
        # cf = (np.prod(ratio, axis=1))**(1/self.enhancement_factor)

        return cf

    def plot(self):

        return stream_processor_plot( self.data, self.output()  )


class ShortLongTerms(object):
    # Multiplex the data after pre-process

    def __init__(self, data, windowlengths=None, statistic='average'): 

        # get (station, scale, sample) array any way (for Trace & Stream inputs)
        self.data, self.original_data = trace2stream(data)

        # pre_processed: array of pre-processed data
        # statistic: sum|average|rms
        # windowlengths: list
        self.pre_processed, self.windowlengths = recursive(data, windowlengths, statistic) 

        # stores input parameters
        self.statistic = statistic

        self.ratio = Ratio(self.output(), self.data)

    def output(self):
        # Multiplex the pre-processed data
        # return cf as (prod( STA/LTA ))^1/N
    
        # Initialize results at the minimal size
        (tmax,nmax) = streamdatadim(self.data)
        nscale = len(self.windowlengths)
        channels = np.ones(( 2, tmax, nscale**2, nmax ))  # np.zeros(( tmax, nmax )) 
        dtiny = np.finfo(0.0).tiny
        
        # along stations
        for station_i, station_data in enumerate(self.pre_processed):
            n_enhancements = -1
            # along scales
            for smallscale_i, smallscale_data in enumerate(station_data):
                for bigscale_i, bigscale_data in enumerate(station_data):

                    if self.windowlengths[smallscale_i] < self.windowlengths[bigscale_i] :

                        n_enhancements += 1
                        # no divide by ~zeros
                        bigscale_data[bigscale_data < dtiny] = dtiny

                        channels[0][station_i][n_enhancements] = smallscale_data
                        channels[1][station_i][n_enhancements] = bigscale_data

            #             buf = smallscale_data / bigscale_data
            #             # no ~zeros
            #             buf[buf < dtiny] = dtiny
            #             # product enhancement (no nans, no infs)
            #             cf[station_i][np.isfinite(buf)] *= buf[np.isfinite(buf)]
            # # rescaling 
            # cf[station_i] **= (1./n_enhancements) 

        for i in range(n_enhancements+1, nscale**2):
            channels = np.delete(channels, n_enhancements+1, axis=2)

        return channels, n_enhancements+1


    def plot(self):
        channels, n = self.output() 
        return stream_multiplexor_plot( self.data, channels )




        


# class Ratio(object):
#     # Processor, returns characteristic function using ratio operator 
    
#     def __init__(self, data, windowlengths=None, statistic='average'):
        
#         self.data = data

#         # add multiplexor
#         self.ShortLongTerms = ShortLongTerms( data, windowlengths,  statistic )

#     def cf(self):
#         return self.ShortLongTerms.cf()

#     def plot(self):
#         return stream_cf_plot( self.data, self.cf() )

#     # def RightLeftTerms(self):
#     #     # Multiplex the pre-processed data
#     #     # return cf as (prod( RT/LT ))^1/N
#     #     cf = []
#     #     return cf

#     # def MultiComponents(self):
#     #     # Multiplex the pre-processed data
#     #     # return cf as (prod( ZT/HT ))^1/N
#     #     cf = []
#     #     return cf

#     # def Plot(self):
        
        


# class Correlate():
#     # ratio proc
#     def __init__(self):
#     def StLt(self):
#     def LtRt(self):
#     def Components(self):
#         # return correlation

# class Multiscale():
#     # scaling
#     def __init__(self)::
#     def StLt(self):
#     def LtRt(self):
#         # return muliscaled proc

# class Derivate():
#     # post-proc
#     def __init__(self):
#         pass
#     def Multiscale(self):
#         # return derivative

# class Kurtosis():
#     # post-proc
#     def __init__(self):
#         pass
#     def Multiscale(self):
#         # return Kurtosis

