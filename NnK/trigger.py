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
from obspy.core.stream import Stream

import tseries 

def correlate_components(stream, scales=None):
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
    (tmax,nmax) = tseries.streamdatadim(stream)
    if scales is None:
        scales = [2**i for i in range(3,999) if 2**i < (nmax - 2**i)]
        scales = np.require(scales, dtype=np.int) 
    
    # Initialize results at the minimal size
    STrms_LTrms = np.ones(( tmax, nmax )) 
    stream_copy = stream.copy()


    for t, trace in enumerate(stream):

        station_traces = stream.select(network=trace.stats.network, 
            station=trace.stats.station, 
            location=trace.stats.location, 
            channel=(trace.stats.channel)[:-1]+'*')
        
        rms_ts, scales_rms = tseries.moving(station_traces, scales=scales)

        for tc, trace_component in enumerate(station_traces):

            df = trace_component.stats.sampling_rate
            npts = trace_component.stats.npts
            time = np.arange(npts, dtype=np.float32) / df

            npairs = 0.
            for scale, rms in enumerate(rms_ts[tc]):
                for largerscale, largerscale_rms in enumerate(rms_ts[tc]):
                    if scale**1.5 <= largerscale:
                        npairs +=1.
                        STrms_LTrms[t][0:npts] += ( rms[0:npts] / largerscale_rms[0:npts] )**2
            
            STrms_LTrms[t][0:npts] = STrms_LTrms[t][0:npts]**(1/npairs)

            # plot
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            fig.suptitle(trace_component.id)
      #      plt.yscale('log', nonposy='clip')

            ax1.plot(time, trace_component.data, 'k')
            ax2.plot(time, STrms_LTrms[t][0:npts], 'k')


            plt.show()
        break
            

        


    #return corr_mat














