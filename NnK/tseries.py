# -*- coding: utf-8 -*-
"""
tseries - Module for time series 
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

# moved to trigger.py
# def streamdatadim(a):
#     """
#     Calculate the dimensions of all data in stream.

#     Given a stream (obspy.core.stream) calculate the minimum dimensions 
#     of the array that contents all data from all traces.

#     This method is written in pure Python and gets slow as soon as there
#     are more then ... in ...  --- normally
#     this does not happen.

#     :type a: ObsPy :class:`~obspy.core.stream`
#     :param a: datastream of e.g. seismogrammes.
#     :rtype: array
#     :return: array of int corresponding to the dimensions of stream.
#     """
#     # 1) Does this
#     # 2) Then does 
#     #    that

#     nmax=0
#     for t, tr in enumerate(a):
#         nmax = max((tr.stats.npts, nmax))

#     return (t+1, nmax)

