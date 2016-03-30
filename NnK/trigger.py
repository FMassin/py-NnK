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
# 	"""
# 	Plot the given seismic wave radiation pattern as a color-coded surface 
# 	or focal sphere (not exactly as a beach ball diagram).
# 	______________________________________________________________________

# 	:param type: String that specifies which trigger is applied (e.g.
# 	    ``'recstalta'``).
# 	:param options: Necessary keyword arguments for the respective
# 		trigger.

# 	.. note::

# 		The raw data is not accessible anymore afterwards.

# 	.. rubric:: _`Supported Trigger`

# 	``'classicstalta'``
# 		Computes the classic STA/LTA characteristic function (uses
# 		:func:`obspy.signal.trigger.classicSTALTA`).

# 	.. rubric:: Example

# 		>>> ss.ssss('sss', ss=1, sss=4)  # doctest: +ELLIPSIS
# 		<...aaa...>
# 		>>> aaa.aaa()  # aaa

# 	.. plot::

# 		from ggg import ggg
# 		gg = ggg()
# 		gg.ggggg("ggggg", ggggg=3456)
# 		gg.ggg()
# 	"""

import re
import copy
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Trace, Stream
from obspy.core.trace import Stats
from obspy.signal.filter import highpass
from source import spherical_to_cartesian
import scipy.signal


def artificial_stream(npts=1000., noise=1., P=[7., 15., 3.**.5], S=[9., 15., 3.**.5]) : 
	"""
	Generate artificial data for testing purposes.
	______________________________________________________________________
	:param npts: data length in samples.
	:param noise: amplitude of noise.
	:param P: P-wave properties [amplitude, frequency, H/Z ratio]
	:param S: S-wave properties [amplitude, frequency, Z/H ratio]
	
	:rtype: stream
	:return: artificial noisy data with [noise only, frequency change, 
		amplitude change, polarization change].
	
	.. note::

		The noise is a random 3d motion projection on 3 cartesian 
		components.

		The amplitude decays are linear and the body-wave span a third of 
		the signal duration (npts/6 samples each).

	.. rubric:: Example

		>>> import trigger
		>>> a = trigger.artificial_stream(npts=1000)
		>>> print a

	.. plot::

		>>> import trigger
		>>> plotTfr((a[1]).data, dt=.01, fmin=0.1, fmax=25)
		>>> plotTfr((a[2]).data, dt=.01, fmin=0.1, fmax=25)
		>>> fig = plt.figure()
		>>> ax = fig.gca(projection='3d')
		>>> ax.plot(a[5].data, a[6].data, a[7].data, label='noise', alpha=.5, color='g')
		>>> ax.plot(a[5].data[npts/3:npts/2],a[6].data[npts/3:npts/2],a[7].data[npts/3:npts/2], label='P', color='b')
		>>> ax.plot(a[5].data[npts/2:npts*2/3],a[6].data[npts/2:npts*2/3],a[7].data[npts/2:npts*2/3], label='S', color='r')
		>>> ax.legend()
		>>> plt.show()
	"""

	Fs = npts/10.
	Fnl = npts/30.
	npts_c = npts+ Fnl
	Pspot = range(npts/3,npts/2)
	Sspot = range(npts/2,npts*2/3)

	stats3_z = Stats({'network':"Test", 'station':"P", 'location':"", 'channel':"Z", 'npts':npts, 'delta':1/Fs})
	stats3_e = Stats({'network':"Test", 'station':"P", 'location':"", 'channel':"E", 'npts':npts, 'delta':1/Fs})
	stats3_n = Stats({'network':"Test", 'station':"P", 'location':"", 'channel':"N", 'npts':npts, 'delta':1/Fs})
	stats2_z = Stats({'network':"Test", 'station':"F", 'location':"", 'channel':"Z", 'npts':npts, 'delta':1/Fs})
	stats2_e = Stats({'network':"Test", 'station':"F", 'location':"", 'channel':"E", 'npts':npts, 'delta':1/Fs})
	stats2_n = Stats({'network':"Test", 'station':"F", 'location':"", 'channel':"N", 'npts':npts, 'delta':1/Fs})
	stats1_z = Stats({'network':"Test", 'station':"A", 'location':"", 'channel':"Z", 'npts':npts, 'delta':1/Fs})
	stats1_e = Stats({'network':"Test", 'station':"A", 'location':"", 'channel':"E", 'npts':npts, 'delta':1/Fs})
	stats1_n = Stats({'network':"Test", 'station':"A", 'location':"", 'channel':"N", 'npts':npts, 'delta':1/Fs})
	stats0_z = Stats({'network':"Test", 'station':"Ns", 'location':"", 'channel':"Z", 'npts':npts, 'delta':1/Fs})
	stats0_e = Stats({'network':"Test", 'station':"Ns", 'location':"", 'channel':"E", 'npts':npts, 'delta':1/Fs})
	stats0_n = Stats({'network':"Test", 'station':"Ns", 'location':"", 'channel':"N", 'npts':npts, 'delta':1/Fs})

	noise_signal = np.asarray([ np.random.random_integers(-np.pi*1000, np.pi*1000, npts)/1000. , 
		np.random.random_integers(-np.pi*1000, np.pi*1000, npts)/1000. , 
		np.random.random_integers(-noise*500, noise*500, npts)/1000. ] )

	noise_signal = np.asarray(spherical_to_cartesian(noise_signal))

	# change the noise pol polarization to isotropic to vertical (P) and horizontal (S)
	pola = np.copy(noise_signal)
	## onde P
	common = np.random.random_integers(-50., 50., len(Pspot)+2)/100.

	p_signal = np.asarray([ np.random.random_integers(-np.pi*1000, np.pi*1000, len(Pspot))/1000. , 
		np.random.random_integers(np.pi*1000/1.3, np.pi*1000/0.7, len(Pspot))/1000. , 
		np.random.random_integers(-noise*500, noise*500, len(Pspot))/1000. ] ) 
	p_signal = np.asarray(spherical_to_cartesian(p_signal))
	s_signal = np.asarray([ np.random.random_integers(-np.pi*1000, np.pi*1000, len(Sspot))/1000. , 
		np.random.random_integers(np.pi*1000/2.3, np.pi*1000/1.7, len(Sspot))/1000. , 
		np.random.random_integers(-noise*500, noise*500, len(Sspot))/1000. ] )
	s_signal = np.asarray(spherical_to_cartesian(s_signal))

	pola[0][Pspot] += p_signal[2]* P[0] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	pola[1][Pspot] += p_signal[1]* P[0] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	pola[2][Pspot] += p_signal[0]* P[0] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	## onde S
	pola[0][Sspot] += s_signal[2]* S[0] * (npts/6 - np.arange(len(Sspot)))/(npts/6.)
	pola[1][Sspot] += s_signal[1]* S[0] * (npts/6 - np.arange(len(Sspot)))/(npts/6.)
	pola[2][Sspot] += s_signal[0]* S[0] * (npts/6 - np.arange(len(Sspot)))/(npts/6.)

	# roughly change amplitudes at P and S wave amplitudes
	ampl = np.copy(noise_signal)

	ampl[0][Pspot] += ampl[0][Pspot] * P[0] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	ampl[0][Pspot] /= np.max(np.abs( ampl[0][Pspot] )) 
	ampl[0][Pspot] *= P[0]/2.
	ampl[1][Pspot] += ampl[1][Pspot] * P[0]*1./P[2] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	ampl[1][Pspot] /= np.max(np.abs( ampl[1][Pspot] )) 
	ampl[1][Pspot] *= P[0]*1/P[2]/2.
	ampl[2][Pspot] += ampl[2][Pspot] * P[0]*1./P[2] * (npts/6 - np.arange(len(Pspot)))/(npts/6.) 
	ampl[2][Pspot] /= np.max(np.abs( ampl[2][Pspot] )) 
	ampl[2][Pspot] *= P[0]*1/P[2]/2.

	ampl[0][Sspot] += ampl[0][Sspot] * S[0]*1./S[2]  * (npts/6 - np.arange(len(Sspot)))/(npts/6.)
	ampl[0][Sspot] /= np.max(np.abs( ampl[0][Sspot]  )) 
	ampl[0][Sspot] *= S[0]*1/S[2]/2.
	ampl[1][Sspot] += ampl[1][Sspot] * S[0]   * (npts/6 - np.arange(len(Sspot)))/(npts/6.)
	ampl[1][Sspot] /= np.max(np.abs( ampl[1][Sspot]  )) 
	ampl[1][Sspot] *= S[0]/2.
	ampl[2][Sspot] += ampl[2][Sspot] * S[0]   * (npts/6 - np.arange(len(Sspot)))/(npts/6.)
	ampl[2][Sspot] /= np.max(np.abs( ampl[2][Sspot]  )) 
	ampl[2][Sspot] *= S[0]/2.


	# roughly remap frequencies at P and S wave frequencies
	freq = np.copy(noise_signal)
	freq[0][Pspot] += P[0] * np.sin(2 * np.pi * P[1] * np.arange(len(Pspot)) / Fs)
	freq[0][Pspot] /= np.max(np.abs( freq[0][Pspot] )) 
	freq[0][Pspot] *= noise/2.
	freq[1][Pspot] += P[0]*1./P[2] * np.sin(2 * np.pi * P[1] * np.arange(len(Pspot)) / Fs)
	freq[1][Pspot] /= np.max(np.abs( freq[1][Pspot] )) 
	freq[1][Pspot] *= noise/2.
	freq[2][Pspot] += P[0]*1/P[2] * np.sin(2 * np.pi * P[1] * np.arange(len(Pspot)) / Fs)
	freq[2][Pspot] /= np.max(np.abs( freq[2][Pspot] )) 
	freq[2][Pspot] *= noise/2.

	freq[0][Sspot] += S[0]*1./S[2] * np.sin(2 * np.pi * S[1] * np.arange(len(Sspot)) / Fs)
	freq[0][Sspot] /= np.max(np.abs( freq[0][Sspot] )) 
	freq[0][Sspot] *= noise/2.
	freq[1][Sspot] += S[0] * np.sin(2 * np.pi * S[1] * np.arange(len(Sspot)) / Fs)
	freq[1][Sspot] /= np.max(np.abs( freq[1][Sspot] )) 
	freq[1][Sspot] *= noise/2.
	freq[2][Sspot] += S[0] * np.sin(2 * np.pi * S[1] * np.arange(len(Sspot)) / Fs)
	freq[2][Sspot] /= np.max(np.abs( freq[2][Sspot] )) 
	freq[2][Sspot] *= noise/2.

	a = Stream(traces=[Trace(data=noise_signal[0], header=stats0_z), \
						Trace(data=freq[0], header=stats2_z), \
						Trace(data=ampl[0], header=stats1_z), \
						Trace(data=ampl[1], header=stats1_e), \
						Trace(data=ampl[2], header=stats1_n), \
						Trace(data=pola[0], header=stats3_z), \
						Trace(data=pola[1], header=stats3_e), \
						Trace(data=pola[2], header=stats3_n)])

	return a


def streamdatadim(a):
	"""
	Given a stream (obspy.core.stream) calculate the minimum dimensions 
	of the array that contents all data from all traces.
	______________________________________________________________________

	:type a: ObsPy :class:`~obspy.core.stream`
	:param a: datastream of e.g. seismogrammes.

	:rtype: array
	:return: array of int corresponding to the dimensions of stream.
	"""

	nmax=0
	for t, tr in enumerate(a):
		nmax = max((tr.stats.npts, nmax))

	return (t+1, nmax)


def trace2stream(trace_or_stream_or_nparray):
	"""
	Given a stream (obspy.core.stream) calculate the minimum dimensions 
	of the array that contents all data from all traces.
	______________________________________________________________________

	:type trace_or_stream_or_nparray: ObsPy :class:`~obspy.core.stream` or
		`~obspy.core.trace` or NumPy :class:`~numpy.ndarray`.
	:param trace_or_stream_or_nparray: input data .

	:rtype 0: ObsPy :class:`~obspy.core.stream`.
	:return 0: stream containing copies of inputed data.
	:rtype 1: input.
	:return 1: copy of inputed data.
	"""

	if isinstance(trace_or_stream_or_nparray, Stream):
		newstream = trace_or_stream_or_nparray
	elif isinstance(trace_or_stream_or_nparray, Trace):
		newstream = Stream()
		newstream.append(trace_or_stream_or_nparray)
	else:
		try:
			dims = trace_or_stream_or_nparray.shape
			if len(dims) == 3:
				newstream = trace_or_stream_or_nparray
			elif len(dims) == 2  :
				newstream = np.zeros(( 1, dims[0], dims[1] )) 
				newstream[0] = trace_or_stream_or_nparray
		except:
			raise Exception('I/O dimensions: only obspy.Stream or obspy.Trace input supported.')

	return newstream, trace_or_stream_or_nparray 


def stream_indexes(data, delta=None, id=None, network=None, station=None, location=None, channel=None, starttime=None, endtime=None, npts=None, maxendtime=None, minstarttime=None, reftime=None):
	"""
	Return the indexes of Stream object with these traces that match the given
	stats criteria (e.g. all traces with ``channel="EHZ"``).
	______________________________________________________________________

	:type: 
	:param: 

	:rtype: 
	:return: 

	.. note::

		works exactely as `~obspy.core.stream.select`.

	"""

	(tmax,nmax) = streamdatadim(data)
	trace_indexes = np.asarray([])
	data_indexes = np.asarray([]) 
	for station_j in range(tmax):
		if delta is not None:
			if delta != (data[station_j]).stats.delta :
				continue
		if starttime is not None:
			if starttime != (data[station_j]).stats.starttime :
				continue
		if endtime is not None:
			if endtime != (data[station_j]).stats.endtime :
				continue
		if npts is not None:
			if npts != (data[station_j]).stats.npts :
				continue
		if id is not None:
			if not fnmatch.fnmatch((data[station_j]).stats.id, id) :
				continue
		if network is not None:
			if not fnmatch.fnmatch((data[station_j]).stats.network, network) :
				continue
		if station is not None:
			if not fnmatch.fnmatch((data[station_j]).stats.station, station) :
				continue
		if location is not None:
			if not fnmatch.fnmatch((data[station_j]).stats.location, location) :
				continue
		if channel is not None:
			if not fnmatch.fnmatch((data[station_j]).stats.channel, channel) :
				continue
		
		if maxendtime is not None:
			if maxendtime < (data[station_j]).stats.starttime :
				continue
		if minstarttime is not None:
			if minstarttime > (data[station_j]).stats.endtime :
				continue
		
		if reftime is not None:
			data_indexes = np.append(data_indexes, ( reftime - (data[station_j]).stats.starttime ) / (data[station_j]).stats.delta )

		trace_indexes = np.append(trace_indexes, station_j)

	return trace_indexes.astype(int), data_indexes.astype(int)


def recursive(a, scales=None, operation=None, maxscale=None):
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

	if maxscale is None:
		maxscale = nmax

	if scales is None:
		scales = [2**i for i in range(4,10) if ((2**i <= (maxscale)) and (2**i <= (nmax - 2**i)))]
		scales = np.require(scales, dtype=np.int) 

	# Initialize results at the minimal size
	timeseries = np.zeros(( tmax, len(scales), nmax )) 
	bandpass_timeseries = np.zeros(( tmax, len(scales), nmax )) 

	a.detrend('linear')
	a.taper(.05, type='triang', max_length=10) 
	# a.filter("highpass", freq=1.)  
	# a.taper(.05, type='triang', max_length=10) 
	# a.detrend('linear')
	# a.filter("highpass", freq=1.)  
	# a.taper(.05, type='triang', max_length=10) 

	for t, tr in enumerate(a) : # the channel-wise calculations      

		# Avoid clock channels 
		if not tr.stats.channel == 'YH':

			for n, s in reversed(list(enumerate(scales))) : # Avoid scales when too wide for the current channel
				#
				npts = tr.stats.npts
				dt = np.zeros((tr.data).shape)
				dt[1:] = np.abs(tr.data[:-1]-tr.data[1:])
				data = tr.data.copy()

				if len(scales)>1 and n < len(scales)-1 :
					tr_filt = tr.copy()
					tr_filt.filter('highpass', freq=1/(scales[n+1]*tr_filt.stats.delta), corners=2, zerophase=True)
					# if len(scales)>2 and n > 0 :
					# 	tr_filt.filter('lowpass', freq=1/(scales[n-1]*tr_filt.stats.delta), corners=2, zerophase=True)
					data = tr_filt.data.copy()

				data *= dt < (np.median(dt)+2.*np.std(dt))
				data = scipy.signal.detrend(data)

				data[data==0] = np.nan
				
				if operation is 'rms':                  
					# The cumulative sum can be exploited to calculate a 
					# moving average (the cumsum function is quite efficient)
					csqr = np.nan_to_num(data**2).cumsum() #np.nancumsum( data ** 2 ) #
				elif (operation is 'averageabs') or  (operation is 'sumabs'):  
					# The cumulative sum can be exploited to calculate a 
					# moving average (the cumsum function is quite efficient)
					csqr = np.nan_to_num(np.abs(data)).cumsum() # np.nancumsum(np.abs( data )) 
				elif (operation is 'average') or  (operation is 'sum'):  
					# The cumulative sum can be exploited to calculate a 
					# moving average (the cumsum function is quite efficient)
					csqr = np.nan_to_num(data).cumsum() #np.nancumsum( data )  
				# Convert to float
				csqr = np.require(csqr, dtype=np.float)

				if (s < (npts - s)) :    
					# Compute the sliding window
					if (operation is 'rms') or (operation is 'average') or (operation is 'averageabs') or (operation is 'sumabs') or (operation is 'sum'):  
						timeseries[t][n][s:npts] = csqr[s:] - csqr[:npts-s]
						#timeseries[t][n][:npts-s] = csqr[s:] - csqr[:npts-s]
						# for average and rms only 
						if operation is not 'sum' and operation is not 'sumabs':
							timeseries[t][n][:] /= s     

						# Pad with modified scale definitions(vectorization ###################################### TODO)
						timeseries[t][n][1:s] = csqr[1:s] - csqr[0]
						# for average and rms only
						if operation is not 'sum' and operation is not 'sumabs':
							timeseries[t][n][1:s] = timeseries[t][n][1:s]/np.asarray(range(1, s), dtype=np.float32)
						# detrending
						timeseries[t][n][1:s] = timeseries[t][n][1:s]+(timeseries[t][n][1:s]-timeseries[t][n][1])*np.asarray(range(s, 1, -1), dtype=np.float32)/s
						# filtering
						f = np.cumsum(timeseries[t][n][:])
						timeseries[t][n][0:s] = (f[s+1]-f[0:s])/np.asarray(range(s+1,1,-1), dtype=np.float32)
					# Avoid division by zero by setting zero values to tiny float
					dtiny = np.finfo(0.0).tiny
					idx = timeseries[t][n] < dtiny
					timeseries[t][n][idx] = dtiny 
					timeseries[t][n][12000:] = dtiny 
					if (operation is 'rms') :
						timeseries[t][n][:npts] = timeseries[t][n][:npts]**.5
	
	return timeseries, scales 

			
def correlationcoef(a, b, scales=None, maxscale=None):

	na = len(a)
	if maxscale is None:
		maxscale = na

	if scales is None:
		scales = [2**i for i in range(3,999) if ((2**i <= (maxscale)) and (2**i <= (na - 2**i)))]
	
	scales = np.require(scales, dtype=np.int) 
	scales = np.asarray(scales)
	nscale = len(scales)
	if nscale == 0 : 
		scales = [max([4, maxscale/10.])]
		nscale = 1
		#print '=>',max([4, maxscale/10.]), scales, nscale
	#print scales, nscale

	cc = np.ones(a.shape)*1.0
	prod_cumsum = np.cumsum( a * b )
	a_squarecumsum = np.cumsum( a**2 )
	b_squarecumsum = np.cumsum( b**2 )

	dtiny = np.finfo(0.0).tiny
	b_squarecumsum[b_squarecumsum < dtiny] = dtiny
	a_squarecumsum[a_squarecumsum < dtiny] = dtiny

	for s in scales :

		scaled_prod_cumsum = prod_cumsum[s:] - prod_cumsum[:-s]
		scaled_a_squarecumsum = a_squarecumsum[s:] - a_squarecumsum[:-s]
		scaled_b_squarecumsum = b_squarecumsum[s:] - b_squarecumsum[:-s]
		
		# scaled_prod_cumsum[scaled_prod_cumsum == 0 ] = 1
		# scaled_a_squarecumsum[scaled_a_squarecumsum == 0 ] = 1
		# scaled_b_squarecumsum[scaled_b_squarecumsum == 0 ] = 1

		cc[s:] *= (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

		# pading with modified def
		scaled_prod_cumsum = prod_cumsum[1:s] - prod_cumsum[0]
		scaled_a_squarecumsum = a_squarecumsum[1:s] - a_squarecumsum[0]
		scaled_b_squarecumsum = b_squarecumsum[1:s] - b_squarecumsum[0]
		
		# scaled_prod_cumsum[scaled_prod_cumsum == 0 ] = 1
		# scaled_a_squarecumsum[scaled_a_squarecumsum == 0 ] = 1
		# scaled_b_squarecumsum[scaled_b_squarecumsum == 0 ] = 1

		cc[1:s] *= 1# (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

	return cc#**(1./nscale)


def stream_trim_cf(stream, cf):

	wavelets = stream.copy()

	(tmax,nmax) = streamdatadim(stream)
	cflets = np.ones(( tmax, 2+(stream[0]).stats.sampling_rate*2 ))  

	for t, trace in enumerate(wavelets):

		cf[t][:trace.stats.sampling_rate]=0
		pick = np.argmax(cf[t])

		l = 0.25 # dominant period

		wlstart = trace.stats.starttime + pick * trace.stats.delta - 0
		wlend = trace.stats.starttime + pick * trace.stats.delta + l
		trace.trim(wlstart, wlend)
		
		cflets[t][:l*trace.stats.sampling_rate+2] = cf[t][pick -1 : pick + l * trace.stats.sampling_rate+1]

		#plotTfr(trace.data, dt=trace.stats.delta, fmin=0.1, fmax=trace.stats.sampling_rate/2)



	# nfft=(wavelets[0]).stats.sampling_rate
	# spec = np.zeros((tmax, nfft // 2 + 1), dtype=np.complex)

	# fig = plt.figure()
	# ax = fig.gca() 
	# f_lin = np.linspace(0, 0.5 / (wavelets[0]).stats.delta, nfft // 2 + 1)

	# for t, trace in enumerate(wavelets): 
	# 	spec[t] = np.abs( np.fft.rfft(trace.data, n=int(nfft)) * trace.stats.delta ) ** 2
	# 	ax.semilogx(f_lin, spec[t])

	return wavelets, cflets

def stream_processor_plot(stream, cf, cfcolor = 'm', ax = None, label = None, shift = 0, f=None, rescale=None):
	
	if ax is None : 
		ax = (plt.figure( figsize=(8, 10) )).gca()
		(ax.get_figure()).tight_layout()

	(tmax,nmax) = streamdatadim(stream)
	labels = ["" for x in range(shift+tmax)]
	anots = ["" for x in range(tmax)]

	if shift>0:
		label = None
		for i,item in enumerate(ax.get_yticklabels()):
			labels[i] = str(item.get_text())

	for t, trace in enumerate(stream):
		df = trace.stats.sampling_rate
		npts = trace.stats.npts
		time = np.arange(npts, dtype=np.float32) / df
		channel=trace.stats.channel #copy.deepcopy()
		if trace.stats.channel[-1] in ('Z','l', 'L', '1'):
			if trace.stats.channel in ('vertical', 'VERTICAL', '1'):
				channel='Z'
			color = '0.5'
		elif trace.stats.channel[-1] in ('E','t', 'T', '2'):
			if trace.stats.channel in ('east', 'EAST', '2'):
				channel='E'
			color = '0.7'
		elif trace.stats.channel[-1] in ('N','h', 'H', '3'):
			if trace.stats.channel in ('north', 'NORTH', '3'):
				channel ='N'
			color = '0.8'
		#print labels[shift+t]
		labels[shift+t] = trace.stats.station[0:3] +'.'+ channel
		anots[t] =  ' %3.1e' % (np.nanmax(np.abs(cf)) - np.nanmin(np.abs(cf)) )
		#ax.text(0, t, anots[t] , verticalalignment='bottom', horizontalalignment='left', color='green')
		if f is None : 
			d = -1*abs(trace.data.copy())
			ax.plot(time, shift+t-0.25+trace.data/(2*np.max(np.abs(trace.data))), color, zorder=1)
		if rescale is None :
			ax.plot(time, shift+t+0.+ ((cf[t][:npts] - np.nanmin(np.abs(cf[t][:npts])) )/(2*(np.nanmax(np.abs(cf[t][:npts])) - np.nanmin(np.abs(cf[t][:npts])))))**1., cfcolor, label=label, zorder=2)         
		else : 
			ax.plot(time, shift+t+0.+cf[t][:npts], cfcolor, label=label, zorder=2)         
		
		label = None

	ax.set_yticks(np.arange(0, shift+tmax, 1.0))
	ax.set_yticklabels(labels)
	#ax.text(0, shift-.25, anots[0] , verticalalignment='bottom', horizontalalignment='left', color=cfcolor)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Channel')
	ax.axis('tight')
	# plt.ylim( 10.5, 13.5) 
	ax.set_xlim( 0, min([120, max(time)])) 

	return ax, shift+t+1


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
		ax.plot(time, t+trace.data/(2*np.max(np.abs(trace.data))), '0.5')

		for c, channel in enumerate(cf[0][t]):
			if np.sum(cf[1][t][c][0:npts]) != 0 :
				if len(cf) > 0 :
					ax.plot(time, t-.5+cf[0][t][c][0:npts]/(np.nanmax(np.abs(cf[0][t][c][0:npts]))), 'r')        
				if len(cf) > 1 :
					ax.plot(time, t-.5+cf[1][t][c][0:npts]/(np.nanmax(np.abs(cf[1][t][c][0:npts]))), 'b')        
				if len(cf) > 2 :
					ax.plot(time, t-.5+cf[2][t][c][0:npts]/(np.nanmax(np.abs(cf[1][t][c][0:npts]))), 'g')        

	plt.yticks(np.arange(0, tmax, 1.0))
	ax.set_yticklabels(labels)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Channel')
	plt.axis('tight')
	plt.tight_layout()

	return ax

def stream_preprocessor_plot(stream,cf):
	
	fig = plt.figure()#figsize=plt.figaspect(1.2))
	ax = fig.gca() 
	(tmax,nmax) = streamdatadim(stream)
	labels = ["" for x in range(tmax)]
	for t, trace in enumerate(stream):
		df = trace.stats.sampling_rate
		npts = trace.stats.npts
		time = np.arange(npts, dtype=np.float32) / df
		labels[t] = trace.id
		ax.plot(time, t+trace.data/(2*np.max(np.abs(trace.data))), '0.5')

		for c, channel in enumerate(cf[t]):
			if np.sum(cf[t][c][0:npts]) != 0 :
					ax.plot(time, t-.5+cf[t][c][0:npts]/(np.nanmax(np.abs(cf[t][c][0:npts]))), 'g')        

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

				for channel_i in range(1,len(self.pre_processed_data)) :

					buf = self.pre_processed_data[0][station_i][enhancement_i] 
					buf /= self.pre_processed_data[channel_i][station_i][enhancement_i]
					# no ~zeros
					buf[buf < dtiny] = dtiny
					#buf[0:100] = np.nan
					# product enhancement (no nans, no infs)
					cf[station_i][np.isfinite(buf)] *= buf[np.isfinite(buf)]

			# rescaling 
			#cf[station_i] **= (1./self.enhancement_factor) 

		# ###################################################################################### why is this not ok ??????
		# ratio = self.pre_processed_data[0] / self.pre_processed_data[1]
		
		# # no ~zeros
		# ratio[ ratio < dtiny ] = dtiny
		# # no nans, no infs
		# ratio[ ~np.isfinite(ratio) ] = 1.

		# # returns product enhanced and rescaled
		# cf = (np.prod(ratio, axis=1))**(1/self.enhancement_factor)

		return cf

	def plot(self, **kwargs):
		return stream_processor_plot( self.data, self.output(), **kwargs)



class Correlate(object):
	def __init__(self, pre_processed_data, data=None, scales=None):

		self.data = data
		self.pre_processed_data = pre_processed_data[0]
		self.enhancement_factor = pre_processed_data[1]
		self.l_windows = pre_processed_data[2]
		self.scales = scales

	def output(self):

		dtiny = np.finfo(0.0).tiny
		(tmax,nmax) = streamdatadim(self.data)
		cf = np.ones(( tmax, nmax ))  

		for station_i, station_data in enumerate(self.pre_processed_data[0]):
			for enhancement_i, enhancement_data in enumerate(self.pre_processed_data[0][station_i]):

				for channel_i in range(1,len(self.pre_processed_data)) :

					if np.nansum(np.abs( self.pre_processed_data[channel_i][station_i][enhancement_i] )) > 0:

						a = self.pre_processed_data[0][station_i][enhancement_i] #highpass(self.pre_processed_data[0][station_i][enhancement_i], 1/(self.l_windows[0][enhancement_i]*(self.data[station_i]).stats.delta)/2, (self.data[station_i]).stats.sampling_rate)
						b = self.pre_processed_data[channel_i][station_i][enhancement_i] #highpass(self.pre_processed_data[channel_i][station_i][enhancement_i], 1/(self.l_windows[0][enhancement_i]*(self.data[station_i]).stats.delta)/2, (self.data[station_i]).stats.sampling_rate)
						
						buf = correlationcoef( a = a, b = b, \
							maxscale = int(self.l_windows[0][enhancement_i]/2.), scales=self.scales)

						#print 'RMS scale = ', self.l_windows[0][enhancement_i], ', scales=', self.scales

						# no ~zeros
						buf[buf < dtiny] = dtiny

						# product enhancement (no nans, no infs)
						cf[station_i][np.isfinite(buf)] *= buf[np.isfinite(buf)] 
						#cf[station_i][np.isfinite(buf)] += (1-buf[np.isfinite(buf)])**2

						# if no signal
						cf[station_i][ np.isnan(self.pre_processed_data[0][station_i][enhancement_i]) ] = np.nan
						cf[station_i][ np.isnan(self.pre_processed_data[channel_i][station_i][enhancement_i]) ] = np.nan


		return 1-(cf)#**(1./self.enhancement_factor))
		#return cf**(.5)

	def plot(self, **kwargs):        
		return stream_processor_plot( self.data, self.output(), **kwargs)


class ShortLongTerms(object):
	# Multiplex the data after pre-process

	def __init__(self, data, scales=None, statistic='averageabs', maxscale=None, **kwargs): 

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

		# pre_processed: array of pre-processed data
		# statistic: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, scales, statistic, maxscale) 

		# stores input parameters
		self.statistic = statistic

		# processors as class attributs
		self.ratio = Ratio(self.output(), self.data)
		self.correlate = Correlate(self.output(), self.data)

	def output(self):
		# Multiplex the pre-processed data
	
		# Initialize results at the minimal size
		(tmax,nmax) = streamdatadim(self.data)
		nscale = len(self.scales)
		channels = np.ones(( 2, tmax, nscale**2, nmax ))  
		l_windows = np.zeros(( 2, nscale**2 ))  
		dtiny = np.finfo(0.0).tiny
		
		# along stations
		for station_i, station_data in enumerate(self.pre_processed):
			n_enhancements = -1
			# along scales
			for smallscale_i, smallscale_data in enumerate(station_data):
				for bigscale_i, bigscale_data in enumerate(station_data):

					# en fait on peut dire len(STA)*9 = len(LTA) d'apres Withers, M., Aster, R., Young, C., Beiriger, J., Harris, M., Moore, S., & Trujillo, J. (1998). A comparison of select trigger algorithms for automated global seismic phase and event detection. Bulletin of the Seismological Society of America, 88(1), 95–106.
					if self.scales[bigscale_i] <= self.scales[smallscale_i]*10 and self.scales[bigscale_i] >= self.scales[smallscale_i]*7:

						n_enhancements += 1
						l_windows[0][n_enhancements] = self.scales[smallscale_i]
						l_windows[1][n_enhancements] = self.scales[bigscale_i]

						# no divide by ~zeros
						bigscale_data[bigscale_data < dtiny] = dtiny

						channels[0][station_i][n_enhancements] = smallscale_data
						channels[1][station_i][n_enhancements] = bigscale_data

		if n_enhancements == -1:
			print "scales must around 1 orders apart (from *7 to *10)"

		for i in range(n_enhancements+1, nscale**2):
			channels = np.delete(channels, n_enhancements+1, axis=2)
			l_windows = np.delete(l_windows, n_enhancements+1, axis=1)

		return channels, n_enhancements+1, l_windows

	def plot(self):
		channels, n, l = self.output()
		return stream_multiplexor_plot( self.data, channels )


class leftRightTerms(object):
	# Multiplex the data after pre-process

	def __init__(self, data, scales=None, statistic='averageabs', maxscale=None): 

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

		# pre_processed: array of pre-processed data
		# statistic: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, scales, statistic, maxscale) 

		# stores input parameters
		self.statistic = statistic

		# processors as class attributs
		self.ratio = Ratio(self.output(), self.data)
		self.correlate = Correlate(self.output(), self.data)

	def output(self):
		# Multiplex the pre-processed data
	
		# Initialize results at the minimal size
		(tmax,nmax) = streamdatadim(self.data)
		nscale = len(self.scales)
		channels = np.zeros(( 2, tmax, nscale**2, nmax ))  ################################################ todo gen as nan 
		l_windows = np.zeros(( 2, nscale**2 ))  
		dtiny = np.finfo(0.0).tiny
		
		# along stations
		for station_i, station_data in enumerate(self.pre_processed):
			n_enhancements = -1

			npts = (self.data[station_i]).stats.npts

			# along scales
			for scale_i, scale_data in enumerate(station_data):

					n_enhancements += 1
					l_windows[0][n_enhancements] = self.scales[scale_i]
					l_windows[1][n_enhancements] = self.scales[scale_i]

					# no divide by ~zeros
					scale_data[scale_data < dtiny] = dtiny

					channels[1][station_i][n_enhancements] = scale_data
					channels[0][station_i][n_enhancements][:-1*(self.scales[scale_i])] = scale_data[self.scales[scale_i]:]

					channels[0][station_i][n_enhancements][-1*(self.scales[scale_i]):] = np.nan
					channels[1][station_i][n_enhancements][npts:] = np.nan

					# apod = (np.require( range(self.scales[scale_i]) , dtype=np.float) / np.require(self.scales[scale_i], dtype=np.float))**0.5
					# channels[0][station_i][scale_i][:self.scales[scale_i]] *= apod
					# channels[1][station_i][scale_i][:self.scales[scale_i]] *= apod

					# channels[0][station_i][n_enhancements][ (npts-2*self.scales[scale_i]):(npts-self.scales[scale_i]) ] *= apod[::-1]
					# channels[1][station_i][n_enhancements][ (npts-self.scales[scale_i]):npts ] *= apod[::-1]


		for i in range(n_enhancements+1, nscale**2):
			channels = np.delete(channels, n_enhancements+1, axis=2)
			l_windows = np.delete(l_windows, n_enhancements+1, axis=1)

		return channels, n_enhancements+1, l_windows

	def plot(self):
		channels, n, l = self.output() 
		return stream_multiplexor_plot( self.data, channels )


class Component(object):
	# Multiplex the data after pre-process

	def __init__(self, data, scales=None, statistic='rms', maxscale=None): 

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

		# pre_processed: array of pre-processed data
		# statistic: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, scales, statistic, maxscale) 

		# stores input parameters
		self.statistic = statistic

		# processors as class attributs
		self.ratio = Ratio(self.output(), self.data)
		self.correlate = Correlate(self.output(), self.data)

	def output(self):
		# Multiplex the pre-processed data
	
		# Initialize results at the minimal size
		(tmax,nmax) = streamdatadim(self.data)
		nscale = len(self.scales)
		channels = np.zeros(( 3, tmax, nscale**2, nmax )) 
		l_windows = np.zeros(( 3, nscale**2 ))  
		dtiny = np.finfo(0.0).tiny
		
		# along stations
		for station_i in range(tmax):
			delta = (self.data[station_i]).stats.delta
			net = (self.data[station_i]).stats.network
			sta = (self.data[station_i]).stats.station
			loc = (self.data[station_i]).stats.location
			chan = (self.data[station_i]).stats.channel
			stime = (self.data[station_i]).stats.starttime
			etime = (self.data[station_i]).stats.endtime
			npts = (self.data[station_i]).stats.npts
			#print station_i, ':',  delta, net, sta, loc, chan, stime, etime, npts

			ZNE = [chan, chan[:-1] + 'N', chan[:-1] + 'E']
			ZNE_sta = [sta, sta, sta]

			if chan[-1] in ('Z', 'N', 'E') :
				ZNE[0] = chan[:-1] + 'Z'
				ZNE[1] = chan[:-1] + 'N'
				ZNE[2] = chan[:-1] + 'E'
			if chan[-1] in ('3', '2', '1') :
				ZNE[0] = chan[:-1] + '3'
				ZNE[1] = chan[:-1] + '2'
				ZNE[2] = chan[:-1] + '1'
			if chan in ('VERTICAL', 'NORTH', 'EAST') :
				ZNE[0] = 'VERTICAL'
				ZNE[1] = 'NORTH'
				ZNE[2] = 'EAST'
			if chan in ('vertical', 'north', 'east') :
				ZNE[0] = 'vertical'
				ZNE[1] = 'north'
				ZNE[2] = 'east'
			if len(sta) > 3 :
				if sta[-1] in ('Z', 'N', 'E') :
					ZNE_sta[0] = sta[:-1] + 'Z'
					ZNE_sta[1] = sta[:-1] + 'N'
					ZNE_sta[2] = sta[:-1] + 'E'
				if sta[-1] in ('z', 'n', 'e') :
					ZNE_sta[0] = sta[:-1] + 'z'
					ZNE_sta[1] = sta[:-1] + 'n'
					ZNE_sta[2] = sta[:-1] + 'e'

			if chan[-1] in ('N', '2') or chan in ('NORTH', 'north'):
				ZNE     = [    ZNE[1],     ZNE[0]]#,     ZNE[2]]
				ZNE_sta = [ZNE_sta[1], ZNE_sta[0]]#, ZNE_sta[2]]
			elif chan[-1] in ('E', '1') or chan in ('EAST', 'east'):
				ZNE     = [    ZNE[2],     ZNE[0]]#,     ZNE[1]]
				ZNE_sta = [ZNE_sta[2], ZNE_sta[0]]#, ZNE_sta[1]]

			ZNE_i = np.asarray([])
			ZNE_di = np.asarray([])
			for i in range(len(ZNE)):
				#print ZNE_sta[i], ZNE[i]
				i, di = stream_indexes(self.data, delta=delta, network=net, station=ZNE_sta[i], location=loc, channel=ZNE[i], minstarttime=stime, maxendtime=etime, reftime=stime )
				ZNE_i = np.append(ZNE_i, i )
				ZNE_di = np.append(ZNE_di, di )

			ZNE_i = ZNE_i.astype(int)
			ZNE_di = ZNE_di.astype(int)
			#print  ZNE_sta, ZNE, ZNE_i, ZNE_di

			for i in range(len(ZNE_i)):
				#print '#', ZNE_i[i], ':',  (self.data[ZNE_i[i]]).stats.delta, (self.data[ZNE_i[i]]).stats.network, (self.data[ZNE_i[i]]).stats.station, (self.data[ZNE_i[i]]).stats.location, (self.data[ZNE_i[i]]).stats.channel, (self.data[ZNE_i[i]]).stats.starttime, (self.data[ZNE_i[i]]).stats.endtime, (self.data[ZNE_i[i]]).stats.npts
				if ZNE_di[i] == 0 :
					s = [0, 0]
					e = [nmax, nmax]
				elif ZNE_di[i] > 0 :
					s = [0, ZNE_di[i]]
					e = [nmax-ZNE_di[i], nmax]
				elif ZNE_di[i] < 0 :
					s = [-1*ZNE_di[i], 0]
					e = [nmax, nmax+ZNE_di[i]]
						
				for scale_i in range(nscale):
					l_windows[i][scale_i] = self.scales[scale_i]
					channels[i][station_i][scale_i][s[0]:e[0]] = self.pre_processed[ZNE_i[i]][scale_i][s[1]:e[1]]
					# channels[i][station_i][scale_i][:s[0]]   = self.pre_processed[ZNE_i[i]][scale_i][s[1]+1]
					# channels[i][station_i][scale_i][e[0]:]   = self.pre_processed[ZNE_i[i]][scale_i][e[1]-1]

		return channels, nscale+1, l_windows

	def plot(self):
		channels, n, l = self.output() 
		return stream_multiplexor_plot( self.data, channels )

		






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

