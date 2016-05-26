# -*- coding: utf-8 -*-
"""
trigger - Module to improve obspy.signal.trigger

This module can be used fro probabilistic estimation of the parameters
of body-waves from continuous data.
_________
.. note::

	Functions and classes are ordered from general to specific.

	For more details on the work related to this module see 
	Massin & Malcolm, 2016: A better automatic body-wave picker with 
	broad applicability. SEG Technical Program.
    
"""


import re
import copy
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from pandas import rolling_kurt
from obspy import read, Trace, Stream
from obspy.core.trace import Stats
from obspy.signal.filter import highpass
from source import spherical_to_cartesian


def streamdatadim(a):
	"""
	Given a stream (obspy.core.stream) calculate the minimum  
	dimensions of the array that contents all data from all traces.
	______
	:type: ObsPy :class:`~obspy.core.stream`
	:param: datastream of e.g. seismogrammes.
	_______
	:rtype: array
	:return: array of int corresponding to the dimensions of stream.
	"""

	nmax=0
	for t, tr in enumerate(a):
		nmax = max((tr.stats.npts, nmax))

	return (t+1, nmax)


def trace2stream(trace_or_stream_or_nparray):
	"""
	Given a stream (obspy.core.stream) calculate the minimum  
	dimensions of the array that contents all data from all traces.
	______
	:type: ObsPy :class:`~obspy.core.stream` or
		`~obspy.core.trace` or NumPy :class:`~numpy.ndarray`.
	:param: input data .
	_______
	:rtype: 
		- ObsPy :class:`~obspy.core.stream`.
		- same as input
	:return:
		- stream containing copies of inputed data.
		- copy of inputed data.
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
	Return the indexes of Stream object with these traces that match the 
	given stats criteria (e.g. all traces with ``channel="EHZ"``) as well 
	as the corresponding indexes of data attributes if requested.
	______
	:type: 
		- identical to the attributes of 
			ObsPy:class:`~obspy.core.trace.Stats`.
		- maxendtime, minstarttime, reftime: 
			ObsPy:class:`~obspy.core.utcdatetime.UTCDateTime` 
			(optional).
	:param: 
		- identical to the attributes of 
			ObsPy:class:`~obspy.core.trace.Stats`
		- maxendtime: optional, latest date and time of the first 
			sample NumPy:ndarray:`~obspy.core.trace.data` given in UTC.
		- minstarttime: optional, earliest date and time of the first 
			sample in trace.data given in UTC.
		- reftime: optional, reference date and time to compare to the 
			first sample in NumPy:ndarray:`~obspy.core.trace.data`
			given in UTC.
	_______
	:rtype: 
		- NumPy:class:`~numpy.ndarray` type int
		- NumPy:class:`~numpy.ndarray` type int
	:return: 
		- indexes of selected traces in stream
		- time difference (in samples) of selected 
			NumPy:ndarray:`~obspy.core.trace.data` in stream: if 
			return[1][i]>0 selected stream[i].data starts  
			after reftime ; else, before.
	_________
	.. note::

		Works similarly to ObsPy:meth:`~obspy.core.stream.select`.

	"""

	# Gets stream dimensions
	(tmax,nmax) = streamdatadim(data)

	# Initiates
	trace_indexes = np.asarray([])
	data_indexes = np.asarray([]) 

	# Loops over traces 
	for station_j in range(tmax):

		# Testing inputs one by one
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
		
		# Testing optional inputs
		if maxendtime is not None:
			if maxendtime < (data[station_j]).stats.starttime :
				continue
		if minstarttime is not None:
			if minstarttime > (data[station_j]).stats.endtime :
				continue
		
		# If all tests passed
		## Selects trace
		trace_indexes = np.append(trace_indexes, station_j)

		## Calculates optional sample difference
		if reftime is not None:
			data_indexes = np.append(data_indexes, ( reftime - (data[station_j]).stats.starttime ) / (data[station_j]).stats.delta )

	return trace_indexes.astype(int), data_indexes.astype(int)


def recursive(a, scales=None, operation=None, maxscale=None):
	"""
	_
	Performs multi-scale calculation by 
	creating series of operations of different subsets of 
	the full data set. This is also called rolling 
	operation.
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`
		- scales: vector (optional).
		- operation: string (optional).
		- maxscale: int (optional).
	:param: 
		- data-stream of e.g. seismograms.
		- scales: scale(s) of time-series operation (in samples).
		- operation: type of operation:
			rms: root mean square over time scales.
			sum: sum over time scales.
			average: average  over time scales.
			sumabs: sum of absolute values over time scales.
			averageabs: average of absolute values over time scales.
		- maxscale: maximum allowed scale (in samples).
	_______
	:rtype: 
		- NumPy:class:`~numpy.ndarray` (channel, scales, samples).
		- NumPy:class:`~numpy.ndarray` vector.
	:return: 
		- multi-scale array of root mean square time series, 
		- calculation scales (samples scale unit).
	_________
	.. note::

		High-pass filtering over 1/(scale/fs) is performed before 
		calculation using ObsPy:class:`~obspy.core.stream.filter`.

	"""
	# 1) Iterate on channels
	# 2) Pre calculate the common part of all scales
	# 3) Iterate on scales 
	# 4) Perform filter, channel/scale calculation 

	a, input_a = trace2stream(a)
	
	# Initialize multiscale if undefined
	if operation is None:
		operation = 'rms'
	(tmax,nmax) = streamdatadim(a)

	if maxscale is None:
		maxscale = nmax

	if scales is None:
		scales = [2**i for i in range(5,10) if ((2**i <= (maxscale)) and (2**i <= (nmax - 2**i)))]
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
				
				npts = tr.stats.npts
				dt = np.zeros((tr.data).shape)
				dt[1:] = np.abs(tr.data[:-1]-tr.data[1:])
				data = tr.data.copy()

				if len(scales)>1 and n < len(scales)-1 :
					tr_filt = tr.copy()
					tr_filt.filter('highpass', freq=1/(scales[n+1]*tr_filt.stats.delta), corners=4, zerophase=True)
					data = tr_filt.data.copy()

				data *= dt < (np.median(dt)+2.*np.std(dt))
				data = detrend(data)

				if operation[0:1] in ('d-'):  
					data = np.gradient(data) 

				data[data==0] = np.nan
				
				# The cumulative sum can be exploited to calculate a 
				# moving average (the cumsum function is quite efficient)
				if operation in ('rms', 'sumsquare', 'd-sumsquare'):                  
					csqr = np.nan_to_num(data**2).cumsum() #np.nancumsum( data ** 2 ) #
				elif operation in ('averageabs', 'sumabs'):  
					csqr = np.nan_to_num(np.abs(data)).cumsum() # np.nancumsum(np.abs( data )) 
				elif operation in ('average', 'sum'):  
					csqr = np.nan_to_num(data).cumsum() #np.nancumsum( data )  
				
				# Convert to float
				csqr = np.require(csqr, dtype=np.float)

				if (s < (npts - s)) :    
					# Compute the sliding window
					timeseries[t][n][s:npts] = csqr[s:] - csqr[:npts-s]
					# for average and rms only 
					if operation not in ('sum', 'sumabs', 'sumsquare', 'd-sumsquare'):
						timeseries[t][n][:] /= s     
					# Pad with modified scale definitions(vectorization ###################################### TODO)
					timeseries[t][n][1:s] = csqr[1:s] - csqr[0]
					# for average and rms only
					if operation not in ('sum', 'sumabs', 'sumsquare', 'd-sumsquare'):
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
					# finish rms case
					if (operation is 'rms') :
						timeseries[t][n][:npts] = timeseries[t][n][:npts]**.5
	
	return timeseries, scales 


def correlationcoef(a, b, scales=None, maxscale=None):
	"""
	Calculate moving cross-correlation coefficients by 
	creating series of operations of different subsets of 
	the full data set.
	______
	:type: 
		- NumPy:class:`~numpy.ndarray` (samples).
		- NumPy:class:`~numpy.ndarray` (samples).
		- scales: vector (optional).
		- maxscale: int (optional).
	:param: 
		- data of e.g. seismograms.
		- data of e.g. seismograms.
		- scales: scale(s) of cross-correlation (in samples).
		- maxscale: maximum allowed scale (in samples).
	_______
	:rtype: NumPy:class:`~numpy.ndarray`
	:return: array of moving cross-correlation coefficients.
	_________
	.. note::

		The signal windows considered for calculation are truncated 
		for sample indexes below time scale.

	"""

	na = len(a)
	if maxscale is None:
		maxscale = na

	if scales is None:
		scales = [2**i for i in range(4,999) if ((2**i <= (maxscale)) and (2**i <= (na - 2**i)))]
	
	scales = np.require(scales, dtype=np.int) 
	scales = np.asarray(scales)
	nscale = len(scales)
	if nscale == 0 : 
		scales = [max([4, maxscale/10.])]
		nscale = 1

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

		cc[s:] *= (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

		# pading with modified def
		scaled_prod_cumsum = prod_cumsum[1:s] - prod_cumsum[0]
		scaled_a_squarecumsum = a_squarecumsum[1:s] - a_squarecumsum[0]
		scaled_b_squarecumsum = b_squarecumsum[1:s] - b_squarecumsum[0]

		cc[1:s] *= (scaled_prod_cumsum / np.sqrt( scaled_a_squarecumsum * scaled_b_squarecumsum ))

	return cc #**(1./nscale)


def stream_processor_plot(stream, cf, cfcolor = 'm', ax = None, label = None, shift = 0, f=None, rescale=None):
	"""
	Plots the body-wave characteristic functions resulting of 
	`~trigger.Ratio.output` and `~trigger.Correlate.output` classes.
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- NumPy:class:`~numpy.ndarray` [channel, samples].
		- cfcolor: string (optional).
		- ax: matplotlib:class:`~matplotlib.axes.Axes` (optional).
		- label: string (optional).
		- shift: int  (optional).
		- f: int (optional).
		- rescale: int (optional).
	:param: 
		- data of e.g. seismograms.
		- characteristic functions associated to data, as given by 
			`~trigger.Ratio.output` and `~trigger.Correlate.output`.
		- cfcolor: color of the over-layed characteristic functions.
		- ax: axis to use for the plot.
		- label: label to use in legend for the characteristic 
			functions.
		- shift: Y-value to start the bottom of the plot.
		- f: flag to deactivate data plotting (any value but None).
		- rescale: flag to deactivate the normalization of 
			characteristic functions (any value but None).
	_______
	:rtype: 
		- matplotlib:class:`~matplotlib.axes.Axes` 
		- int
	:return: 
		- axis to use for the plot.
		- Y-value of the top of the plot.
	___________
	.. rubric:: Example

		>>> import trigger
		>>> stlt = trigger.ShortLongTerms(trigger.artificial_stream(npts=5000))
		>>> ax,shift=trigger.stream_processor_plot(stlt.data, stlt.ratio.output(), cfcolor='r', label=r'$^M\bar{ST}/\bar{LT}$')
		>>> trigger.stream_processor_plot(stlt.data, stlt.correlate.output(), ax=ax, cfcolor='g', label=r'$^M\bar{ST}\star\bar{LT}$', f="nodata")
		>>> ax.legend()

	"""
	
	if ax is None : 
		ax = (plt.figure( figsize=(8, 5) )).gca()
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
		cf[t][:200]=0.0
		cf[t][-200:]=0.0 
		if f is None : 
			d = -1*abs(trace.data.copy())
			ax.plot(time, shift+t-0.25+trace.data/(2.*np.max(np.abs(trace.data))), color, zorder=1)
		if rescale is None :
			ax.plot(time, shift+t+0.+ ((cf[t][:npts] - np.nanmin(np.abs(cf[t][:npts])) )/(2.*(np.nanmax(np.abs(cf[t][:npts])) - np.nanmin(np.abs(cf[t][:npts])))))**1., cfcolor, label=label, zorder=2)         
		else : 
			ax.plot(time, shift+t+0.+cf[t][:npts], cfcolor, label=label, zorder=2)         
		
		if not label is None :
			ax.legend()

		label = None

	ax.set_yticks(np.arange(0, shift+tmax, 1.0))
	ax.set_yticklabels(labels)
	#ax.text(0, shift-.25, anots[0] , verticalalignment='bottom', horizontalalignment='left', color=cfcolor)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Channel')
	ax.axis('tight')
	# plt.ylim( 10.5, 13.5) 
	plt.tight_layout()
	ax.set_xlim( 0, min([120, max(time)])) 

	return ax, shift+t+1


def stream_multiplexor_plot(stream,cf):
	"""
	Plots the multi-scale time-series resulting from the multiplexors
	used in `~trigger` for characteristic function calculation 
	(from `~trigger.ShortLongTerms` or `~trigger.leftRightTerms` or 
	`~trigger.Component` classes).
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- NumPy:class:`~numpy.ndarray` [data-stream, channel, sample].
	:param: 
		- data of e.g. seismograms.
		- multi-scale time-series associated to data, as given by 
			`~trigger.ShortLongTerms.output` or 
			`~trigger.leftRightTerms.output` or 
			`~trigger.Component.output`.
	_______
	:rtype: 
		- matplotlib:class:`~matplotlib.axes.Axes` 
	:return: 
		- axis to use for the plot.
	___________
	.. rubric:: Example

		>>> import trigger
		>>> stlt = trigger.ShortLongTerms(trigger.artificial_stream(npts=500))
		>>> trigger.stream_multiplexor_plot(stlt.data, (stlt.output())[0])

	"""
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


def artificial_stream(npts=1000., noise=1., P=[10., 15., 3.**.5], S=[17., 15., 3.**.5]) : 
	"""
	Generate artificial data for testing purposes.
	______
	:type: 
		- npts: int (optional).
		- noise: float (optional).
		- P: list (1x3 type float [optional]).
		- S: list (1x3 type float [optional]).
	:param: 
		- npts: data length in samples.
		- noise: amplitude of noise.
		- P: P-wave properties [amplitude, frequency, H/Z ratio]
		- S: S-wave properties [amplitude, frequency, Z/H ratio]	
	_______
	:rtype: ObsPy :class:`~obspy.core.stream`
	:return: artificial noisy data with [noise only, frequency change, 
		amplitude change, polarization change].
	_________
	.. note::

		The noise is a random 3d motion projection on 3 cartesian 
		components.

		The amplitude decays are linear and the body-wave span a third
		of the signal duration (npts/6 samples each).
	___________
	.. rubric:: Example

		>>> import trigger
		>>> a = trigger.artificial_stream(npts=1000)
		>>> print a
	_________
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


class ShortLongTerms(object):

	"""
	Produces an estimation/proxy of the probability of seismic 
	body-wave arrival in continuous seismic records.

	In practice, it sets an instance of ShortLongTerms that can be 
	used to form a matrix in which the data-stream of cell [0, c, n] 
	has to be compared with cells [1, c, n]. For the data of a given 
	channel (c) there are n pairs of time-series to compare. 
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- scale: list (multi-scaling by default, optional).
		- statistic: string (default 'averageabs', optional).
		- maxscale: int (default 'None', optional).
	:param: 
		- data of e.g. seismograms.
		- scales: or length of the window used for pre-processing with
			`~trigger.recursive`.
		- statistic: operation parameter in pre-processing with
			`~trigger.recursive`.
		- maxscale: maximum scale the window used for pre-processing 
			with `~trigger.recursive`.
	___________
	.. rubric:: _`Default Attributes`
		- `~trigger.ShortLongTerms.output`: returns the results.
		- `~trigger.ShortLongTerms.plot`: displays the output with 
			`~trigger.stream_multiplexor_plot`.
		- `~trigger.ShortLongTerms.correlate`: uses the 
			`~trigger.Correlate` class to compare data-streams.
		- `~trigger.ShortLongTerms.ratio`: uses the 
			`~trigger.Ratio` class to compare data-streams.
	___________
	.. rubric:: Example

		Plot the data-streams:
		>>> import trigger
		>>> cf = trigger.ShortLongTerms(trigger.artificial_stream(npts=5000), statistic='rms')
		>>> cf.plot()
		
		Plot the characteristic function:
		>>> cf.correlate.plot()

	"""

	def __init__(self, data, preprocessor='averageabs', scales=None, maxscale=None): 

		# stores input parameters
		self.preprocessor = preprocessor
		self.scales = scales
		self.maxscale = maxscale

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

	def output(self):
		# Multiplex the pre-processed data	

		# pre_processed: array of pre-processed data
		# preprocessor: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, self.scales, self.preprocessor, self.maxscale) 

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


class LeftRightTerms(object):

	"""
	Produces an estimation/proxy of the probability of seismic 
	body-wave arrival in continuous seismic records. 

	In practice, it sets an instance of leftRightTerms that can be 
	used to form a matrix in which the data-stream in cell [0, c, n] 
	has to be compared with cells [1, c, n]. For the data of a given 
	channel (c) there are n pairs of time-series to compare.
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- scale: list (multi-scaling by default, optional).
		- statistic: string (default 'averageabs', optional).
		- maxscale: int (default 'None', optional).
	:param: 
		- data of e.g. seismograms.
		- scales: or length of the window used for pre-processing with
			`~trigger.recursive`.
		- statistic: operation parameter in pre-processing with
			`~trigger.recursive`.
		- maxscale: maximum scale the window used for pre-processing 
			with `~trigger.recursive`.
	___________
	.. rubric:: _`Default Attributes`
		- `~trigger.LeftRightTerms.output`: returns the results.
		- `~trigger.LeftRightTerms.plot`: displays the output with 
			`~trigger.stream_multiplexor_plot`.
		- `~trigger.LeftRightTerms.correlate`: uses the 
			`~trigger.Correlate` class to compare data-streams.
		- `~trigger.LeftRightTerms.ratio`: uses the 
			`~trigger.Ratio` class to compare data-streams.
	___________
	.. rubric:: Example

		Plot the data-streams:
		>>> import trigger
		>>> cf = trigger.LeftRightTerms(trigger.artificial_stream(npts=5000), statistic='rms')
		>>> cf.plot()
		
		Plot the characteristic function:
		>>> cf.correlate.plot()

	"""

	def __init__(self, data, preprocessor='averageabs', scales=None, maxscale=None): 

		# stores input parameters
		self.preprocessor = preprocessor
		self.scales = scales
		self.maxscale = maxscale

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

	def output(self):
		# Multiplex the pre-processed data	

		# pre_processed: array of pre-processed data
		# preprocessor: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, self.scales, self.preprocessor, self.maxscale) 

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


class Components(object):

	"""
	Produces an estimation/proxy of the probability of seismic 
	body-wave arrival in continuous seismic records. 

	In practice, it sets an instance of leftRightTerms that can be 
	used to form a matrix in which the data-stream in cell [0, c, n] 
	has to be compared with cells [1, c, n], [2, c, n], etc. For the data
	of a given channel (c), with Nc component of the same channel, 
	there are Nc*n pairs of time-series to compare. 
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- scale: list (multi-scaling by default, optional).
		- statistic: string (default 'averageabs', optional).
		- maxscale: int (default 'None', optional).
	:param: 
		- data of e.g. seismograms.
		- scales: or length of the window used for pre-processing with
			`~trigger.recursive`.
		- statistic: operation parameter in pre-processing with
			`~trigger.recursive`.
		- maxscale: maximum scale the window used for pre-processing 
			with `~trigger.recursive`.
	___________
	.. rubric:: _`Default Attributes`
	___________
	.. rubric:: _`Default Methods`
		- `~trigger.Component.output`: returns the results.
		- `~trigger.Component.plot`: displays the output with 
			`~trigger.stream_multiplexor_plot`.
		- `~trigger.Component.correlate`: uses the 
			`~trigger.Correlate` class to compare data-streams.
		- `~trigger.Component.ratio`: uses the 
			`~trigger.Ratio` class to compare data-streams.
	___________
	.. rubric:: Example

		Plot the data-streams:
		>>> import trigger
		>>> cf = trigger.Component(trigger.artificial_stream(npts=5000), statistic='rms')
		>>> cf.plot()

		Plot the characteristic function:
		>>> cf.correlate.plot()

	"""

	def __init__(self, data, preprocessor='rms', scales=None, maxscale=None): 

		# stores input parameters
		self.preprocessor = preprocessor
		self.scales = scales
		self.maxscale = maxscale

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

	def output(self):
		# Multiplex the pre-processed data	

		# pre_processed: array of pre-processed data
		# preprocessor: sum|average|rms
		# scales: list
		self.pre_processed, self.scales = recursive(self.data, self.scales, self.preprocessor, self.maxscale) 

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


class Ratio(object):
	"""
	Produces a not-to-scale proxy of the probability of seismic 
	body-wave arrival in continuous seismic records. 
	
	In practice, it sets an instance of Ratio that can be used to 
	apply division between the first and all other time-series of the 
	same channel given in a multi-scale data-stream.
	______
	:type: 
		- NumPy:class:`~numpy.ndarray` (channel, scales, samples).
		- data: ObsPy:class:`~obspy.core.stream` (optional).
	:param: 
		- multi-scale data-stream, pre-processed with 
			`~trigger.ShortLongTerms` or `~trigger.leftRightTerms` or
			`~trigger.Component`
		- data of e.g. seismograms. 
	___________
	.. rubric:: _`Default Attributes`
		- `~trigger.Ratio.output`: returns the results.
		- `~trigger.Ratio.plot`: displays the output with 
			`~trigger.stream_processor_plot`.
	___________
	.. rubric:: Example

		Get some data-streams:
		>>> import trigger
		>>> data = trigger.artificial_stream(npts=5000)
		>>> data_streams = (trigger.ShortLongTerms(data)).output()
		
		Get the operator instance and plot:
		>>> cf = trigger.Ratio(data_streams, data)
		>>> cf.plot()

	"""
	def __init__(self, data, multiplexor = 'shortlongterms', preprocessor = 'averageabs', **kwargs): #, pre_processed_data, data=None):

		self.data = data
		self.multiplexor = multiplexor
		self.preprocessor = preprocessor
		
		self.shortlongterms = ShortLongTerms(data, preprocessor=self.preprocessor, **kwargs)
		self.leftrightterms = LeftRightTerms(data, preprocessor=self.preprocessor, **kwargs)
		self.components = Components(data, **kwargs)

	def output(self):

		if self.multiplexor in ('shortlongterms', 'stlt'):
			pre_processed_data = self.shortlongterms.output()
		elif self.multiplexor in ('leftrightterms', 'ltrt'):
			pre_processed_data = self.leftrightterms.output()
		elif self.multiplexor in ('components', 'comp'):
			pre_processed_data = self.components.output()
		
		self.pre_processed_data = pre_processed_data[0]
		self.enhancement_factor = pre_processed_data[1]
		self.l_windows = pre_processed_data[2]

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

		# # no ~zeros
		# ratio[ ratio < dtiny ] = dtiny
		# # no nans, no infs
		# ratio[ ~np.isfinite(ratio) ] = 1.

		# returns product # enhanced and rescaled		
		return cf         # cf = (np.prod(ratio, axis=1))**(1/self.enhancement_factor)

	def plot(self, **kwargs):
		return stream_processor_plot( self.data, self.output(), **kwargs)


class Correlate(object):
	"""
	Produces an estimation of the normalized probability of seismic 
	body-wave arrival in continuous seismic records. 

	In practice, it sets an instance of Correlate that can be used to
	apply correlation between the first and all other time-series of
	the same channel given in a multi-scale data-streams.
	______
	:type: 
		- NumPy:class:`~numpy.ndarray` (channel, scales, samples).
		- data: ObsPy:class:`~obspy.core.stream` (optional).
		- scale: list (multi-scaling by default, optional).
	:param: 
		- multi-scale data-stream, pre-processed with 
			`~trigger.ShortLongTerms` or `~trigger.leftRightTerms` or
			`~trigger.Component`.
		- data: of e.g. seismograms.
		- scales: or length of the window used for correlation
	___________
	.. rubric:: _`Default Attributes`
		- `~trigger.Correlate.output`: returns the results.
		- `~trigger.Correlate.plot`: displays the output with 
			`~trigger.stream_processor_plot`.
	___________
	.. rubric:: Example

		Get some data-streams:
		>>> import trigger
		>>> data = trigger.artificial_stream(npts=5000)
		>>> data_streams = (trigger.ShortLongTerms(data)).output()
		
		Get the operator instance and plot:
		>>> cf = trigger.Correlate(data_streams, data, scales=[10])
		>>> cf.plot()

		Idem, activating multi-scaling:
		>>> mcf = trigger.Correlate(data_streams, data)
		>>> mcf.plot()

	"""
	def __init__(self, data, multiplexor = 'components', preprocessor = 'rms', procscales=None, **kwargs): #, pre_processed_data, data=None):

		self.data = data
		self.multiplexor = multiplexor
		self.preprocessor = preprocessor
		self.procscales = procscales
		
		self.shortlongterms = ShortLongTerms(data, preprocessor=self.preprocessor, **kwargs)
		self.leftrightterms = LeftRightTerms(data, preprocessor=self.preprocessor, **kwargs)
		self.components = Components(data, preprocessor=self.preprocessor, **kwargs)

	def output(self):

		if self.multiplexor in ('shortlongterms', 'stlt'):
			pre_processed_data = self.shortlongterms.output()
		elif self.multiplexor in ('leftrightterms', 'ltrt'):
			pre_processed_data = self.leftrightterms.output()
		elif self.multiplexor in ('components', 'comp'):
			pre_processed_data = self.components.output()
		
		self.pre_processed_data = pre_processed_data[0]
		self.enhancement_factor = pre_processed_data[1]
		self.l_windows = pre_processed_data[2]

		dtiny = np.finfo(0.0).tiny
		(tmax,nmax) = streamdatadim(self.data)
		cf = np.ones(( tmax, nmax ))  

		for station_i, station_data in enumerate(self.pre_processed_data[0]):
			for enhancement_i, enhancement_data in enumerate(self.pre_processed_data[0][station_i]):

				for channel_i in range(1,len(self.pre_processed_data)) :

					if np.nansum(np.abs( self.pre_processed_data[channel_i][station_i][enhancement_i] )) > 0:

						a = self.pre_processed_data[0][station_i][enhancement_i] 
						b = self.pre_processed_data[channel_i][station_i][enhancement_i]
						
						buf = correlationcoef( a = a, b = b, \
							maxscale = int(self.l_windows[0][enhancement_i]/3.), scales=self.procscales)
							#scales = [ int(self.l_windows[0][enhancement_i]/8.) ] )

						# no ~zeros
						buf[buf < dtiny] = dtiny

						# product enhancement (no nans, no infs)
						cf[station_i][np.isfinite(buf)] *= buf[np.isfinite(buf)] 

						# if no signal
						cf[station_i][ np.isnan(self.pre_processed_data[0][station_i][enhancement_i]) ] = np.nan
						cf[station_i][ np.isnan(self.pre_processed_data[channel_i][station_i][enhancement_i]) ] = np.nan

		cf = 1-cf # **(1./self.enhancement_factor))
		cf[ cf< dtiny ] = dtiny

		return cf

	def plot(self, **kwargs):        
		return stream_processor_plot( self.data, self.output(), **kwargs)



class predom_period(object):

	"""
	Produces an estimation/proxy of the pre-dominant period in 
	continuous seismic records. 
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
		- scale: list (multi-scaling by default, optional).
		- statistic: string (default 'averageabs', optional).
		- maxscale: int (default 'None', optional).
	:param: 
		- data of e.g. seismograms.
		- scales: or length of the window used for pre-processing with
			`~trigger.recursive`.
		- statistic: operation parameter in pre-processing with
			`~trigger.recursive`.
		- maxscale: maximum scale the window used for pre-processing 
			with `~trigger.recursive`.
	___________
	.. rubric:: _`Default Attributes`
	___________
	.. rubric:: _`Default Methods`
		- `~trigger.predom_period.output`: returns the results.
		- `~trigger.predom_period.plot`: displays the output with 
			`~trigger.stream_multiplexor_plot`.
		- `~trigger.predom_period.correlate`: uses the 
			`~trigger.Correlate` class to compare data-streams.
		- `~trigger.predom_period.ratio`: uses the 
			`~trigger.Ratio` class to compare data-streams.
	___________
	.. rubric:: Example

		Plot the data-streams:
		>>> import trigger
		>>> ... .plot()

	"""

	def __init__(self, data, preprocessor='sumsquare', scales=[800], maxscale=1000 ): 

		# stores input parameters
		self.preprocessor = preprocessor
		self.scales = scales
		self.maxscale = maxscale

		# get (station, scale, sample) array any way (for Trace & Stream inputs)
		self.data, self.original_data = trace2stream(data.copy())

	def output(self):
		# Multiplex the pre-processed data
		
		from obspy.signal.tf_misfit import cwt

		(tmax,nmax) = streamdatadim(self.data)
		nscale = len(self.scales)
		channels = np.zeros(( 2, tmax, nscale**2, nmax )) 
		l_windows = np.zeros(( 2, nscale**2 ))  

		timeseries, scales = recursive(self.data, operation=self.preprocessor, scales=self.scales, maxscale=self.maxscale) 
		d_timeseries, scales = recursive(self.data, operation='d-sumsquare', scales=self.scales, maxscale=self.maxscale)  

		for trace_i, trace in enumerate(self.data):

			for tserie_i in range((d_timeseries.shape)[1]):
				channels[0, trace_i, :] += 2 * np.pi * (timeseries[trace_i, tserie_i, :] / (d_timeseries[trace_i, tserie_i, :]/trace.stats.delta))**.5
			channels[0, trace_i, :] /= (d_timeseries.shape)[1]

			fmin = 0.1
			fmax=1./trace.stats.delta/2

			tf = np.abs(cwt(trace, trace.stats.delta, w0=6, fmin=fmin, fmax=fmax, nf=32))
			f = np.logspace(np.log10(fmin), np.log10(fmax), tf.shape[0])
			channels[1, trace_i, :] = 1./f[tf.argmax(axis=0)]

		return channels, nscale+1, l_windows

	def plot(self):
		channels, n, l = self.output() 
		return stream_multiplexor_plot( self.data, channels )




def trigger_onset(charfct, thr_on=.1, trace=None, thr_off=None, max_len_delete=True, onset_refine=True):
	"""
	Calculate trigger on and off times.

	Given thr_on and thr_off calculate trigger on and off times from
	characteristic function.

	This method is written in pure Python and gets slow as soon as there
	are more then 1e6 triggerings ("on" AND "off") in charfct --- normally
	this does not happen.

	:type charfct: NumPy :class:`~numpy.ndarray`
	:param charfct: Characteristic function of e.g. STA/LTA trigger
	:type thr_on: float
	:param thr_on: Value above which trigger (of characteristic function)
	               is activated (higher threshold)
	:type thr_off: float
	:param thr_off: Value below which trigger (of characteristic function)
	    is deactivated (lower threshold)
	:type max_len: int
	:param max_len: Maximum length of triggered event in samples. A new
	                event will be triggered as soon as the signal reaches
	                again above thr_on.
	:type max_len_delete: bool
	:param max_len_delete: Do not write events longer than max_len into
	                       report file.
	:rtype: List
	:return: Nested List of trigger on and of times in samples
	"""
	# 1) find indices of samples greater than threshold
	# 2) calculate trigger "of" times by the gap in trigger indices
	#    above the threshold i.e. the difference of two following indices
	#    in ind is greater than 1
	# 3) in principle the same as for "of" just add one to the index to get
	#    start times, this operation is not supported on the compact
	#    syntax
	# 4) as long as there is a on time greater than the actual of time find
	#    trigger on states which are greater than last of state an the
	#    corresponding of state which is greater than current on state
	# 5) if the signal stays above thr_off longer than max_len an event
	#    is triggered and following a new event can be triggered as soon as
	#    the signal is above thr_on

	from collections import deque
	from obspy.signal.tf_misfit import cwt
	
	if thr_off is None:
		thr_off = thr_on/2.

	thr_d = .5
	n=2

	dtrace = np.gradient(abs(trace.data))
	dcharfct = np.gradient(charfct.copy())
	acharfct = np.gradient(dcharfct.copy())

	if isinstance(trace, Trace):
		fmin = 0.1
		fmax=1./trace.stats.delta/2
		tf = np.abs(cwt(trace, trace.stats.delta, w0=6, fmin=fmin, fmax=fmax, nf=32))
		f = np.logspace(np.log10(fmin), np.log10(fmax), tf.shape[0])
		tp = 1./f[tf.argmax(axis=0)]
	else:
		max_len_delete=False

	ind1 = np.where( (dcharfct > thr_on) + (charfct > thr_on) )[0] #(charfct > thr_on) + (smoothed_dcharfct > thr_on*thr_d) )[0]

	if len(ind1) == 0:
	    return []
	
	on = deque([ind1[0]])

	# last occurence is missed by the diff, add it manually
	on.extend(ind1[np.where(np.diff(ind1) > 1)[0] + 1].tolist())

	# shift where each onset begins 
	if onset_refine:
		of = deque([])
		for e in range(len(on)): 
			#print on[e]
			tmax =  on[e] + np.argmax( dcharfct[on[e]:]<0 )
			onset_period =  on[e] + 5*np.argmax( dcharfct[on[e]:]<0 )
			on[e] = on[e] - (np.argmax( np.fliplr(np.atleast_2d(dcharfct[np.max([0,on[e]-10]):on[e]]))[0]<0 ) )
			# for t in range(on[e]-1,0,-1): 
			# 	if np.max(dcharfct[np.max([0,t-10]):t]) >= 0. : 
			# 		on[e] = t
			# 		if dcharfct[np.max([0,t-1])] < 0. or np.min(charfct[np.max([0,t-10]):t-1]) >= charfct[t+1] : #np.sum(dcharfct[np.max([0,t-10]):tmax]) >= np.sum(dcharfct[t:tmax])*.85:  #or np.sum(dtrace[np.max([0,t-10]):tmax]) >= np.sum(dtrace[t:tmax]):
			# 			break
			# 	# else:
			# 	# 	break
			#onset_period =  tmax + np.argmax( charfct[tmax:] < charfct[on[e]] + (charfct[tmax]-charfct[on[e]])/2. )
			#onset_period =  on[e]+ ((np.mean( tp[on[e]: np.min([onset_period, len(dcharfct)-1]) ]))/tr.stats.delta)
			of.extend([onset_period])
			#print '   ->',on[e],of[e],onset_period
	else:
		of = deque([-1])

		# determine the indices where charfct falls below off-threshold
		ind2 = np.where( (charfct > thr_off) )[0]
		ind2_ = np.empty_like(ind2, dtype=bool)
		ind2_[:-1] = np.diff(ind2) > 1

		# last occurence is missed by the diff, add it manually
		ind2_[-1] = True
		of.extend(ind2[ind2_].tolist())

	pick = []

	for e in range(len(on)): 
		#print 'on[e], of[e]:',on[e], of[e] 
		# while on[0] <= of[0]:
		# 	print '   on[0] <= of[0] : poped',on[0]
		# 	on.popleft()
		while of[e] < on[e]:
			#print '   of[e] < on[e] : poped',of[0]
			of.popleft()

		if max_len_delete:
			trig_d = (of[e]-on[e])*trace.stats.delta
			trig_p = np.mean( tp[on[e]:of[e] ])
			if trig_d < (trig_p*2/3.) : #or trig_d > (trig_p*5/3.)
				# on.popleft()
				# of.popleft()
				if len(on) == 0:
					return []
				continue

		pick.append([on[e], of[e]])

	return np.array(pick, dtype=np.int64)


def plot_trigger(show=True, charfct=None, thr_on=.1, trace=None, thr_off=None, **kwargs):
	"""
	Plot characteristic function of trigger along with waveform data and
	trigger On/Off from given thresholds.

	:type trace: :class:`~obspy.core.trace.Trace`
	:param trace: waveform data
	:type charfct: :class:`numpy.ndarray`
	:param charfct: characteristic function as returned by a trigger in
	    :mod:`obspy.signal.trigger`
	:type thr_on: float
	:param thr_on: threshold for switching trigger on
	:type thr_off: float
	:param thr_off: threshold for switching trigger off
	:type show: bool
	:param show: Do not call `plt.show()` at end of routine. That way,
	    further modifications can be done to the figure before showing it.
	"""
	import matplotlib.pyplot as plt
	if thr_off is None:
		thr_off = thr_on/2.

	df = trace.stats.sampling_rate
	npts = trace.stats.npts
	t = np.arange(npts, dtype=np.float32) / df
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(t, trace.data, 'k')
	ax2 = fig.add_subplot(212, sharex=ax1)
	ax2.plot(t, charfct, 'k')
	on_off = np.array(trigger_onset(charfct, thr_on, trace, thr_off, **kwargs))
	i, j = ax1.get_ylim()
	try:
		ax1.vlines(on_off[:, 0] / df, i, j, color='r', lw=2,
					label="Trigger On")
		ax1.vlines(on_off[:, 1] / df, i, j, color='b', lw=2,
					label="Trigger Off")
		ax1.legend(loc=2)
	except IndexError:
		pass
	ax2.axhline(thr_on, color='red', lw=1, ls='--', label="Threshold On")
	ax2.axhline(thr_off, color='blue', lw=1, ls='--', label="Threshold Off")
	ax2.set_xlabel("Time after %s [s]" % trace.stats.starttime.isoformat())
	ax2.set_ylim([0., np.max(charfct)])
	ax2.legend(loc=2)
	fig.suptitle(trace.id)
	fig.canvas.draw()
	if show:
		plt.show()
	return ax1

def row_derivate(cf, **kwargs):

	return (np.gradient(cf, **kwargs))[1]

def row_kurtosis(cf, window=100, **kwargs):

	return rolling_kurt(cf, window, **kwargs)


def stream_trim_cf(stream, cf, threshold=.2):
	"""
	Extract wavelets of seismic body-wave arrival from continuous 
	seismic records. 

	In practice, it detects the onset of local maximum in the 
	characteristic functions using `~trigger` and trims the related 
	data.  
	______
	:type: 
		- ObsPy:class:`~obspy.core.stream`.
	:param: 
		- data of e.g. seismograms.
	_______
	:rtype: 
		- 
	:return: 
		- 
	___________
	.. rubric:: Example

		>>> import trigger

	"""
	l = 0.25 # dominant period

	(tmax,nmax) = streamdatadim(stream)
	cflets = np.zeros(( tmax, 2+(stream[0]).stats.sampling_rate*2 ))  
	cflets[:] = np.nan

	wavelets = stream.copy()
	dcf = abs((np.gradient(cf))[1])
	dcf = abs((np.gradient(dcf))[1])
	dcf = abs((np.gradient(dcf))[1])

	cflets = dcf.copy()
	cflets *= (cf+(1-threshold))**2.
	#cflets[cflets<threshold/100.] = 0.


	# for t, trace in enumerate(wavelets):

	# 	cf[t][:trace.stats.sampling_rate]=0

	# 	pick = np.argmax(cf[t])
	# 	if pick > 0 :
	# 		#pick += np.argmax(dcf[t][pick - l/trace.stats.delta : pick + l/trace.stats.delta ]) - l/trace.stats.delta

	# 		pick += np.argmax(dcf[t][pick - l/trace.stats.delta : pick ]) - l/trace.stats.delta

	# 		pick += np.argmax(d2cf[t][pick - l/trace.stats.delta/10 : pick ]) - l/trace.stats.delta/10

	# 		# noise_cf = [ np.nanmedian(cf[t][pick - l/trace.stats.delta : pick ]), np.nanstd(cf[t][pick - l/trace.stats.delta : pick ]) ]
	# 		# pick += np.argmax(cf[t][pick - l/trace.stats.delta : pick ]>noise_cf[0]+noise_cf[1]) - l/trace.stats.delta

	# 	wlstart = trace.stats.starttime + pick*trace.stats.delta - l
	# 	wlend = trace.stats.starttime + pick*trace.stats.delta  + l
	# 	trace.trim(wlstart, wlend)
		
	# 	test = cf[t][pick - l/trace.stats.delta : pick + l/trace.stats.delta ]
	# 	cflets[t][:len(test)] = test

	# 	#plotTfr(trace.data, dt=trace.stats.delta, fmin=0.1, fmax=trace.stats.sampling_rate/2)



	# # nfft=(wavelets[0]).stats.sampling_rate
	# # spec = np.zeros((tmax, nfft // 2 + 1), dtype=np.complex)

	# # fig = plt.figure()
	# # ax = fig.gca() 
	# # f_lin = np.linspace(0, 0.5 / (wavelets[0]).stats.delta, nfft // 2 + 1)

	# # for t, trace in enumerate(wavelets): 
	# # 	spec[t] = np.abs( np.fft.rfft(trace.data, n=int(nfft)) * trace.stats.delta ) ** 2
	# # 	ax.semilogx(f_lin, spec[t])

	return wavelets, cflets


# def ggg(...):
# 	"""
# 	Plot the given seismic wave radiation pattern as a color-coded surface 
# 	or focal sphere (not exactly as a beach ball diagram).
# 	_______
# 	:param: 
#		- (type) String that specifies which trigger is applied (e.g.
# 	    	``'recstalta'``).
# 	:param: 
#		- (option) Necessary keyword arguments for the respective
# 		trigger.
#	_________
# 	.. note::

# 		The raw data is not accessible anymore afterwards.
#	___________
# 	.. rubric:: _`Supported Trigger`

# 	``'classicstalta'``
# 		Computes the classic STA/LTA characteristic function (uses
# 		:func:`obspy.signal.trigger.classicSTALTA`).
#	___________
# 	.. rubric:: Example

# 		>>> ss.ssss('sss', ss=1, sss=4)  # doctest: +ELLIPSIS
# 		<...aaa...>
# 		>>> aaa.aaa()  # aaa
#	_________
# 	.. plot::

# 		from ggg import ggg
# 		gg = ggg()
# 		gg.ggggg("ggggg", ggggg=3456)
# 		gg.ggg()
# 	"""
