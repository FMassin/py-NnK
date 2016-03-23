# -*- coding: utf-8 -*-
"""
wrapper - Module for wrapping of existing functions 
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
import glob
import sys  
import os
import random
import re
import numpy as np
from obspy.core.stream import Stream, read
from matplotlib.patches import Circle
from  obspy.clients.filesystem.sds import Client
from  obspy.core.utcdatetime import UTCDateTime

def colormap_rgba(ax = None, bits = 256., labels=['R', 'G', 'B', 'A'] ):

    half_bits = bits/2.

    RGBA_map_NSPE = np.ones([bits,bits,4]) 
    c = np.matlib.repmat(np.arange(bits),bits,1)
    l = np.transpose(c)

    g = [ half_bits-np.cos(np.pi/6)*half_bits , half_bits+np.sin(np.pi/6)*half_bits]
    b = [ half_bits+np.cos(np.pi/6)*half_bits , half_bits+np.sin(np.pi/6)*half_bits]
    
    l_triangle = (((b[1]**2+half_bits**2)**.5)) 
    hc = ( l_triangle/2 / np.cos(np.pi/6) )


    RGBA_map_NSPE[:,:,0] = (((half_bits - c)**2 + (       0. - l)**2 )**.5)    # red is noise
    RGBA_map_NSPE[:,:,1] = (((     g[0] - c)**2 + (     g[1] - l)**2 )**.5)    # green is S
    RGBA_map_NSPE[:,:,2] = (((     b[0] - c)**2 + (     b[1] - l)**2 )**.5)    # Blue is P
    RGBA_map_NSPE[:,:,3] = (((half_bits - c)**2 + (half_bits - l)**2 )**.5)    # brighness is error

    RGBA_map_NSPE[RGBA_map_NSPE[:,:,3]>=half_bits,:] = 0
    for i in [0, 1, 2] :
        RGBA_map_NSPE[RGBA_map_NSPE[:,:,i]>=l_triangle,i] = l_triangle

    RGBA_map_NSPE[:,:,0] /= np.max(RGBA_map_NSPE[:,:,0])
    RGBA_map_NSPE[:,:,1] /= np.max(RGBA_map_NSPE[:,:,1])
    RGBA_map_NSPE[:,:,2] /= np.max(RGBA_map_NSPE[:,:,2])
    RGBA_map_NSPE[:,:,3] = RGBA_map_NSPE[:,:,3]**0.3
    RGBA_map_NSPE[:,:,3] /= np.max(RGBA_map_NSPE[:,:,3])

    RGBA_map_NSPE[:,:,:3] = 1-RGBA_map_NSPE[:,:,:3]

    #RGBA_map_NSPE[:,:,0] = 0 # mute red
    #RGBA_map_NSPE[:,:,1] = 0 # mute green
    #RGBA_map_NSPE[:,:,2] = 0 # mute blue
    #RGBA_map_NSPE[:,:,3] = 1 # mute brightness

    imgplot = ax.imshow(RGBA_map_NSPE, interpolation='nearest')#, extent=(0,bits,0,bits))
    
    ax.text(         half_bits,     0-half_bits/10, labels[0], verticalalignment='center', horizontalalignment='center')
    ax.text( g[0]-half_bits/10,  g[1]+half_bits/10, labels[1], verticalalignment='center', horizontalalignment='center', rotation=-60.)
    ax.text( b[0]+half_bits/10,  b[1]+half_bits/10, labels[2], verticalalignment='center', horizontalalignment='center', rotation=60.)
    ax.text(         half_bits,          half_bits, labels[3], verticalalignment='center', horizontalalignment='center')
    ax.add_patch(Circle((half_bits, half_bits), half_bits,fill=False,edgecolor="white",linewidth=3) )
    ax.add_patch(Circle((half_bits, half_bits), half_bits-half_bits/50,fill=False) )
    ax.axis('off')
    ax.set_xlim([-2, bits+2])
    ax.set_ylim([bits+2, -2])

    return imgplot

def readallchannels(dataset, operation='eventdir'):
    """
    wrapps obspy.core.stream.read so various seismic file 
    formats can be read in one pass.

    Assumes data files are organized in event directories.
    This can be improved
    
    :type dataset: list
    :param dataset: files to read.
    :rtype: class obspy.core.stream
    :return: class of data streams.
    """
    # 1) Iterate over full files names
    # 2) Get all available channel of each file 
    # 3) Read individually in stream.

    

    
    # Reading the waveforms
    m=0
    eventset=[]
    waveformset = Stream()
    for e, eventfile in enumerate(dataset):  

        eventfile = eventfile.strip()
        waveformset += read(eventfile)
        eventset.append(e)
        (root,ext)=os.path.splitext(eventfile)
        channelset = glob.glob(os.path.dirname(eventfile)+'/*'+waveformset[-1].stats.station+'*'+ext)
        
        for c, channelfile in enumerate(channelset):    
            if not eventfile == channelfile :
                eventset.append(e)
                waveformset += read(channelfile) # file with uncorrect metadata may be ignored (ex: YHHN in station metatdata)
                
                #df = waveformset[-1].stats.sampling_rate
                #cft = classicSTALTA(waveformset[-1].data, int(5 * df), int(10 * df))
                #plotTrigger(waveformset[-1], cft, 1.5, 0.5)

    return waveformset

def rand2mat(client, D_range, M_range, T_range, metadata, ratetarget=.95, ntarget=1000, snr=3):

    M_step = (M_range[1]-M_range[0])/2.
    D_step = (D_range[1]-D_range[0])/2.
    [M_grid, D_grid] = np.meshgrid( M_range, D_range)
    occupied = np.zeros([len(D_range),len(M_range),4]) 
    RGBA_NSPE = np.zeros([len(D_range),len(M_range),4]) 

    target = len(M_range)*len(D_range)* 4 * ratetarget

    ntest = 0

    for arrival in np.random.permutation(len(metadata)) :

        D_cell = np.where(np.logical_and(D_range >= metadata[arrival][5]-D_step, D_range < metadata[arrival][5]+D_step))
        M_cell = np.where(np.logical_and(M_range >= metadata[arrival][9]-M_step, M_range < metadata[arrival][9]+M_step))
        
        #print metadata[arrival]
        #print D_cell, M_cell, T_range.index(metadata[arrival][2][0]), occupied[D_cell[0], M_cell[0],T_range.index(metadata[arrival][2][0])]
        
        if (metadata[arrival][2][0] in T_range) and (len(D_cell[0]) == 1) and (len(M_cell[0]) == 1):       

            T_cell = T_range.index(metadata[arrival][2][0])

            if  occupied[D_cell[0], M_cell[0],T_cell] == 0 :#and np.isfinite(metadata[arrival][8]):            

                t = 0
                try:
                    t = UTCDateTime(str(metadata[arrival][4]))  
                except:
                    pass#print metadata[arrival][4]            
                try:
                    t += (metadata[arrival][6]-t.hour)*60*60 + \
                        (metadata[arrival][7]-t.minute)*60 + \
                        (metadata[arrival][8]-(t.second+t.microsecond/1000000.))
                except:
                    pass#print  metadata[arrival][6], metadata[arrival][7], metadata[arrival][8]

                if t>0:
                    st = client.get_waveforms(metadata[arrival][0], metadata[arrival][1], "*", "*[ENZ]", t-10, t+10)
                    st.trim(t-10, t+10)  
                    
                    if len(st) > 1 and st[0].stats.starttime == t-10 :
                        if np.sum((st[0]).data[(st[0]).stats.npts/2.:(st[0]).stats.npts/2.+40]**2.) > snr*np.sum((st[0]).data[(st[0]).stats.npts/2.-40:(st[0]).stats.npts/2.]**2.) : 
                            
                            RGBA_NSPE[D_cell[0], M_cell[0], T_cell] = int(arrival)
                            
                            occupied[D_cell[0], M_cell[0],3] = 1
                            occupied[D_cell[0], M_cell[0],0] = 1
                            occupied[D_cell[0], M_cell[0],T_cell] = 1

                            #print D_cell[0], M_cell[0], T_cell
                            if np.sum(occupied) >= target: 
                                print "fill breaking"
                                break               
        ntest +=1
        if ntest >= ntarget: 
            print "N breaking"
            break 

    return RGBA_NSPE

def readfullfilenames(paths, operation=None):
    """
    Wrapps readlines so several text files could be 
    loaded in one pass.

    Reads full file names and paths in given catalog 
    files.

    Optionally stitch catalog content with the path of
    the given catalog.

    
    :type dataset: list
    :param dataset: files (patterns) to read.
    :rtype: list
    :return: contents of files.
    """
    # 1) Test files and search patterns.
    # 2) Read files and add to returned list.
    # 3) Concatenate list and catalog paths if asked.
    
    
    files=[]
    dataset=[]

    for p in paths:
        files.extend(glob.glob(p))

    
    # Incremental slurps 
    md=-1
    for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        #print "Reading",name,"..."
        with open (name, "r") as myfile:
            dataset.extend(myfile.readlines()) # get ride of \n !!!
        
        if operation is 'relative':
            pathname=os.path.dirname(name)
            for d, data in enumerate(dataset):
                if d >= md:
                    dataset[d]=pathname+'/'+data.strip()
                    md=d

    ## Stream processing version
    ## the stream version seems less efficient than the slurp version
    ## probably due to the incremental increase of list dataset
    ## Also something is wrong in the returned dataset
    #
    # for fname in files:
    #     print "Reading",fname,"..."
    #     if operation is None :
    #         with open(fname, 'r+') as f:
    #             for line in f:
    #                 dataset.extend(line)

    #     if operation is 'relative':
    #         pathname=os.path.dirname(fname)
    #         with open(fname, 'r+') as f:
    #             for line in f:                    
    #                 dataset.extend(pathname+'/'+line)    

    return dataset

def randomsample(dataset,Nsample,searchregex=None, test=None) :
    """
    Wrapps random.sample so list elements are randomly
    resampled to the given dimension (2nd arg).

    Optionally filter the sample with regular expression.

    Optionally filter the sample with only existing full
    file names in elements.

    
    :type dataset: list
    :param dataset: files (patterns) to resample.
    :type Nsample: Variable
    :param Nsample: total number in returned sample.
    :type searchregex: String
    :param searchregex: regular expression for filtering.
    :type test: String
    :param test: None or 'existingonly' to test file existence.
    :rtype: list
    :return: contents of files.
    """
    # 1) Test files and search patterns.
    # 2) Read files and add to returned list.
    # 3) Concatenate list and catalog paths if asked.
    

    # Usual instance of random.sample, if no optinal test are 
    # requested, the wrapper is not usefull at all
    tempdataset = random.sample(dataset, Nsample)

    if (test is 'existingonly') or (searchregex):

        for e, eventfile in enumerate(tempdataset):  
            
            test1 = True
            if test is 'existingonly':
                test1 = os.path.exists(eventfile)

            test2 = re.search(searchregex, eventfile)
            
            while (not test1) or (not test2):
                
                eventfile = random.sample(dataset, 1)
                eventfile=eventfile[0]
                
                test1 = True
                if test is 'existingonly':
                    test1 = os.path.exists(eventfile)

                test2 = re.search(searchregex, eventfile)
            
            tempdataset[e]=eventfile

    return tempdataset

