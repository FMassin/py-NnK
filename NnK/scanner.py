# -*- coding: utf-8 -*-
"""
source - Module for seismic sources modeling.

This module provides class hierarchy for earthquake modeling and 
 representation.
______________________________________________________________________

.. note::    
    
    Functions and classes are ordered from general to specific.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from obspy import read, Trace, Stream
from obspy.core.trace import Stats
from obspy.core.event.source import farfield
from obspy.imaging.scripts.mopad import MomentTensor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
from mpl_toolkits.basemap import Basemap

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    
    v = [3, 5, 0]
	axis = [4, 4, 1]
	theta = 1.2 

	print(np.dot(rotation_matrix(axis,theta), v)) 
	# [ 2.74911638  4.77180932  1.91629719]

    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    

def globe(r=1., n=100.):

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)
    
    return [r * radius * np.cos(theta), r * radius * np.sin(theta), r * z]

def sphere(r=1.,n=100.):
    """
    Produce the polar coordinates of a sphere. 
    ______________________________________________________________________
    :type r, n: variables
    :param r, n: radius and number of resolution points.
    
    :rtype : list
    :return : 3d list of azimuth, polar angle and radial distance.

    .. seealso::

        numpy.linspace : produces the resolution vectors.

        numpy.meshgrid : produce the grid from the vectors.

    .. rubric:: Example

        # import the module
        import source

        # 100 coordinates on sphere or radius 5
        points = source.sphere(r=5, n=100)

        # Unit sphere of 50 points
        points = source.sphere(n=50)               

    .. plot::

        # run one of the example
        points = source.spherical_to_cartesian(points)

        # plot using matplotlib 
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax=fig.gca(projection='3d')
        ax.set_aspect("equal")
        ax.plot_wireframe(points[0], points[1], points[2], color="r")
        plt.show() 
    ______________________________________________________________________
    """
    # Get the resolution (sorry this is ugly)
    c = 0.038 ;
    na = n**(.5+c)
    nt = n**(.5-c)
    [a, t] = np.meshgrid( np.linspace(0, 2*np.pi, na+1), np.linspace(0, 1*np.pi, nt) )
    r = np.ones(a.shape)*r
    
    #golden_angle = np.pi * (3 - np.sqrt(5))
    #theta = golden_angle * np.arange(n)
    #z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    #radius = np.sqrt(1 - z * z)
    
    #points = np.zeros((n, 3))
    #points[:,0] = radius * np.cos(theta)
    #points[:,1] = radius * np.sin(theta)
    #points[:,2] = z

    return [ a, t, r ] #cartesian_to_spherical(points.T) #


def cartesian_to_spherical(vector):
    """
    Convert the Cartesian vector [x, y, z] to spherical coordinates 
    [azimuth, polar angle, radial distance].
    ______________________________________________________________________
    :type vector : 3D array, list | np.array
    :param vector :  The vector of cartessian coordinates.

    :rtype : 3D array, np.array
    :return : The spherical coordinate vector.
    
    .. note::

        This file is extracted & modified from the program relax (Edward 
            d'Auvergne).

    .. seealso::
    
        http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
    ______________________________________________________________________
    
    """

    # Make sure we got np.array 
    if np.asarray(vector) is not vector:
        vector = np.asarray(vector)

    # The radial distance.
    radius = np.sqrt((vector**2).sum(axis=0))

    # The horizontal radial distance.
    rh = np.sqrt(vector[0]**2 + vector[1]**2) 

    # The polar angle.
    takeoff = np.arccos( vector[2] / radius )
    takeoff[radius == 0.0] = np.pi / 2 * np.sign(vector[2][radius == 0.0])
    #theta = np.arctan2(vector[2], rh)

    # The azimuth.
    azimuth_trig = np.arctan2(vector[1], vector[0])

    # Return the spherical coordinate vector.
    return [azimuth_trig, takeoff, radius]


def spherical_to_cartesian(vector):
    """
    Convert the spherical coordinates [azimuth, polar angle
    radial distance] to Cartesian coordinates [x, y, z].

    ______________________________________________________________________
    :type vector : 3D array, list | np.array
    :param vector :  The spherical coordinate vector.

    :rtype : 3D array, np.array
    :return : The vector of cartesian coordinates.
    
    .. note::

        This file is extracted & modified from the program relax (Edward 
            d'Auvergne).

    .. seealso::
    
        http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
    ______________________________________________________________________
    """

    # Unit vector if r is missing
    if len(vector) == 2 :
        radius =1
    else:
        radius=vector[2]

    # Trig alias.
    sin_takeoff = np.sin(vector[1])

    # The vector.
    x = radius * sin_takeoff * np.cos(vector[0])
    y = radius * sin_takeoff * np.sin(vector[0])
    z = radius * np.cos(vector[1])

    return [x, y, z]


def project_vectors(b, a):
    """
    Project the vectors b on vectors a.
    ______________________________________________________________________
    :type b : 3D np.array
    :param b :  The cartesian coordinates of vectors to be projected.

    :type a : 3D np.array
    :param a :  The cartesian coordinates of vectors to project onto.

    :rtype : 3D np.array
    :return : The cartesian coordinates of b projected on a.
    
    .. note::

        There is maybe better (built in numpy) ways to do this.

    .. seealso::
    
        :func:`numpy.dot()` 
        :func:`numpy.dotv()`
    ______________________________________________________________________
    """

    # Project 
    ## ratio (norm of b on a / norm of a)
    proj_norm = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] 
    proj_norm /=  (a[0]**2 + a[1]**2 + a[2]**2)+0.0000000001
    ## project on a
    b_on_a = proj_norm * a


    return b_on_a


def vector_normal(XYZ, v_or_h):
    """
    Compute the vertical or horizontal normal vectors.
    ______________________________________________________________________
    :type XYZ : 3D np.array
    :param b :  The cartesian coordinates of vectors.

    :type v_or_h : string
    :param v_or_h :  The normal direction to pick.

    :rtype : 3D np.array
    :return : Requested normal on vertical or horizontal plane 
        ('v|vert|vertical' or 'h|horiz|horizontal').
    
    .. rubric:: _`Supported v_or_h`
        
        In origin center sphere: L: Radius, T: Parallel, Q: Meridian

        ``'v'``
            Vertical.

        ``'h'``
            horizontal.

        ``'r'``
            Radial component.

        ``'m'``
            The meridian component (to be prefered to vertical).

    .. note::

        There is maybe better (built in numpy) ways to do this.

    .. seealso::
    
        :func:`numpy.cross()` 
    ______________________________________________________________________
    """

    oneheightydeg =  np.ones(XYZ.shape) * np.pi
    ninetydeg =  np.ones(XYZ.shape) * np.pi/2.

    ATR = np.asarray(cartesian_to_spherical(XYZ))

    if v_or_h in ('Q', 'm', 'meridian', 'n', 'nhr', 'normal horizontal radial', 'norm horiz rad'): 
        ATR_n = np.array([ ATR[0] , ATR[1] - ninetydeg[0], ATR[2]])
        XYZ_n = np.asarray(spherical_to_cartesian(ATR_n))
    elif v_or_h in ('T', 'h', 'horizontal', 'horiz'): 
        ATR_n = np.array([ ATR[0] - ninetydeg[0], ATR[1]+(ninetydeg[0]-ATR[1]) , ATR[2]])
        XYZ_n = np.asarray(spherical_to_cartesian(ATR_n))
    elif v_or_h in ('v', 'vertical', 'vertical'): 
        ATR_n = np.array([ ATR[0], ATR[1]-ATR[1] , ATR[2]])
        XYZ_n = np.asarray(spherical_to_cartesian(ATR_n))
    elif v_or_h in ('L', 'r', 'radial', 'self'): 
        XYZ_n = XYZ

    return XYZ_n

def mt_full(mt):
    """    
    Takes 6 components moment tensor and returns full 3x3 moment 
    tensor.
    ______________________________________________________________________
    :type mt : list or np.array.
    :param mt : moment tensor NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz, 
        the six independent components).

    :rtype : np.array 3x3.
    :return : full 3x3 moment tensor.

    .. note::

        Use obspy.imaging.scripts.mopad to get correct input.
    ______________________________________________________________________
    """

    # Make sure we got np.array 
    if np.asarray(mt) is not mt:
        mt = np.asarray(mt)

    if len(mt) == 6:        
        mt = np.array(([[mt[0], mt[3], mt[4]],
            [mt[3], mt[1], mt[5]],
            [mt[4], mt[5], mt[2]]]))

    if mt.shape != (3,3) :       
        raise Exception('I/O dimensions: only 1x6 or 3x3 input supported.')
    
    return mt


def mt_angles(mt): 
    """    
    Takes 6 components and returns fps tri-angles in degrees, with 
    deviatoric, isotropic, DC and CLVD percentages.
    ______________________________________________________________________
    :type mt : list or np.array.
    :param mt : moment tensor NM x 6 (Mxx, Myy, Mzz, 
        Mxy, Mxz, Myz, the six independent 
        components).

    :rtype : np.array [[1x3], [1x4]].
    :return : full 3x3 moment tensor.

    .. note::
        Nan value are returned for unknown parameters.

        The deviatoric percentage is the sum of DC and CLVD percentages.

        Use obspy.imaging.scripts.mopad to get correct input.
    ______________________________________________________________________
    """

    # Make sure we got np.array 
    if np.asarray(mt) is not mt:
        mt = np.asarray(mt)

    # Getting various formats
    ## if given strike, dip, rake
    if mt.shape == (3,):
        strike, dip, rake = mt
        DC = 100
        CLVD = 0
        iso = 0
        devi = 100
    
    ## if given [[strike, dip, rake], [strike, dip, rake]] (e.g. by MoPad)
    elif mt.shape == (2,3) :
        strike, dip, rake = mt[0]
        DC = 100
        CLVD = 0
        iso = 0
        devi = 100
    
    ## if given [strike, dip, rake, devi]
    elif mt.shape == (4,):
        strike, dip, rake, devi = mt
        DC = np.nan
        CLVD = np.nan
        iso = 0

    ## if given [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    elif mt.shape == (6,) :
        
        mt = MomentTensor(mt,'XYZ') 
        
        DC = mt.get_DC_percentage()
        CLVD = mt.get_CLVD_percentage()
        iso = mt.get_iso_percentage()
        devi = mt.get_devi_percentage()

        mt = mt_angles(mt.get_fps())
        strike, dip, rake = mt[0]

    ## if given full moment tensor
    elif mt.shape == (3,3) :

        mt = mt_angles([mt[0,0], mt[1,1], mt[2,2], mt[0,1], mt[0,2], mt[1,2]])

        strike, dip, rake = mt[0]
        DC, CLVD, iso, devi = mt[1]

    else:         
        raise Exception('I/O dimensions: only [1|2]x3, 1x[4|6] and 3x3 inputs supported.')

    return np.array([[strike, dip, rake], [DC, CLVD, iso, devi]])
    

def disp_component(xyz, disp,comp):
	
	## Get direction(s)
    if comp == None: 
        amplitude_direction = disp      
    else :
        amplitude_direction = vector_normal(xyz, comp)
    ## Project 
    disp_projected = project_vectors(disp, amplitude_direction) 
    ## Norms
    amplitudes = np.sqrt(disp_projected[0]**2 + disp_projected[1]**2 + disp_projected[2]**2) 
    ## Amplitudes (norm with sign)
    amplitudes *= np.sign(np.sum(disp * amplitude_direction, axis=0)+0.00001)
    
    return amplitudes, disp_projected 
    
    
def plot_seismicsourcemodel(disp, xyz, style='*', mt=None, comp=None, ax=None, alpha=0.5, wave='P', cbarxlabel=None, insert_title='', cb = 1) :
    """
    Plot the given seismic wave radiation pattern as a color-coded surface 
    or focal sphere (not exactly as a beach ball diagram).
    ______________________________________________________________________
    :type G : 3D array, list | np.array
    :param G : The vector of cartessian coordinates of the radiation 
        pattern.

    :type xyz : 3D array, list | np.array
    :param xyz : The cartessian coordinates of the origin points of the
        radiation pattern.

    :type style : string
    :param style : type of plot.

    :type mt : list | np.array
    :param mt : moment tensor definition supported by MoPad, used in title.

    :type comp : string
    :param comp : component onto radiation pattern is projected to 
        infer amplitudes (norm and sign).

    :rtype : graphical object
    :return : The axe3d including plot.

    .. rubric:: _`Supported style`

        ``'*'``
            Plot options q, s and p together.

        ``'b'``
            Plot a unit sphere, amplitude sign being coded with color.
             (uses :func:`numpy.abs`).

        ``'q'``
            This a comet plot, displacements magnitudes, directions and 
             final state are given, respectively by the colors, lines and 
             points (as if looking at comets).  

        ``'s'``
            Plot the amplitudes as surface coding absolute amplitude as 
             radial distance from origin and amplitude polarity with color.
             (uses :func:`numpy.abs`).

        ``'f'``
            Plot the amplitudes as a wireframe, coding absolute amplitude as 
             radial distance. The displacement polarities are not 
             represented.

        ``'p'``
            Plot the polarities sectors using a colorcoded wireframe. 
             The magnitudes of displacements are not represented.

    .. rubric:: _`Supported comp`
        
        in origin center sphere: L: Radius, T: Parallel, Q: Meridian
        
        ``'None'``
            The radiation pattern is rendered as is if no comp option is 
            given (best for S wave).

        ``'v'``
            Vertical component (best for S or Sv wave).

        ``'h'``
            horizontal component (best for S or Sh waves, null for P).

        ``'r'``
            Radial component (best for P waves, null for S).

        ``'m'``
            The meridian component (best for S or Sn waves, null for P).

    .. note::

        The radiation pattern is rendered as is if no comp option is given.

        In the title of the plot, SDR stand for strike, dip, rake angles 
        and IDC stand for isotropic, double couple, compensated linear 
        vector dipole percentages.

    .. seealso::
    
        For earthquake focal mechanism rendering, use 
        obspy.imaging.beachball.Beachball() : 
        https://docs.obspy.org/packages/obspy.imaging.html#beachballs

        For more info see  and obspy.imaging.mopad_wrapper.Beachball().
    ______________________________________________________________________
    """

    # Component
    amplitudes, disp_projected = disp_component(xyz, disp, comp)
    
    ## Easiness
    U, V, W = disp_projected
    X, Y, Z = xyz
    amplitudes_surf = np.abs(amplitudes)/np.max(np.abs(amplitudes))

    # Initializing 
    ## Initializing the colormap machinery
    norm = matplotlib.colors.Normalize(vmin=np.min(amplitudes),vmax=np.max(amplitudes))
    c_m = matplotlib.cm.Spectral
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    ## Initializing the plot
    if ax == None:
        fig = plt.figure(figsize=plt.figaspect(1.2))
        ax = fig.gca(projection='3d', xlabel='Lon.', ylabel='Lat.')
    ## Initializing the colorbar
    cbar=[]
    if style not in ('frame', 'wireframe') and cb == 1:
        cbar = plt.colorbar(s_m,orientation="vertical",fraction=0.07, shrink=.7, aspect=10, ax = ax)
        if cbarxlabel==None:
        	cbarxlabel = 'Displacement amplitudes'
        cbar.ax.set_ylabel(cbarxlabel)
    ## Initializing figure keys
    if mt is not None :
        [strike, dip, rake], [DC, CLVD, iso, deviatoric] = mt_angles(mt)
        ax.set_title(r''+wave+'-wave '+insert_title.lower()+'\n(' + str(int(strike)) + ', ' + str(int(dip)) + ', ' + str(int(rake)) + ')$^{SDR}$, (' + str(int(iso)) + ', ' + str(int(DC)) + ', ' + str(int(CLVD)) + ')$^{\%IDC}$')
    
    ## Force axis equal
    arrow_scale = (np.max(xyz)-np.min(xyz))/10
    if style in ('p', 'polarities','b', 'bb', 'beachball'):
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    elif style in ('f', 'w', 'frame', 'wireframe', 's', 'surf', 'surface'): 
        max_range = np.array([(X*np.abs(amplitudes)).max()-(X*np.abs(amplitudes)).min(), (Y*np.abs(amplitudes)).max()-(Y*np.abs(amplitudes)).min(), (Z*np.abs(amplitudes)).max()-(Z*np.abs(amplitudes)).min()]).max() / 2.0
    else : 
        max_range = np.array([(X+arrow_scale).max()-(X+arrow_scale).min(), (Y+arrow_scale).max()-(Y+arrow_scale).min(), (Z+arrow_scale).max()-(Z+arrow_scale).min()]).max() / 2.0
    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()


    # Simple styles   
    ## For a wireframe surface representing amplitudes
    if style in ('f', 'w', 'frame', 'wireframe') :
        
        ax.plot_wireframe(X*amplitudes_surf, Y*amplitudes_surf, Z*amplitudes_surf, rstride=1, cstride=1, linewidth=0.5, alpha=alpha)

    ## For focal sphere, with amplitude sign (~not a beach ball diagram) on unit sphere 
    if style in ('*', 'p', 'polarities'):

        if style is '*':
            alpha =0.1

        polarity_area = amplitudes.copy()
        polarity_area[amplitudes > 0] = np.nan  
        polarity_area[amplitudes <= 0] = 1
        ax.plot_wireframe(X*polarity_area, Y*polarity_area, Z*polarity_area, color='r', rstride=1, cstride=1, linewidth=.5, alpha=alpha)

        polarity_area[amplitudes <= 0] = np.nan 
        polarity_area[amplitudes > 0] = 1
        ax.plot_wireframe(X*polarity_area, Y*polarity_area, Z*polarity_area, color='b', rstride=1, cstride=1, linewidth=.5, alpha=alpha)

    ## For ~ beach ball diagram 
    if style in ('b', 'bb', 'beachball'):

        polarity_area = amplitudes.copy()
        polarity_area[amplitudes >= 0] = 1  
        polarity_area[amplitudes < 0] = -1
        ax.plot_surface(X, Y, Z, linewidth=0, rstride=1, cstride=1, facecolors=s_m.to_rgba(polarity_area)) 

	## For focal sphere, with amplitude colorcoded on unit sphere
    if style in ('x'):

        ax.plot_surface(X, Y, Z, linewidth=0, rstride=1, cstride=1, facecolors=s_m.to_rgba(amplitudes), alpha=alpha) 
	       
    # Complexe styles
    ## For arrow vectors on obs points
    if style in ('*', 'q', 'a', 'v', 'quiver', 'arrow', 'vect', 'vector', 'vectors') :    

        cmap = plt.get_cmap()
        # qs = ax.quiver(X.flatten(), Y.flatten(), Z.flatten(), U.flatten(), V.flatten(), W.flatten(), pivot='tail', length=arrow_scale ) 
        # qs.set_color(s_m.to_rgba(amplitudes.flatten())) #set_array(np.transpose(amplitudes.flatten()))

        Us, Vs, Ws = disp_projected * arrow_scale / disp_projected.max()
        ax.scatter((X+Us).flatten(), (Y+Vs).flatten(), (Z+Ws).flatten(), c=s_m.to_rgba(amplitudes.flatten()), marker='.', linewidths=0)

        for i, x_val in np.ndenumerate(X):
            ax.plot([X[i], X[i]+Us[i]], [Y[i], Y[i]+Vs[i]], [Z[i], Z[i]+Ws[i]],'-', color=s_m.to_rgba(amplitudes[i]) , linewidth=1)
        

    ## For a color-coded surface representing amplitudes
    if style in ('*', 's', 'surf', 'surface'):

        ax.plot_surface(X*amplitudes_surf, Y*amplitudes_surf, Z*amplitudes_surf,linewidth=0.5, rstride=1, cstride=1, facecolors=s_m.to_rgba(amplitudes))    

    
	
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    ax.view_init(15,65)

    plt.show() 
    return ax, cbar


def energy_seismicsourcemodel(G, XYZ) :    
    """
    Evaluate statistical properties of the given seismic wave radiation 
    pattern.
    ______________________________________________________________________
    :type G : 3D array, list | np.array
    :param G : The vector of cartessian coordinates of the radiation 
        pattern.

    :type XYZ : 3D array, list | np.array
    :param XYZ : The cartessian coordinates of the origin points of the
        radiation pattern.

    :rtype : list
    :return : The statistical properties of amplitudes [rms, euclidian norm, average].

    .. todo::

        Compute more properties, add request handling.
    ______________________________________________________________________
    """

    #amplitudes_correction = np.sum(G * XYZ, axis=0)
    #amplitudes_correction /= np.max(np.abs(amplitudes_correction))
    amplitudes = np.sqrt( G[0]**2 + G[1]**2 + G[2]**2 ) #* amplitudes_correction            
    
    weigths = (np.max(abs(amplitudes))- amplitudes)**2
    
    # Classic rms
    rms = np.sqrt(np.nansum((weigths*amplitudes)**2)/np.sum(weigths)) # prod(amplitudes.shape))
    # Euclidian norm
    norm = np.sqrt(np.nansum((weigths*amplitudes)**2))
    # Amplitude average
    average = np.nansum(np.abs(amplitudes*weigths))/np.sum(weigths)
    
    return [rms, norm, average]


class Aki_Richards(object):
    """
    Set an instance of class Aki_Richards() that can be used for 
    earthquake modeling based on Aki and Richards (2002, eq. 4.29).
    ______________________________________________________________________
    :type mt : list
    :param mt : The focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - 
        the six independent components of the moment tensor).

    .. note::

        This object is composed of three methods : radpat, plot and 
        energy.

    .. seealso::

            :meth: `radpat`
            :meth: `plot`
            :meth: `energy`            
    ______________________________________________________________________

    """  

    def __init__(self, mt):
        self.mt = mt
        
    def radpat(self, wave='P', obs_cart=None, obs_sph=None):
        """
        Returns the farfield radiation pattern (normalized displacement) based 
        on Aki and Richards (2002, eq. 4.29) and the cartesian coordinates of 
        the observation points.
        ______________________________________________________________________
        :type self : object: Aki_Richards
        :param self : This method use de self.mt attribute, i.e. the focal 
            mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the six 
            independent components of the moment tensor)

        :type wave: String
        :param wave: type of wave to compute

        :type obs_cart : list | np.array 
        :param obs_cart : 3D vector array specifying the observations points
            in cartesian coordinates. 

        :type obs_sph : list | np.array 
        :param obs_sph : 3D vector array specifying the observations points
            in spherical coordinates (radians). The default is a unit sphere.

        :rtype : np.array 
        :return : 3D vector array with same shape than requested, that 
            contains the displacement vector for each observation point.

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.
            
            In origin center sphere: L: Radius, T: Parallel, Q: Meridian
            
            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

            ``'Sm'``
                Projection of S on the meridian of the focal sphere.

        .. note::

            This is based on MMesch, ObsPy, radpattern.py :
                https://github.com/MMesch/obspy/blob/radpattern/obspy/core/event/radpattern.py

        .. seealso::
    
            Aki, K., & Richards, P. G. (2002). Quantitative Seismology. (J. 
                Ellis, Ed.) (2nd ed.). University Science Books

        .. todo::

            Implement Sv and Sh wave (see below)
        ______________________________________________________________________
        """

        # Special cases ######################################################
        if wave in ('Sv', 'Sv', 'S_v', 'Sv wave', 'Sv-wave'):

            ## Get S waves
            disp, obs_cart = self.radpat(wave='S', obs_cart=obs_cart, obs_sph=obs_sph)
            ## Project on Sv component
            disp = project_vectors(disp, vector_normal(obs_cart, 'v')) 

            return disp, obs_cart

        elif wave in ('Sq', 'SQ', 'Sm', 'SM', 'SN', 'Sn', 'Snrh', 'Snrh wave', 'Snrh-wave'):

            ## Get S waves
            disp, obs_cart = self.radpat(wave='S', obs_cart=obs_cart, obs_sph=obs_sph)
            ## Project on Sh component
            disp = project_vectors(disp, vector_normal(obs_cart, 'Q')) 

            return disp, obs_cart

        elif wave in ('St', 'ST', 'SH', 'Sh', 'S_h', 'Sh wave', 'Sh-wave'):

            ## Get S waves
            disp, obs_cart = self.radpat(wave='S', obs_cart=obs_cart, obs_sph=obs_sph)
            ## Project on Sh component
            disp = project_vectors(disp, vector_normal(obs_cart, 'T')) 

            return disp, obs_cart
        ######################################################################

        # Get full mt
        Mpq = mt_full(self.mt)
        

        # Get observation points
        if obs_sph == None:
            obs_sph = sphere(r=1.,n=1000.)
        ## Get unit sphere, or spherical coordinate if given 
        if obs_cart == None :
            obs_cart = spherical_to_cartesian(obs_sph)
        ## Make sure they are np.array 
        if np.asarray(obs_cart) is not obs_cart:
            obs_cart = np.asarray(obs_cart)
        ## Keeping that in mind
        requestdimension = obs_cart.shape
        obs_cart = np.reshape(obs_cart, (3, np.prod(requestdimension)/3))
        
        # Displacement array    
        # precompute directional cosine array
        dists = np.sqrt(obs_cart[0] * obs_cart[0] + obs_cart[1] * obs_cart[1] +
                        obs_cart[2] * obs_cart[2])

        # In gamma, all points are taken to a unit distance, same angle:
        gammas = obs_cart / dists

        # initialize displacement array
        ndim, npoints = obs_cart.shape
        disp = np.empty(obs_cart.shape)

        # loop through points
        for ipoint in range(npoints):
            
            gamma = gammas[:, ipoint]

            if wave in ('S', 'S wave', 'S-wave'):

                # loop through displacement component [n index]
                Mp = np.dot(Mpq, gamma)
                for n in range(ndim):
                    psum = 0.0
                    for p in range(ndim):
                        deltanp = int(n == p)
                        psum += (gamma[n] * gamma[p] - deltanp) * Mp[p]
                    disp[n, ipoint] = psum

            elif wave in ('P', 'P wave', 'P-wave'):

                gammapq = np.outer(gamma, gamma)
                gammatimesmt = gammapq * Mpq
                
                # loop through displacement component [n index]
                for n in range(ndim):
                    disp[n, ipoint] = gamma[n] * np.sum(gammatimesmt.flatten())

        # Reshape to request dimensions
        obs_cart = np.reshape(obs_cart, requestdimension)
        disp = np.reshape(disp, requestdimension)

        return disp, obs_cart

    def plot(self, wave='P',style='*', comp=None, ax=None, cbarxlabel=None, insert_title='', cb=1) :
        """
        Plot the radiation pattern.
        ______________________________________________________________________
        :type self : object: Aki_Richards
        :param self : This method use de self.mt attribute, i.e. the focal 
            mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the six 
            independent components of the moment tensor)

        :type wave: String
        :param wave: type of wave to compute

        :type style : string
        :param style : type of plot.

        :type comp : string
        :param comp : component onto radiation pattern is projected to 
            infer amplitudes (norm and sign).

        :rtype : graphical object
        :return : The axe3d including plot.

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.

            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

            ``'Sm'``
                Projection of S on the meridian of the focal sphere.

        .. rubric:: _`Supported style`

            ``'*'``
                Plot options q, s and p together.

            ``'b'``
                Plot a unit sphere, amplitude sign being coded with color.
                 (uses :func:`numpy.abs`).

            ``'q'``
                This a comet plot, displacements magnitudes, directions and 
                 final state are given, respectively by the colors, lines and 
                 points (as if looking at comets).  

            ``'s'``
                Plot the amplitudes as surface coding absolute amplitude as 
                 radial distance from origin and amplitude polarity with color.
                 (uses :func:`numpy.abs`).

            ``'f'``
                Plot the amplitudes as a wireframe, coding absolute amplitude as 
                 radial distance. The displacement polarities are not 
                 represented.

            ``'p'``
                Plot the polarities sectors using a colorcoded wireframe. 
                 The magnitudes of displacements are not represented.

        .. rubric:: _`Supported comp`

            In origin center sphere: L: Radius, T: Parallel, Q: Meridian
            
            ``'None'``
                The radiation pattern is rendered as is if no comp option is 
                given (best for S wave).

            ``'v'``
                Vertical component (best for S or Sv wave).

            ``'h'``
                horizontal component (best for S or Sh waves, null for P).

            ``'r'``
                Radial component (best for P waves, null for S).

            ``'n'``
                The meridian component (best for S or Sn waves, null for P).

        .. note::

            The radiation pattern is rendered as is if no comp option is given.

            In the title of the plot, SDR stand for strike, dip, rake angles 
            and IDC stand for isotropic, double couple, compensated linear 
            vector dipole percentages.

        .. seealso::

                :func: `plot_seismicsourcemodel`
                :class: `Aki_Richards.radpat`
        ______________________________________________________________________

        """
        # Get radiation pattern
        G, XYZ = self.radpat(wave)
        if comp == None :
            if wave in ('Sv', 'Sv wave', 'Sv-wave'):
                comp='Q'
            elif wave in ('Sq', 'SQ', 'Sm', 'Sn', 'Snrh','Snrh wave', 'Snrh-wave'):
                comp='Q'
            elif wave in ('ST', 'St', 'Sh', 'Sh wave', 'Sh-wave'):
                comp='T'
            elif wave in ('P', 'P wave', 'P-wave'):
                comp='L'

        return plot_seismicsourcemodel(G, XYZ, style=style, mt=self.mt, comp=comp, ax=ax, wave=wave, cbarxlabel=cbarxlabel, insert_title=insert_title, cb =cb)

    def energy(self, wave='P') :   
        """
        Get statistical properties of radiation pattern.
        ______________________________________________________________________
        :type self : object: Aki_Richards
        :param self : This method use de self.mt attribute, i.e. the focal 
            mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the six 
            independent components of the moment tensor)

        :type wave: string
        :param wave: type of wave to compute

        :rtype : list
        :return : The statistical properties of amplitudes [rms, euclidian norm, average].

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.

            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

            ``'Sm'``
                Projection of S on the meridian of the focal sphere.

        .. seealso::

                :func: `energy_seismicsourcemodel`
                :class: `Aki_Richards.radpat`
        ______________________________________________________________________

        """ 

        # Get radiation pattern and estimate energy
        G, XYZ = self.radpat(wave)  
        estimators = energy_seismicsourcemodel(G, XYZ)
        return estimators


class Vavryeuk(object):
    """
    Set an instance of class Vavryeuk() that can be used for 
    earthquake modeling based on Vavryèuk (2001).
    ______________________________________________________________________
    :type mt : list
    :param mt : The focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - 
        the six independent components of the moment tensor).

    :type poisson: variable
    :param poisson: Poisson coefficient (0.25 by default).

    .. note::

        This object is composed of three methods : radpat, plot and 
        energy.

    .. seealso::

            :meth: `radpat`
            :meth: `plot`
            :meth: `energy`            
    ______________________________________________________________________

    """  

    def __init__(self, mt, poisson=0.25):
        self.mt = mt
        self.poisson = poisson
        
    def radpat(self, wave='P', obs_cart=None, obs_sph=None):
        """
        Returns the farfield radiation pattern (normalized displacement) based 
        on Vavryèuk (2001) and the cartesian coordinates of the observation 
        points.
        ______________________________________________________________________
        :type self : object: Vavryeuk
        :param self : This method use de self.poisson and self.mt attributes, 
            i.e. the focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - 
            the six independent components of the moment tensor).

        :type wave: String
        :param wave: type of wave to compute

        :type obs_cart : list | np.array 
        :param obs_cart : 3D vector array specifying the observations points
            in cartesian coordinates. 

        :type obs_sph : list | np.array 
        :param obs_sph : 3D vector array specifying the observations points
            in spherical coordinates (radians). The default is a unit sphere. 

        :rtype : np.array 
        :return : 3D vector array with same shape than requested, that 
            contains the displacement vector for each observation point.

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.

            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

        .. note::

            This is based on Kwiatek, G. (2013/09/15). Radiation pattern from 
            shear-tensile seismic source. Revision: 1.3 :
            http://www.npworks.com/matlabcentral/fileexchange/43524-radiation-pattern-from-shear-tensile-seismic-source

        .. seealso::            

            [1] Vavryèuk, V., 2001. Inversion for parameters of tensile 
             earthquakes.” J. Geophys. Res. 106 (B8): 16339–16355. 
             doi: 10.1029/2001JB000372.

            [2] Ou, G.-B., 2008, Seismological Studies for Tensile Faults. 
             Terrestrial, Atmospheric and Oceanic Sciences 19, 463.

            [3] Kwiatek, G. and Y. Ben-Zion (2013). Assessment of P and S wave 
             energy radiated from very small shear-tensile seismic events in 
             a deep South African mine. J. Geophys. Res. 118, 3630-3641, 
             doi: 10.1002/jgrb.50274

        .. rubric:: Example

            ex = NnK.source.SeismicSource([0,0,0,0,0,1])
            Svect, obs =  ex.Vavryeuk.radpat('S')                   

        .. plot::

            ex.Vavryeuk.plot()
        ______________________________________________________________________

        """
        # 1) Calculate moving rms and average, see moving()
        # 2) ...
        # 3) ... 
        
        poisson = self.poisson

        # Get angle from moment tensor
        [strike, dip, rake], [DC, CLVD, iso, deviatoric] = mt_angles(self.mt) 
        ## in radians
        [strike, dip, rake] = np.deg2rad([strike, dip, rake])
        ## convert DC ratio to angle
        MODE1 = np.arcsin((100-DC)/100.)        

        # Get observation points
        if obs_cart == None :
            obs_cart = spherical_to_cartesian(sphere(r=1.,n=1000.))
        ## Get unit sphere, or spherical coordinate if given 
        if obs_sph == None :
            obs_sph = cartesian_to_spherical(obs_cart)
        else:
            obs_cart = spherical_to_cartesian(obs_sph)

        ## Make sure they are np.array 
        if np.asarray(obs_sph) is not obs_sph:
            obs_sph = np.asarray(obs_sph)
        if np.asarray(obs_cart) is not obs_cart:
            obs_cart = np.asarray(obs_cart)
        
        # Radiation patterns
        # G is the amplitude for each observation point
        ## Observations are given in spherical angles
        AZM = obs_sph[0]
        TKO = obs_sph[1]

        ## Make sure we got np.array 
        if np.asarray(TKO) is not TKO:
            TKO = np.asarray(TKO)
        if np.asarray(AZM) is not AZM:
            AZM = np.asarray(AZM)

        ## Tensile definitions by Vavryèuk (2001)
        if wave in ('P', 'P-wave', 'P wave'):
            G = np.cos(TKO)*(np.cos(TKO)*(np.sin(MODE1)*(2*np.cos(dip)**2 - (2*poisson)/(2*poisson - 1)) + np.sin(2*dip)*np.cos(MODE1)*np.sin(rake)) - np.cos(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) + np.sin(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1))) + np.sin(AZM)*np.sin(TKO)*(np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1)) + np.cos(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) + np.sin(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.sin(2*strike)*np.cos(rake)*np.sin(dip) - np.sin(2*dip)*np.cos(strike)**2*np.sin(rake)) - np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.cos(strike)**2*np.sin(dip)**2))) - np.cos(AZM)*np.sin(TKO)*(np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) - np.sin(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) + np.cos(AZM)*np.sin(TKO)*(np.cos(MODE1)*(np.sin(2*dip)*np.sin(rake)*np.sin(strike)**2 + np.sin(2*strike)*np.cos(rake)*np.sin(dip)) + np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.sin(dip)**2*np.sin(strike)**2)))
        
        elif wave in ('ST', 'St', 'SH', 'Sh', 'Sh-wave', 'Sh wave', 'SH-wave', 'SH wave'):
            G = np.cos(TKO)*(np.cos(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1)) + np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike))) - np.sin(AZM)*np.sin(TKO)*(np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.cos(AZM)*(np.cos(MODE1)*(np.sin(2*strike)*np.cos(rake)*np.sin(dip) - np.sin(2*dip)*np.cos(strike)**2*np.sin(rake)) - np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.cos(strike)**2*np.sin(dip)**2))) + np.cos(AZM)*np.sin(TKO)*(np.sin(AZM)*(np.cos(MODE1)*(np.sin(2*dip)*np.sin(rake)*np.sin(strike)**2 + np.sin(2*strike)*np.cos(rake)*np.sin(dip)) + np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.sin(dip)**2*np.sin(strike)**2)) + np.cos(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)))
            
        elif wave in ('SV', 'Sv', 'Sv-wave', 'Sv wave', 'SV-wave', 'SV wave'):
            G = np.sin(AZM)*np.sin(TKO)*(np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1)) + np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.sin(2*strike)*np.cos(rake)*np.sin(dip) - np.sin(2*dip)*np.cos(strike)**2*np.sin(rake)) - np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.cos(strike)**2*np.sin(dip)**2))) - np.cos(TKO)*(np.sin(TKO)*(np.sin(MODE1)*(2*np.cos(dip)**2 - (2*poisson)/(2*poisson - 1)) + np.sin(2*dip)*np.cos(MODE1)*np.sin(rake)) + np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) - np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1))) + np.cos(AZM)*np.sin(TKO)*(np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) + np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.sin(2*dip)*np.sin(rake)*np.sin(strike)**2 + np.sin(2*strike)*np.cos(rake)*np.sin(dip)) + np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.sin(dip)**2*np.sin(strike)**2)))
        
        ## Re-using the same programme to get other things ... 
        elif wave in ('S', 'S-wave', 'S wave'):

            # for such definition this the less ugly
            Gsh = np.cos(TKO)*(np.cos(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1)) + np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike))) - np.sin(AZM)*np.sin(TKO)*(np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.cos(AZM)*(np.cos(MODE1)*(np.sin(2*strike)*np.cos(rake)*np.sin(dip) - np.sin(2*dip)*np.cos(strike)**2*np.sin(rake)) - np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.cos(strike)**2*np.sin(dip)**2))) + np.cos(AZM)*np.sin(TKO)*(np.sin(AZM)*(np.cos(MODE1)*(np.sin(2*dip)*np.sin(rake)*np.sin(strike)**2 + np.sin(2*strike)*np.cos(rake)*np.sin(dip)) + np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.sin(dip)**2*np.sin(strike)**2)) + np.cos(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)))
            Gsv = np.sin(AZM)*np.sin(TKO)*(np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1)) + np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.sin(2*strike)*np.cos(rake)*np.sin(dip) - np.sin(2*dip)*np.cos(strike)**2*np.sin(rake)) - np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.cos(strike)**2*np.sin(dip)**2))) - np.cos(TKO)*(np.sin(TKO)*(np.sin(MODE1)*(2*np.cos(dip)**2 - (2*poisson)/(2*poisson - 1)) + np.sin(2*dip)*np.cos(MODE1)*np.sin(rake)) + np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) - np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*dip)*np.cos(strike)*np.sin(rake) - np.cos(dip)*np.cos(rake)*np.sin(strike)) - np.sin(2*dip)*np.cos(strike)*np.sin(MODE1))) + np.cos(AZM)*np.sin(TKO)*(np.sin(TKO)*(np.cos(MODE1)*(np.cos(2*dip)*np.sin(rake)*np.sin(strike) + np.cos(dip)*np.cos(rake)*np.cos(strike)) - np.sin(2*dip)*np.sin(MODE1)*np.sin(strike)) + np.cos(TKO)*np.sin(AZM)*(np.cos(MODE1)*(np.cos(2*strike)*np.cos(rake)*np.sin(dip) + (np.sin(2*dip)*np.sin(2*strike)*np.sin(rake))/2) - np.sin(2*strike)*np.sin(dip)**2*np.sin(MODE1)) - np.cos(AZM)*np.cos(TKO)*(np.cos(MODE1)*(np.sin(2*dip)*np.sin(rake)*np.sin(strike)**2 + np.sin(2*strike)*np.cos(rake)*np.sin(dip)) + np.sin(MODE1)*((2*poisson)/(2*poisson - 1) - 2*np.sin(dip)**2*np.sin(strike)**2)))

            G = np.sqrt(Gsh**2 + Gsv**2)

        elif wave in ('S/P', 's/p'):
            G = self.radpat('S', )/self.radpat('P', obs_sph = obs_sph)  

        elif wave in ('P/S', 'p/s'):
            G, poubelle  = self.radpat('P', obs_sph = obs_sph)/self.radpat('S', obs_sph = obs_sph)  

        elif wave in ('SH/P', 'sh/p'):
            G, poubelle  = self.radpat('SH', obs_sph = obs_sph)/self.radpat('P', obs_sph = obs_sph)

        elif wave in ('SV/P', 'sv/p'):
            G, poubelle  = self.radpat('SV', obs_sph = obs_sph)/self.radpat('P', obs_sph = obs_sph)

        elif wave in ('SH/S', 'sh/s'):
            G, poubelle  = self.radpat('SH', obs_sph = obs_sph)/self.radpat('S', obs_sph = obs_sph)

        elif wave in ('SV/S', 'sv/s'):
            G, poubelle  = self.radpat('SV', obs_sph = obs_sph)/self.radpat('S', obs_sph = obs_sph)

        ## Making sure you get that error.
        else:
            print 'Can t yet compute this wave type.'

        ## transform G into vector x,y,z          
        G_cart = spherical_to_cartesian(np.asarray([AZM, TKO, G]))
        obs_cart = spherical_to_cartesian(np.asarray([AZM, TKO]))

        #return G, [AZM, TKO] 
        return np.asarray(G_cart), np.asarray(obs_cart)


    def plot(self, wave='P',style='*', ax=None, cbarxlabel=None) :
        """
        Plot the radiation pattern.
        ______________________________________________________________________
        :type self : object: Vavryeuk
        :param self : This method use de self.mt attribute, i.e. the focal 
            mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the six 
            independent components of the moment tensor)

        :type wave: String
        :param wave: type of wave to compute

        :type style : string
        :param style : type of plot.

        :rtype : graphical object
        :return : The axe3d including plot.

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.

            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

        .. rubric:: _`Supported style`

            ``'*'``
                Plot options q, s and p together.

            ``'b'``
                Plot a unit sphere, amplitude sign being coded with color.
                 (uses :func:`numpy.abs`).

            ``'q'``
                This a comet plot, displacements magnitudes, directions and 
                 final state are given, respectively by the colors, lines and 
                 points (as if looking at comets).  

            ``'s'``
                Plot the amplitudes as surface coding absolute amplitude as 
                 radial distance from origin and amplitude polarity with color.
                 (uses :func:`numpy.abs`).

            ``'f'``
                Plot the amplitudes as a wireframe, coding absolute amplitude as 
                 radial distance. The displacement polarities are not 
                 represented.

            ``'p'``
                Plot the polarities sectors using a colorcoded wireframe. 
                 The magnitudes of displacements are not represented.

        .. note::

            The radiation pattern is rendered as is if no comp option is given.

            In the title of the plot, SDR stand for strike, dip, rake angles 
            and IDC stand for isotropic, double couple, compensated linear 
            vector dipole percentages.

        .. seealso::

                :func: `plot_seismicsourcemodel`
                :class: `Vavryeuk.radpat`
        ______________________________________________________________________

        """

        # Get radiation pattern and plot
        G, XYZ = self.radpat(wave)
        return plot_seismicsourcemodel(G, XYZ, style=style, mt=self.mt, comp='L',ax=ax, cbarxlabel=cbarxlabel)

    def energy(self, wave='P') :   
        """
        Get statistical properties of radiation pattern.
        ______________________________________________________________________
        :type self : object: Vavryeuk
        :param self : This method use de self.mt attribute, i.e. the focal 
            mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the six 
            independent components of the moment tensor)

        :type wave: string
        :param wave: type of wave to compute

        :rtype : list
        :return : The statistical properties of amplitudes [rms, euclidian norm, average].

        .. rubric:: _`Supported wave`

            ``'P'``
                Bodywave P.

            ``'S' or 'S wave' or 'S-wave``
                Bodywave S.

            ``'Sv'``
                Projection of S on the vertical.

            ``'Sh'``
                Projection of S on the parallels of the focal sphere. 

        .. seealso::

                :func: `energy_seismicsourcemodel`
                :class: `Vavryeuk.radpat`
        ______________________________________________________________________

        """  
        # Get radiation pattern and estimate energy
        G, XYZ = self.radpat(wave)  
        estimators = energy_seismicsourcemodel(G, XYZ)
        return estimators



class SeismicSource(object):
    """
    Set an instance of class SeismicSource() that can be used for 
    earthquake modeling.
    ______________________________________________________________________
    :type mt : list
    :param mt : Definition of the focal mechanism supported 
        obspy.imaging.scripts.mopad.MomentTensor().

    :type poisson: variable
    :param poisson: Poisson coefficient (0.25 by default).

    .. example::

        ex = NnK.source.SeismicSource([1,2,3,4,5,6])
        ex.Aki_Richards.plot('P')

    .. note::

        This object is composed of three classes : MomentTensor, 
        Aki_Richards and Vavryeuk.

    .. seealso::

            :class: `obspy.imaging.scripts.mopad.MomentTensor`
            :class: `Aki_Richards`
            :class: `Vavryeuk`            
    ______________________________________________________________________

    """  
    # General attribut definition
    notes = 'Ceci est à moi'

    def __init__(self, mt, poisson=0.25):
		self.MomentTensor = MomentTensor(mt,debug=2)
		self.Aki_Richards = Aki_Richards(np.asarray(self.MomentTensor.get_M('XYZ')))  
		self.Vavryeuk     = Vavryeuk(np.asarray(self.MomentTensor.get_M('XYZ')),poisson = poisson)  


def degrade(wavelets, shift = [-.1, .1], snr = [.5, 5.]):
    
    shift = np.random.random_integers(int(shift[0]*1000), int(shift[1]*1000), len(wavelets))/1000. # np.linspace(shift[0], shift[1], wavelets.shape[0])
    snr = np.random.random_integers(int(snr[0]*1000), int(snr[1]*1000), len(wavelets))/1000. # np.linspace(snr[0], snr[1], wavelets.shape[0])
    
    for i,w in enumerate(wavelets):
        w = w.data
        tmp = np.max(abs(w)).copy()
        wavelets[i].data += (np.random.random_integers(-100,100,len(w))/100.) * np.max(w)/snr[i]
        wavelets[i].data /= tmp #*np.max(abs(self.wavelets[i]))
        
        tmp = np.zeros( len(w)*3. )
        index = [ int(len(w)+(shift[i]*len(w))), 0 ]
        index[1] = index[0]+len(w)
        tmp[index[0]:index[1]] = wavelets[i].data[:int(np.diff(index)) ]
        wavelets[i].data[-len(w):] = tmp[-2*len(w):-len(w)]
    
    return wavelets

def plot_wavelet(bodywavelet, style = '*', ax=None):
    
    ## Initializing the plot
    if ax == None:
         ax = (plt.figure(figsize=plt.figaspect(1.))).gca(projection='3d')
                
    axins = inset_axes(ax,
                   width="60%", # width = 30% of parent_bbox
                   height="30%", 
                   loc=3)


    if hasattr(bodywavelet, 'SeismicSource'):
        bodywavelet.SeismicSource.Aki_Richards.plot(wave='P', style='s', ax=ax)
    
    wavelets = []
    ws = [] 
    lmax = 0
    for i,w in enumerate(bodywavelet.Stream):
        #print bodywavelet.Stream[i].data.shape
        wavelets.append(w.data)
        ws.append( np.sum( w.data[:int(len(w.data)/2.)]) )
        lmax = np.max([ lmax , len(w.data) ])
        axins.plot(w.data+i/100.)
    
    axins.set_xlim([0 , lmax])
    axins.set_xlabel('Time')
    axins.set_ylabel('Amplitude')
    
    wavelets = np.ascontiguousarray(wavelets)
    ws = np.ascontiguousarray(ws)
 
    ax.scatter(bodywavelet.observations['cart'][0][ws>=0.],
                bodywavelet.observations['cart'][1][ws>=0.],
                bodywavelet.observations['cart'][2][ws>=0.],
                marker='+', c='b')
    ax.scatter(bodywavelet.observations['cart'][0][ws<0.],
                bodywavelet.observations['cart'][1][ws<0.],
                bodywavelet.observations['cart'][2][ws<0.],
                marker='o', c= 'r')
    
    for i in range(bodywavelet.observations['n']):
        ax.text(bodywavelet.observations['cart'][0][i],
                 bodywavelet.observations['cart'][1][i],
                 bodywavelet.observations['cart'][2][i],
                 '%s' % (str(i)))
        
    

    ax.view_init(15,65)

class ArticialWavelets(object):
    """
        Set an instance of class ArticialWavelets() that can be used to get
        artificial wavelets.
        ______________________________________________________________________
        
        .. note::
        
        This object is composed of two methods : get and plot.
        ______________________________________________________________________
    """
    def __init__(self, n_wavelet= 50, az_max = 8*np.pi):
        
        self.mt = [-1,-1,-1]
        self.n_wavelet = n_wavelet
        self.az_max = az_max
        
        # Generates
        signs = np.ones([self.n_wavelet,1])
        signs[self.n_wavelet/2:] *=-1
        self.wavelets = np.tile([0.,1.,0.,-1.,0.], (self.n_wavelet, 1)) * np.tile(signs, (1, 5))
        self.obs_cart = np.asarray(globe(n=n_wavelet))
        self.obs_sph = np.asarray(cartesian_to_spherical(self.obs_cart))
    
    def get(self):
        
        return self.wavelets, self.obs_sph, self.obs_cart

    def degrade(self, shift = [-.1, .1], snr = [.5, 5.]):

        self.wavelets = degrade(self.wavelets, shift,snr)
    
    def plot(self, style = '*'):
        
        plot_wavelet(self, style)


class SyntheticWavelets(object):
    """
        Set an instance of class SyntheticWavelets() that can be used to get
        synthetic wavelets.
        ______________________________________________________________________
        :type self : object: 
        :param self : 
        
            self.Stream
            self.observations {'n_wavelet'
                               'mt'
                               'types'
                               'cart'
                               'sph'
             
            self.MomentTensor : optional, 
            self.SeismicSource : optional,  
            
        
        .. note::
        
        This object is composed of two methods : get and plot.
        
        .. TODO::
        put all of this in a obspy stream
        ______________________________________________________________________
        """
    def __init__(self, n = 50, mt=None):
                
        if mt == None :
            mt = [np.random.uniform(0,360) ,np.random.uniform(-90,90),np.random.uniform(0,180)]
            
        # Gets the seismic source model
        self.SeismicSource = SeismicSource(mt)
        self.MomentTensor = MomentTensor(mt,debug=2)

        # 
        self.observations = {'types': np.tile('P', (n, 1)),
                             'cart' : np.asarray(globe(n=n)),
                             'sph'  : np.asarray(cartesian_to_spherical(np.asarray(globe(n=n)))),
                             'mt'   : mt,
                             'n'    : n }
        
        # Generates template wavelets #####################
        t = np.linspace(.0, 1., 20, endpoint=False)
        wave = np.sin(2 * np.pi * t )  
        wavelets = np.tile(wave, (n , 1)) 
        
        ## radiation pattern at given angles
        source_model = SeismicSource( mt )
        disp, observations_xyz = source_model.Aki_Richards.radpat(wave='P', obs_sph = self.observations['sph'] )                   
        #disp = farfield( np.ravel(self.MomentTensor.get_M())[[0,4,8,1,2,5]] , self.observations['cart'] , 'P')
        amplitudes, disp_projected = disp_component(self.observations['cart'], disp, 'r')

        ## Apply modeled amplitudes to artificial wavelets
        wavelets *= np.swapaxes(np.tile(np.sign(amplitudes),(wavelets.shape[1],1)),0,1)
        
        #
        self.Stream = Stream()        
        for i in range(n):
            self.Stream.append(Trace(data=wavelets[i,:] , 
                                     header=Stats({'network' : "SY",
                                                   'station' : str(i),
                                                   'location': "00",
                                                   'channel' : "EHL",   # in origin center sphere: L: Radius, T: Parallel, Q: Meridian
                                                   'npts'    : len(t),
                                                   'delta'   : 1/((t[-1]-t[0])/len(t)) }) ))        
        
    def get(self):
        
        return self
    
    def degrade(self, shift = [-.1, .1], snr = [.5, 5.]):
        
        self.__init__(self.n_wavelet, self.mt)

        self.Stream = degrade( self.Stream , shift,snr)
                
            
    def plot(self, style = '*'):
         
        plot_wavelet( self, style)


class BodyWavelets(object):
    """
        Set an instance of class SyntheticWavelets() that can be used to get
        synthetic wavelets.
        ______________________________________________________________________
    
        .. note::
        
        This object is composed of two methods : get and plot.
        ______________________________________________________________________
        """
    def __init__(self, f= 100., starts=-0.05, ends=0.2, n_wavelet= 50, sds= "/Users/massin/Documents/Data/ANT/sds/", catalog = "/Users/massin/Desktop/arrivals-hybrid.Id-Net-Sta-Type-W_aprio-To-D-Az-Inc-hh-mm-t_obs-tt_obs-tt_calc-t_calc-res-W_apost-F_peak-F_median-A_max-A_unit-Id-Ot-Md-Lat-Lon-Dep-Ex-Ey-Ez-RMS-N_P-N_S-D_min-Gap-Ap-score_S-score_M", sacs= "/Users/massin/Documents/Data/WY/dtec/2010/01/21/20100121060160WY/", nll = "//Users/massin/Documents/Data/WY/dtec/2010/01/21/20100121060160WY/20100121060160WY.UUSS.inp.loc.nlloc/WY_.20100121.061622.grid0.loc.hyp", mode="sds-column"):

        from obspy.core.stream import read

        self.n_wavelet = n_wavelet
        self.az_max = []
        self.mt = []
        self.wavelets = np.asarray([])
        self.obs_sph = np.asarray([])
        npoints = f*(ends - starts)-1
        
        
        if mode == "sds-column" :
            
            from obspy.core.utcdatetime import UTCDateTime
            from obspy.clients.filesystem.sds import Client

            client = Client(sds)

            names=('Networks','Stations','Waves', 'weights','Origin time', 'Distances', 'Azimuth', 'Takeoff', 'Hours', 'Minuts', 'Seconds', 'Magnitudes', 'Id')
            columns=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 0)
            metadata = np.genfromtxt(catalog, usecols=columns, comments="Id", names=names, dtype=None)

            for arrival,md in enumerate(metadata):
                if metadata[arrival][5] > 1. and self.wavelets.shape[0] < n_wavelet :
                    
                    time = "%17.3f" % metadata[arrival][4]
                    time = UTCDateTime(time[:8]+'T'+time[8:])
                    time += (metadata[arrival][8]-time.hour)*60*60 + \
                        (metadata[arrival][9]-time.minute)*60 + \
                        (metadata[arrival][10]-(time.second+time.microsecond/1000000.))
                    
                    st = client.get_waveforms(metadata[arrival][0], metadata[arrival][1], "*", "*[Z]", time+2*starts, time+2*ends) #
                    #st.trim(time+starts, time+ends)
                    st.detrend()
                    st.normalize()
                    st.interpolate(f, starttime=time+starts, npts=npoints)
                    for t,Trace in enumerate(st):
                        if self.wavelets.shape[0] < n_wavelet :
                            
                            data = np.zeros((1,npoints))
                            data[:] = Trace.data[:npoints]
                            angles = np.zeros((1,3))
                            angles[:] = ([ np.deg2rad(metadata[arrival][6]), np.deg2rad(metadata[arrival][7]), 1.0]) # metadata[arrival][5]])
                            if  self.wavelets.shape[0] == 0 :
                                self.wavelets = data
                                self.obs_sph = angles
                            else :
                                self.wavelets = np.concatenate((self.wavelets, data), axis=0)
                                self.obs_sph =  np.concatenate((self.obs_sph, angles), axis=0)

        elif mode == "sac-nll" :
            
            from obspy import read_events
            import glob
            cat = read_events(nll)
            
            for p,pick in enumerate(cat[0].picks):
                if self.wavelets.shape[0] < n_wavelet :
                    #print pick.waveform_id, pick.phase_hint, pick.time,
                    for a,arrival in enumerate(cat[0].origins[0].arrivals):
                        if arrival.pick_id == pick.resource_id:
                            #print arrival.azimuth, arrival.takeoff_angle, arrival.distance
                            seismic_waveform_file = sacs+"*"+pick.waveform_id.station_code+"*Z*.sac.linux"
                            if len(glob.glob(seismic_waveform_file))>0:
                                st = read(seismic_waveform_file)
                                st.detrend()
                                st.normalize()
                                st.interpolate(f, starttime=pick.time+starts, npts=npoints)
                                for t,Trace in enumerate(st):
                                    if self.wavelets.shape[0] < n_wavelet :
                                        
                                        data = np.zeros((1,npoints))
                                        data[:] = Trace.data[:npoints]/np.max(abs(Trace.data[:npoints]))
                                        angles = np.zeros((1,3))
                                        angles[:] = ([ np.deg2rad(arrival.azimuth), np.deg2rad(arrival.takeoff_angle), 1.0]) # metadata[arrival][5]])
                                        if  self.wavelets.shape[0] == 0 :
                                            self.wavelets = data
                                            self.obs_sph = angles
                                        else :
                                            self.wavelets = np.concatenate((self.wavelets, data), axis=0)
                                            self.obs_sph =  np.concatenate((self.obs_sph, angles), axis=0)
                                            
        self.obs_sph = (self.obs_sph.T)
        self.obs_cart = np.asarray( spherical_to_cartesian(self.obs_sph))
        self.n_wavelet = self.wavelets.shape[0]

    def get(self):
        
        return self.wavelets, self.obs_sph, self.obs_cart

    def degrade(self, shift = [-.1, .1], snr = [.5, 5.]):
        
        self.__init__(self.n_wavelet, self.az_max,self.mt)
        self.wavelets = degrade(self.wavelets, shift,snr)

    def plot(self, style = '*'):
        
        plot_wavelet(self, style)



class RoughDCScan(object):

    '''
        .. To do : 
            [x] use obspy stream
            [ ] use obspy.core.event.source.farfield  | obspy.taup.taup.getTravelTimes
            [ ] linear scan
            [ ] separate pdf plot (richer)
    '''
    
    def __init__(self, n_model = 20):
        '''
            Sets model space
        '''
        
        # Prepares
        self.scanned = 0
        
        ## Wave types
        self.waves = ['P', 'S']
        self.components = [['L'], ['T', 'Q'] ] # in origin center sphere: L: Radius, T: Parallel, Q: Meridian

        ## Precision
        radiation_pattern_angular_precision = 99 # just a factor, not actual number of points (to be improved ?)

        ## DC exploration step
        self.n_model = n_model
        strikes = np.linspace(0,180,self.n_model**(1/3.)+1)
        dips = np.linspace(0,90,self.n_model**(1/3.))
        slips = np.linspace(-180,180,self.n_model**(1/3.)+1)
        
        ## Sources (any convention of MoPad can be used)
        source_mechanisms = np.asarray(  np.meshgrid(strikes, dips, slips)  )*1.  #, sparse=True
        source_mechanisms = source_mechanisms.transpose( np.roll(range(len(source_mechanisms.shape)),-1) )
        N = np.prod(source_mechanisms.shape[:-1])
        flat2coordinate = np.asarray( np.unravel_index( range(N), source_mechanisms.shape[:-1] ) )

        ## Initial grids for modeling
        ### Observations (in trigo convention)
        self.atr = cartesian_to_spherical(globe(n=500)) 
        observations_atr_nparray = np.asarray(self.atr)
#         # test plot ##################################################
#         ax = (plt.figure()).gca(projection='3d')                     #
#         ax.scatter(globe(n=500)[0], globe(n=500)[1], globe(n=500)[2])#
#         ##############################################################
        
        ### Machinery
        self.modeled_amplitudes = {}
        self.source_mechanisms = {'Mo':[], 'P-axis':[], 'T-axis':[], 'rms':[], 'xcorr':[]}

        ## Loops over models #
        for i in range(N):   #            
            source_model = SeismicSource( source_mechanisms[ tuple(flat2coordinate[:,i]) ])
            
            sm = tuple(source_mechanisms[ tuple(flat2coordinate[:,i]) ])
            self.source_mechanisms[    'Mo'].append(sm)
            self.source_mechanisms['P-axis'].append( cartesian_to_spherical(source_model.MomentTensor.get_p_axis()))
            self.source_mechanisms['T-axis'].append( cartesian_to_spherical(source_model.MomentTensor.get_t_axis()))
            self.source_mechanisms[   'rms'].append(0.0)
            self.source_mechanisms[ 'xcorr'].append(0.0)
            
            ### Loops over wave types ##########
            for wi,w in enumerate(self.waves): #
                displacement_xyz, observations_xyz = source_model.Aki_Richards.radpat(wave=w, obs_sph = self.atr)
            
                #### Loops over components of waves #########
                for ci,c in enumerate(self.components[wi]): #                  
                    self.modeled_amplitudes[ sm, w, c ], disp_projected = disp_component(observations_xyz, displacement_xyz, c) 
                    
#                     # tests amplitudes values #####################################################
#                     ax, cbar = source_model.Aki_Richards.plot(style='*', insert_title='('+c+' compo)', wave=w) 
#                     plus = self.modeled_amplitudes[ sm, w, c ]>0
#                     minus = self.modeled_amplitudes[ sm, w, c ]<=0
#                     ax.plot(displacement_xyz[0][plus], 
#                             displacement_xyz[1][plus], 
#                             displacement_xyz[2][plus], '+b')   #
#                     ax.plot(displacement_xyz[0][minus], 
#                             displacement_xyz[1][minus], 
#                             displacement_xyz[2][minus], 'or')   #
#                     if w=='S' and c=='m':
#                         return #######################################################################
            
#                    # re-indexes each obs point by sph. coordinate in rad ###############
#                     for line in range(observations_atr_nparray.shape[1]):              #
#                         #for col in range(observations_atr_nparray.shape[2]):          #
#                         obs_atr = tuple(observations_atr_nparray[:,line])              #
#                         self.modeled_amplitudes[ obs_atr, sm, w, c ] = amplitudes[line]#
#                         #print '[',obs_atr, sm, w, c,']=',amplitudes[line]##############
 
        print i+1,"models generated for:",
        print np.prod(np.asarray(self.atr[0]).shape), "observation points",
        print wi+1,"waves",
        print ci+1,"components."

        #for key, value in self.modeled_amplitudes.iteritems():
        #   print key, value
        
    def _data_init(self, data):
        '''
            Sets data indexes in model space                 
        '''        
        
        # Initiate ######################
        self.data = data #
        self.data_wavelets = []         #
        self.data_taperwindows = []     #
        #################################
        
        # Makes array with data with corresponding taper ########
        for i,w in enumerate(data.Stream):                      #
            # stores wavelet ####################################
            self.data_wavelets.append(w.data)    #
            # prepare taper #####################################
            taper = 2.*(1+(np.sort(range(len(w.data)))[::-1]))  #
            taper[taper>len(w.data)] = len(w.data)              #
            taper /= len(w.data)                                #
            # stores taper ######################################
            self.data_taperwindows.append(taper)                #
        #########################################################
        
        # Machinery #####################################################
        self.data_wavelets = np.asarray(self.data_wavelets)             #  
        self.data_amplitudes = np.asarray(self.data_wavelets)*0.        #
        #################################################################

        # get observation indexes corresponding to model space ##
        self.data_indexes = np.zeros([ self.data_wavelets.shape[0], len(self.atr[0].shape) ], dtype=np.int32)
        for i in range(len(data.Stream)):                       #
            distances = np.sqrt( (data.observations['sph'][0,i]-self.atr[0])**2. + (data.observations['sph'][1,i]-self.atr[1])**2. + (data.observations['sph'][2,i]-self.atr[2])**2. )
            d = np.argmin(distances)
            self.data_indexes[i,:] = np.unravel_index( d, self.atr[0].shape)
            if np.rad2deg(np.min( distances )) > 10: 
                print "Warning: unreliable amplitude modeling for station", data.Stream[i].stats.station 
        ##########################################################
        
        # search optimal stack #########
        opt_stack = np.zeros((9999))   #
        lm = 0                         #
        ## loops over trace ##############################
        for w,wavelet in enumerate(self.data_wavelets):  #
            l = len(wavelet)                             #
            lm = np.max([lm, l])                         #
            if np.sum((opt_stack[:l]+wavelet*self.data_taperwindows[w]*-1)**2.) > np.sum((opt_stack[:l]+wavelet*self.data_taperwindows[w])**2.):
                opt_stack[:l] += wavelet*self.data_taperwindows[w]*-1 
            else:                                                 
                opt_stack[:l] += wavelet*self.data_taperwindows[w]
        ##################################################
        
        # stores optimal stack #############################
        self.data_optimal_stack = opt_stack[:lm]           # Warning: optimal stack is tapered !!!!
        self.power_optimal_stack = np.sum((opt_stack)**2.) #
        ####################################################
        
#         # test plot ################################
#         ax = (plt.figure()).gca()                  #
#         ax.plot(np.transpose(self.data_wavelets))  #
#         ax.plot(self.data_optimal_stack)           #
#         ############################################

    def scan(self, data=SyntheticWavelets(mt=None)):
        '''
            Explore the model space with data.
            
            data.Stream
            data.observations {'n_wavelet'
                               'mt'
                               'types'
                               'cart'
                               'sph'
             
            data.MomentTensor : optional, 
            data.SeismicSource : optional, 
        '''
        
        #print data.observations['mt']
        # Get : ########################
        #   - optimal data stack       #
        #   - model's indexes for data #
        self._data_init(data)          #
        ################################

        
        
        # Scans source ### how to make it linear ? ###########
        for i,Mo in enumerate(self.source_mechanisms['Mo']): #
            
            # gets pdf value #################################
            for j,wavelet in enumerate(self.data_wavelets):
                self.data_amplitudes[j][:] = self.modeled_amplitudes[ Mo, 
                                                                      data.observations['types'][j,0], 
                                                                      data.Stream[j].stats.channel[-1] ][ tuple(self.data_indexes[j]) ]
    
            corrected_wavelets = np.sign(self.data_amplitudes) * self.data_wavelets * self.data_taperwindows
            stack_wavelets = np.sum(corrected_wavelets, axis=0)
            self.source_mechanisms['rms'][i] = np.sum(stack_wavelets**2)*np.sign(np.sum(stack_wavelets)) / self.power_optimal_stack
            ##################################################
        
        # Gets brightest cell
        self.best_likelyhood = [self.source_mechanisms['Mo'][np.argmax(self.source_mechanisms['rms'])], np.max(self.source_mechanisms['rms']) ]

        # Important
        self.scanned = 1
    
    def corrected_data(self, mt, data):
        
        results = copy.deepcopy(data)
        
        results.SeismicSource = SeismicSource(mt)
        results.MomentTensor = MomentTensor(mt,debug=2)        
        results.observations['mt'] = mt
        
        amplitudes={}
        source_model = SeismicSource( mt )
        
        ### Loops over wave types ##########
        for wi,w in enumerate(self.waves): #            
            #disp = farfield( np.ravel(results.MomentTensor.get_M())[[0,4,8,1,2,5]] , results.observations['cart'] , w)
            disp, observations_xyz = source_model.Aki_Richards.radpat(wave=w, obs_sph = results.observations['sph'] )        
            #### Loops over components of waves #########
            for ci,c in enumerate(self.components[wi]): #                  
                amplitudes[ w, c ], disp_projected = disp_component(results.observations['cart'], disp, c)
        
        for i in range(len(results.Stream)):              
            results.Stream[i].data *= np.sign(amplitudes[
                                                         results.observations['types'][i,0], 
                                                         results.Stream[i].stats.channel[-1] ][i])
                
        return results
    
    def plot(self, scanned=1, data=SyntheticWavelets(mt=None), sol = None, style = '*'):
                
        if self.scanned == 0 or scanned == 0 :
            self.scan(data)

        # Plots
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax1 = plt.subplot2grid((1,2), (0, 0), projection='3d')
        ax2 = plt.subplot2grid((1,2), (0, 1), projection='3d')
        
        ## plots data
        plot_wavelet(self.data, style, ax=ax1)
        
        ## plots best result
        self.best_likelyhood[0]
        plot_wavelet( self.corrected_data(self.best_likelyhood[0], self.data) , style, ax=ax2)
        
    
    def PT_pdf(self):
        '''
            Produces pdfs
        '''
        
        # pdf grid for P-T axis 
        test = np.meshgrid( np.linspace(-np.pi,np.pi,50) , np.linspace(0.,np.pi,20) ) 
        self.pdf = {'azim':test[0], 
                    'incl':test[1], 
                    'rad': np.ones(test[1].shape), 
                    'P/T axis': np.zeros([test[1].shape[0], test[1].shape[1], 2])}
        i_to_field = ['azim','incl','rad']
        
        ## Machinery                                         
        coef_mem = np.zeros(self.pdf['P/T axis'].shape)
        
        # Scans source ### how to make it linear ? ###########
        for i,Mo in enumerate(self.source_mechanisms['Mo']): #
            # makes smooth pdf along P-T space ################################
            coef = np.zeros(self.pdf['P/T axis'].shape)
            for n in range(len(self.source_mechanisms['P-axis'][i])): ###################
                coef[:,:, 0] += ( self.source_mechanisms['P-axis'][i][n] - self.pdf[i_to_field[n]] )**2. #
                coef[:,:, 1] += ( self.source_mechanisms['T-axis'][i][n] - self.pdf[i_to_field[n]] )**2. #
#             coef = np.sqrt(coef)
#             while len(coef[abs(coef)>np.pi**2]):
#                 coef[coef>np.pi**2] -= np.pi**2
#                 coef[coef<-1*np.pi**2] += np.pi**2
#             coef = abs(coef)   
            coef[coef<.00001] = .00001 
            coef = 1./coef
            self.pdf['P/T axis'] += self.source_mechanisms['rms'][i] * coef                                    #    
            coef_mem += coef                                                  #
        #######################################################################
        
        # Smoothes ###################
        self.pdf['P/T axis'] /= coef_mem #
        ##############################


         
        # create figure, add axes
        ax = (plt.figure(figsize=plt.figaspect(1.))).gca()
     
        # make orthographic basemap.
        m = Basemap(resolution='c',projection='ortho',lat_0=90.,lon_0=0.)
        
        # define parallels and meridians to draw.
        parallels = np.arange(0.,90.,10.)
        meridians = np.arange(0.,360.,20.)
      
        ## draw coastlines, parallels, meridians.
        m.drawparallels(parallels)
        m.drawmeridians(meridians)#  
        
#         # transform to nx x ny regularly spaced 5km native projection grid
#         nx = int((m.xmax-m.xmin)/5000.)+1; ny = int((m.ymax-m.ymin)/5000.)+1
#         topodat = m.transform_scalar(self.pdf['P/T axis'][:,:,0],np.rad2deg(self.pdf['azim'][0,:]),np.rad2deg(self.pdf['incl'][:,0]),100,50)
#         # plot image over map with imshow.
#         im = m.imshow(topodat,plt.cm.RdBu_r)
   
   
        ## compute native x,y coordinates of grid.
        x, y = m(np.rad2deg(self.pdf['azim']),np.rad2deg(self.pdf['incl']))
        
        colormaps = [plt.cm.OrRd, plt.cm.Blues ]
        colorlines = ['r', 'b']
        for i in range(self.pdf['P/T axis'].shape[2]):                
            ## set desired contour levels.
            clevs = np.linspace(-1.,1.,10) #
            
            ## plot SLP contours.
            CS1 = m.contour(x,y,self.pdf['P/T axis'][:,:,i],clevs,linewidths=0.5,colors=colorlines[i],animated=True)
            CS2 = m.contourf(x,y,self.pdf['P/T axis'][:,:,i],clevs,cmap=colormaps[i],animated=True, alpha=.5)
        

 
        
#         # 3d Plots
#         fig = plt.figure(figsize=plt.figaspect(.5))
#         ax1 = plt.subplot2grid((1,2), (0, 0), projection='3d', aspect='equal', ylim=[-1,1], xlim=[-1,1], zlim=[-1,1], title='P. axis.', xlabel='Lon.', ylabel='Lat.')
#         ax2 = plt.subplot2grid((1,2), (0, 1), projection='3d', aspect='equal', ylim=[-1,1], xlim=[-1,1], zlim=[-1,1], title='T. axis.', xlabel='Lon.', ylabel='Lat.')
#          
#         ## Pressure
#         XYZ = np.asarray(spherical_to_cartesian([np.pi/2. - self.pdf['azim'] - np.pi, self.pdf['incl'], self.pdf['rad']]))
#         G = np.asarray(spherical_to_cartesian([self.pdf['azim'], self.pdf['incl'], self.pdf['rad']*self.pdf['P/T axis'][:,:,0] ]))
#         plot_seismicsourcemodel(G, XYZ, comp='r', style='s', ax=ax1, cbarxlabel='P-axis likelyhood', alpha=1.)
#   
#         ## Tension
#         G = np.asarray(spherical_to_cartesian([self.pdf['azim'], self.pdf['incl'], self.pdf['rad']*self.pdf['P/T axis'][:,:,1] ]))
#         plot_seismicsourcemodel(G, XYZ, comp='r', style='s', ax=ax2, cbarxlabel='T-axis likelyhood', alpha=1.)
#           
#         view = (15,65)
#         ax2.view_init(view[0],view[1])
#         ax1.view_init(view[0],view[1])
        
    def centroid(self):
        '''
            Centroid solutions
        '''
        
        ## weights
        self.SDSl_centroid_serie = []
        for t,threshold in enumerate(np.linspace(0.05,.95,50)):
            
            ### weigths with likelyhood difference
            w = self.SDSl[:,-1].copy()
            ### rejects a percentile of max likelyhood
            test = np.sort(w)
            test = test[ int(len(test)*threshold) ]   #test = test[ np.argmin(abs( np.cumsum(test)-np.sum(test)*.9 )) ] # centroid 95%
            w[w<test] = 0. # every val below the median of positive val are cancelled
            ### weigths with distance to max likelyhood
            d = np.sqrt(np.sum((self.SDSl[:,:3].copy() - np.tile(self.SDSl_likely ,(self.SDSl.shape[0],1)))**2., axis=1))
            while len(d[d>180.])>0:
                d[d>180.] = d[d>180.]-180.
            d[d==0.] = .000000001
            d /= np.max(abs(d))
            ### rejects antipode to max likelyhood
            d[d>(1.-threshold)] = 1. # every val below the median of positive val are cancelled
            w /= d
            w -= np.min(w)
            ## centroid
            ### stacks centroid
            SDSl_centroid = np.nansum(self.SDSl[:,:3]*np.transpose(np.tile(w, (3,1))),axis=0)/np.nansum(w)
            SDSl_centroid = np.append(SDSl_centroid, threshold)
            ### add power
            example = SeismicSource(SDSl_centroid)
            disp, xyz = example.Aki_Richards.radpat(wave='P')
            modeled_amplitudes , disp_projected = disp_component(xyz, disp, 'r')
            SDSl_centroid = np.append(SDSl_centroid, self.stack(self.data, modeled_amplitudes))
            
            ### stores centroid
            print int(SDSl_centroid[0]),int(SDSl_centroid[1]), int(SDSl_centroid[2])
            self.SDSl_centroid_serie.append(SDSl_centroid)
            self.SDSl_centroid = np.append(SDSl_centroid, self.stack(self.data, modeled_amplitudes))
        
        self.SDSl_centroid_serie = np.asarray(self.SDSl_centroid_serie)
        
        #print "Scanner results"
        #print "Centroid:", self.SDSl_centroid
        #print "Max. likelyhood:", self.SDSl_localmax

    
    
    def demo(self, scanned=1, data=SyntheticWavelets(n=10), sol = None):

        view = (15,65)
        n = 5
        data.degrade(snr=[5.,10.], shift = [0.,0.])
        if self.scanned == 0 or scanned == 0 :
            self.scan(data)
        
        if sol == None:
            sol = self.SDSl_centroid
        
        changes = [-90., 180., -90.]
        
        fig = plt.figure(figsize=plt.figaspect(.95))
        ax_data = plt.subplot2grid((2,4), (1,0), ylim=[-1,(data.n_wavelet)/100.+1.], xlim=[0,data.wavelets.shape[1]-1], xlabel='Time (sample).', ylabel='Amplitude.', title='Data.')
        ax_data.plot(np.transpose(data.wavelets) * np.tile(range(data.n_wavelet,data.n_wavelet*2), (data.wavelets.shape[1], 1))/20.);
        
        ax_model = plt.subplot2grid((2,4), (0,0), projection='3d', aspect='equal', ylim=[-1,1], xlim=[-1,1], zlim=[-1,1])
        ax_model.set_title('Data coverage and polarities')
        ws = np.sum(data.wavelets[:,:int(data.wavelets.shape[1]/2.)], axis=1)
        ax_model.scatter(data.obs_cart[0][ws>=0],data.obs_cart[1][ws>=0],data.obs_cart[2][ws>=0],c='b')
        ax_model.scatter(data.obs_cart[0][ws<0],data.obs_cart[1][ws<0],data.obs_cart[2][ws<0],c='r')
        
        ax_model.tick_params(
                             axis='both',          # changes apply to the x-axis
                             which='both',      # both major and minor ticks are affected
                             right='off',      # ticks along the bottom edge are off
                             left='off',         # ticks along the top edge are off
                             top='off',      # ticks along the bottom edge are off
                             bottom='off',         # ticks along the top edge are off
                             labelbottom='off',
                             labelleft='off')
        ax_model.view_init(view[0],view[1])

        for i,change in enumerate(changes):
            
            ax_model = plt.subplot2grid((2,4), (0, i+1), projection='3d', aspect='equal', ylim=[-1,1], xlim=[-1,1], zlim=[-1,1])
            ax_data = plt.subplot2grid((2,4), (1, i+1), ylim=[-1,(data.n_wavelet)/100.+1.], xlim=[0,data.wavelets.shape[1]-1])
            sol[0] += change
            if sol[0]<0:
                sol[0] +=360
            elif sol[0]>360:
                sol[0] -= 360
            if i == 2 :
                example = data.source
            else:
                example = SeismicSource(sol[:3])
            
            ax_model, cbar = example.Aki_Richards.plot(wave='P', style='s', ax=ax_model, cbarxlabel='Solution amplitudes', cb=0)
            disp, xyz = example.Aki_Richards.radpat(wave='P', obs_sph = data.obs_sph)
            amplitudes, disp_projected = disp_component(xyz, disp, 'r')
            ax_model.scatter(data.obs_cart[0][amplitudes>=0],data.obs_cart[1][amplitudes>=0],data.obs_cart[2][amplitudes>=0],c='b')
            ax_model.scatter(data.obs_cart[0][amplitudes<0],data.obs_cart[1][amplitudes<0],data.obs_cart[2][amplitudes<0],c='r')
            
            ax_model.view_init(view[0],view[1])
            ax_model.set_title('Trial '+str(i+1))
            
            if i==1:
                ax_data.set_xlabel('Time (sample).')
                ax_data.set_title('Data with correction.')
            
            ax_data.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            right='off',      # ticks along the bottom edge are off
                            left='off',         # ticks along the top edge are off
                            labelleft='off')
            ax_model.tick_params(
                                axis='both',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                right='off',      # ticks along the bottom edge are off
                                left='off',         # ticks along the top edge are off
                                top='off',      # ticks along the bottom edge are off
                                bottom='off',         # ticks along the top edge are off
                                labelbottom='off',
                                labelleft='off')
            
            corrected_wavelets = np.swapaxes(np.tile(np.sign(amplitudes),(data.wavelets.shape[1],1)),0,1) * data.wavelets
            ax_data.plot(np.transpose(corrected_wavelets) * np.tile(range(data.n_wavelet,data.n_wavelet*2), (corrected_wavelets.shape[1], 1))/20.);



# function skeleton
# def ggg(...):
#         """
#         Run a ...
#
#         :param type: String that specifies ... (e.g. ``'skeleton'``).
#         :param options: Necessary keyword arguments for the respective
#             type.
#
#         .. note::
#
#             The raw data is/isn't accessible anymore afterwards.
#
#         .. rubric:: _`Supported type`
#
#         ``'skeleton'``
#             Computes the ... (uses 
#               :func:`dependency.core.Class1.meth1`).
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





        
