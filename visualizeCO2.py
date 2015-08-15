"""
        Subroutines for visualizing CO2 simulation results
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from display_notebook import display_animation

def getImgParam(param):
    # plot setting for different resolution
    resolution = param.resolution
    param = {       'nx': 59,
                    'ny': 55,
                    'x_well1': 18,
                    'x_well2': 43,
    }
    if resolution is 'low':
        param['nx'] = 59
        param['ny'] = 55
        param['x_well1'] = 18
        param['x_well2'] = 43
    elif resolution is 'medium':
        param['nx'] = 117
        param['ny'] = 109
        param['x_well1'] = 34
        param['x_well2'] = 84
    elif resolution is 'high':
        param['nx'] = 234
        param['ny'] = 217
        param['x_well1'] = 67
        param['x_well2'] = 167
    else:
        print "resolution must be low, medium or high"

    return param

def plotCO2vid(x_list,param):
    # Create a video of maps stored in list x
    def vec2map(x):
        x_map = np.reshape(x,(param['ny'],param['nx']), order = 'F')
        return x_map

    fig = plt.figure()
        
    ims = []
    for i in range(len(x_list)):
        x = vec2map(x_list[i])
        im = plt.imshow(x,cmap=plt.get_cmap('jet'))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
        repeat_delay=1000)
    ani.save('dynamic_images.mp4')
    plt.show()

def plotCO2map(x,cov,param):
    # Display a CO2 map
    # x: list of maps
    # param: settings of plots
    # figure()
    f, (ax1, ax2) = plt.subplots(1,2)
    def vec2map(x):
        x_map = np.reshape(x,(param['ny'],param['nx']), order = 'F')
        x_map = x_map[:,param['x_well1']:param['x_well2']]
        return x_map
    x_map = vec2map(x[-1])
    cov_map = vec2map(cov[-1])
    im1 = ax1.imshow(x_map,interpolation = 'nearest')
    f.colorbar(im1, ax = ax1)
    im1.set_clim(-0.15, 0.62)
    ax1.set_title('Mean')
    im2 = ax2.imshow(cov_map, interpolation = 'nearest')
    f.colorbar(im2, ax = ax2)
    ax2.set_title('Variance')
    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
# cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])# horizontal colorbar
    # plt.colorbar()

def data2timeseries(y, ind):
    # y: list of nparray stores data for each time step
    # ind: the index of sensor pair/raypath whose measurement to be extracted
    # measurement: output is a time-series of measurement observed at ind
    nt = len(y)
    measurement = np.zeros(nt)
    for i in range(nt):
      measurement[i] = y[i][ind]

    return measurement

def plotCO2data(y,ind1,ind2):
    nt = len(y)
    time = range(nt)
    data1 = data2timeseries(y,ind1)
    data2 = data2timeseries(y,ind2)
    f, (ax1, ax2) = plt.subplots(1,2, sharex=True)
    # ax1 = plt.subplot(121)
    ax1.plot(time,data1)
    ax1.set_title('CO2 not on the raypath')
    ax1.set_ylim([-0.5, 2.5])
    ax1.set_xlim([0,40])
    # ax2 = plt.subplot(122)
    ax2.plot(time,data2)
    ax2.set_ylim([-0.5, 2.5])
    ax2.set_xlim([0,40])
    ax2.set_title('CO2 on the raypath')

def plotCO2_data_map(x,y,ind1,ind2,param):
    """
    ** input **
    x: numpy.array(m,1)
        current state

    y: list (nt,1)
        list of measurement until current state
    """
    # Survey geometry
    n_source        =   6
    n_receiver      =   48
    x_source        =   np.zeros((n_source,1))
    y_source        =   np.linspace(1655,1675,n_source)
    z_source        =   np.zeros((n_source,1))
    x_receiver      =   30*np.ones((n_receiver,1))                              
    y_receiver      =   np.linspace(1630,1680,n_receiver) 
    z_receiver      =   np.zeros((n_receiver,1))
    
    # 1. x_map
    x_map = np.reshape(x,(param['ny'],param['nx']), order = 'F')
    x_map = x_map[:,param['x_well1']:param['x_well2']]
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    im = ax1.imshow(x_map,interpolation = 'nearest',extent=[0,30,1686.9,1619.9])
    fig.colorbar(im, ax = ax1)
    im.set_clim(0.0, 0.7)
    ax1.plot([x_source[0],x_receiver[ind1]],[y_source[0],y_receiver[ind1]],color = 'y')
    ax1.plot([x_source[0],x_receiver[ind2]],[y_source[0],y_receiver[ind2]],color = 'g')
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Depth(m)')
    ax1.set_title('CO2 map')
    ax1.set_xlim([0,30])
    ax1.set_ylim([1686.9,1619.9])


    nt = len(y)
    time = np.arange(0,nt*3,3)
    data1 = data2timeseries(y,ind1)
    data2 = data2timeseries(y,ind2)
    # f, (ax1, ax2) = plt.subplots(1,2, sharex=True)
    # ax1 = plt.subplot(121)
    ax2.plot(time,data1,color = 'y')
    ax2.set_title('Travel time 1', color = 'y')

    ax2.set_xlabel('Time (hours)')
    ax2.set_ylim([-0.5, 7])
    ax2.set_xlim([0,120])
    # ax2 = plt.subplot(122)
    ax3.plot(time,data2,color = 'g')
    ax3.set_ylim([-0.5, 7])
    ax3.set_xlim([0,120])
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Travel time 2',color = 'g')

def scale_barplot():
    """
    Bar chart demo with pairs of bars grouped for easy comparison.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    n_groups = 3
    
    runtime_kf = np.array([1.2, 19, 4.4*60])*60
    
    runtime_hikf = np.array([0.14, 0.57, 2.25])*60
    
    storage_kf = (25, 32, 34, 20, 25)
    storage_hikf = (3, 5, 2, 3, 3)
    
    fig, ax = plt.subplots()
    
    # index = np.arange(n_groups)
    index = np.log10(np.array([1, 4, 16]))
    bar_width = 0.20
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
       
    rects1 = plt.bar(index, runtime_kf, bar_width,
                     alpha=opacity,
                     color='b',
    #                  yerr=std_men,
                     error_kw=error_config,
                     label='KF')
    
    rects2 = plt.bar(index + bar_width, runtime_hikf, bar_width,
                     alpha=opacity,
                     color='r',
    #                  yerr=std_women,
                     error_kw=error_config,
                     label='HiKF')
    
    x1, x2 = index[0]+0.1, index[2]+0.1
    y1 = np.log10(runtime_kf[0]/2)
    y2 = y1 + 2*(x2 - x1)
    plt.plot([x1,x2],[10**y1,10**y2], color = 'b', label = 'O($N^2$)')
    
    x1, x2 = index[0]+0.3, index[2]+0.3
    y1 = np.log10(runtime_hikf[0]/2)
    y2 = y1 + (x2 - x1)
    plt.plot([x1,x2],[10**y1,10**y2], color = 'r', label = 'O($N$)')

    ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.xlabel('Resolution/N')
    plt.ylabel('Runtime (s)')
    # plt.ylim([0, 5])
    plt.title('Log-log plot of runtime')
    plt.xticks(index + bar_width, ('low', 'medium', 'high'))
    
    # Change location of the legend
    plt.legend(bbox_to_anchor=(0.3, 1))
               # bbox_transform=plt.gcf().transFigure)
    
    # Add texts on top of last bar
    def autolabel(rect,text):
        # attach some text labels
        # for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%s'%(text),
                    ha='center', va='bottom')
    
    autolabel(rects1[0],'1.2 min')
    autolabel(rects1[1],'19 min')
    autolabel(rects1[2],'4.4 hour')
    autolabel(rects2[2],'2.25 min')
    autolabel(rects2[1],'0.57 min')
    autolabel(rects2[0],'0.14 min')
    
    plt.tight_layout()
    plt.show()
    