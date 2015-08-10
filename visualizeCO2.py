"""
        Subroutines for visualizing CO2 simulation results
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

def getImgParam(resolution):
    # plot setting for different resolution
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

def plotCO2vid(x_list):
    # Create a video of maps stored in list x
    param = getParam('low')
    return 1

def plotCO2map(x,param):
    # Display a CO2 map
    # x: list of maps
    # param: settings of plots
    figure()
    x_map = np.reshape(x[-1],(param['ny'],param['nx']), order = 'F')
    x_map = x_map[:,param['x_well1']:param['x_well2']]
    imshow(x_map,interpolation = 'nearest')
    plt.colorbar()

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

def scale_barplot():
    """
    Bar chart demo with pairs of bars grouped for easy comparison.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    n_groups = 3
    
    runtime_kf = np.log10(np.array([1.2, 19, 4.4*60])*60)
    
    runtime_hikf = np.log10(np.array([0.14, 0.57, 2.25])*60)
    
    storage_kf = (25, 32, 34, 20, 25)
    storage_hikf = (3, 5, 2, 3, 3)
    
    fig, ax = plt.subplots()
    
    # index = np.arange(n_groups)
    index = np.log10(np.array([1, 4, 16]))
    bar_width = 0.25
    
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
    
    plt.xlabel('Resolution')
    plt.ylabel('Runtime')
    plt.ylim([0, 5])
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
    
    autolabel(rects1[2],'4.4 hour')
    autolabel(rects2[2],'2.25 min')
    
    
    plt.tight_layout()
    plt.show()
    