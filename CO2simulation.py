import scipy.io
import numpy as np

# class CO2simulation(object):
	# def __init__(self, resolution):
		# """
		# 	resolution: low(59x55), medium(117x109), large(217x234)
		# """
		# Load Vp1.mat, Vp2.mat, Vp3.mat into
resolution = 'low'  
if resolution is 'low':
    Vp_dict = scipy.io.loadmat('./data/Res1.mat')
elif resolution is 'medium':
	Vp_dict = scipy.io.loadmat('./data/Res2.mat')
elif resolution is 'high':
	Vp_dict = scipy.io.loadmat('./data/Res3.mat')
else:
	print 'select resolution among low, medium and high'

x_true_array = Vp_dict['truemodel']
x_loc_array = Vp_dict['xc']
y_loc_array = Vp_dict['yc']
H_mtx = Vp_dict['H']

x_true = [] #np.array(41)
for i in range(41):
	x_true.append(x_true_array[i][0])

print x_true[40].shape
print type (x_true_array)
print len(x_true)
