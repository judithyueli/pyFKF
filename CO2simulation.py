import scipy.io
import numpy as np
"""
	Milestones:
		[] load res.mat files
		[] create an object and output x, data
		[] def move: from x(k) to x(k+1)
		[] def sense: from x(k) to y(k)
		[] def move_and_sense: from x(k) to x(k+1) and y(k+1)
"""
class CO2simulation(object):
  def __init__(self, resolution):
    """
			resolution: low(59x55), medium(117x109), large(217x234)
	"""
	# Load Vp1.mat, Vp2.mat, Vp3.mat into
	# resolution = 'lows'
    if resolution is 'low':
      Vp_dict = scipy.io.loadmat('./data/Res1.mat')
      self.dim_x = 59*55  
    elif resolution is 'medium':
      Vp_dict = scipy.io.loadmat('./data/Res2.mat')
      self.dim_x = 117*109
    elif resolution is 'high':
      Vp_dict = scipy.io.loadmat('./data/Res3.mat')
      self.dim_x = 217*234
    else:
      print 'select resolution among low, medium and high'

#   if resolution is 'low':
#     Vp_dict = scipy.io.loadmat('./data/Res1.mat')
#     self.dim_x = 59*55
#   elif resolution is 'medium':
#     Vp_dict = scipy.io.loadmat('./data/Res2.mat')
#     self.dim_x = 117*109
# elif resolution is 'high':
#   Vp_dict = scipy.io.loadmat('./data/Res3.mat')
#   self.dim_x = 217*234
# else:
#   print 'select resolution among low, medium and high'

# x_true_array = Vp_dict['truemodel']
# x_loc_array = Vp_dict['xc']
# y_loc_array = Vp_dict['yc']
# H_mtx = Vp_dict['H']

# x_true = [] #np.array(41)
#   for i in range(41):
# 	x_true.append(x_true_array[i][0])

#   print x_true[40].shape
#   print type (x_true_array)
#   print len(x_true)
