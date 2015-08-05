import scipy.io
import numpy as np
import sys

class CO2simulation(object):
    def __init__(self, resolution):
        """
            x:
                current state (CO2 slowness)
            count:
                current time step (max = 40)
            dim_x:
                state dimension
            resolution: 
                low(59x55), medium(117x109), large(217x234)
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

        # load simulation results for 41 time steps
        x_true_array = Vp_dict['truemodel']
        self.x_true = [] #np.array(41)
        for i in range(41):
          x_mtx = x_true_array[i][0]
          self.x_true.append(x_mtx.flatten('F'))

        # load grid information
        self.x_loc_array = Vp_dict['xc']
        self.y_loc_array = Vp_dict['yc']

        # load sensor measurement operator
        self.H_mtx = Vp_dict['H']

        # initialize state
        self.count = 0;
        self.x = self.x_true[self.count]

    def move(self):
      # simulate the CO2 moves from step k to k+1
      self.count = self.count + 1
      try:
        self.x = self.x_true[self.count]
      except IndexError:
        print "Error in move(): the index %d exceeds the maximum time step" % self.count
        exit(1)

    def sense(self):
      # simulate measuring the CO2 induced travel-time delay at current step
      measurement = np.dot(self.H_mtx, self.x)
      return measurement

    def move_and_sense(self):
      # simulate CO2 move from k to k+1 and measurement at k+1
        self.move()
        return self.sense()
