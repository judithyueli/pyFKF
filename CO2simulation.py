import scipy.io
import numpy as np
import sys

class CO2simulation(object):
    def __init__(self, resolution):
        """
            x:  numpy.array(x_dim,1)
                current state (CO2 slowness)
            
            count: int
                current time step (max = 40)
            
            x_dim: scalar
                state dimension
            
            resolution: string
                low(59x55), medium(117x109), large(217x234)

            grid: numpy.array(x_dim,dim)
                (x,y,z) coordinates for each grid block

        """
            # Load Vp1.mat, Vp2.mat, Vp3.mat into
            # resolution = 'lows'
        if resolution is 'low':
            Vp_dict = scipy.io.loadmat('./data/Res1.mat')
            self.x_dim = 59*55   
        elif resolution is 'medium':
            Vp_dict = scipy.io.loadmat('./data/Res2.mat')
            self.x_dim = 117*109
        elif resolution is 'high':
            Vp_dict = scipy.io.loadmat('./data/Res3.mat')
            self.x_dim = 217*234
        else:
            print 'select resolution among low, medium and high'

        # load simulation results for 41 time steps
        x_true_array = Vp_dict['truemodel']
        self.x_true = [] #np.array(41)
        for i in range(41):
          x_mtx = x_true_array[i][0]
          self.x_true.append(x_mtx.flatten('F'))

        # load grid information
        x_loc_array = Vp_dict['xc']
        y_loc_array = Vp_dict['yc']
        [X,Y] = np.meshgrid(x_loc_array, y_loc_array)
        X, Y = X.flatten(order = 'F'), Y.flatten(order = 'F')
        self.grid = np.hstack((X.reshape(len(X),1),Y.reshape(len(Y),1)))

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
