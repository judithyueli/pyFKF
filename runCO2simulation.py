from filterpy.kalman import KalmanFilter
from CO2simulation import CO2simulation
import common
import numpy as np
import visualizeCO2 as vco2
import matplotlib.pyplot as plt
from HiKF import HiKF
from numpy import dot, zeros, eye, isscalar

def CO2_kf_filter(resolution):
	"""filter matrices initialization"""
	CO2 = CO2simulation(resolution)
	x_dim = CO2.x_dim
	z_dim = CO2.H_mtx.shape[0]
	theta = (1.14e-3,1e-5)
	kf = KalmanFilter(x_dim, z_dim)
	kf.P = np.zeros((x_dim,x_dim))
	kf.x = np.zeros((x_dim,1))
	kf.H = CO2.H_mtx
	kf.F = np.identity(x_dim)
	kf.R = theta[1]*kf.R
	grid = CO2.grid
	# create Q
	Q = common.getGCF(grid, 0.5, 900)
	kf.Q = theta[0]*theta[1]*Q
	return kf
	
def CO2_hikf_filter(resolution):
	"""HiKF filter matrices initialization"""
	CO2 = CO2simulation(resolution)
	x_dim = CO2.x_dim
	z_dim = CO2.H_mtx.shape[0]
	theta = (1.14e-3,1e-5)
	kf = HiKF(x_dim, z_dim)
	kf.x = np.zeros((x_dim,1))
	kf.H = CO2.H_mtx
	kf.F = np.identity(x_dim)
	kf.R = theta[1]*kf.R
	grid = CO2.grid
	# create Q
	Q = common.getGCF(grid, 0.5, 900)
	kf.PHT = np.zeros([x_dim, z_dim])
	kf.QHT = theta[0]*theta[1]*dot(Q,kf.H.T)
	kf.var = np.zeros((x_dim,1))
	kf.sigma = theta[0]*theta[1]
	return kf

def CO2_filter(resolution,nsteps, filtertype):
	"""main loop of filtering"""
	CO2 = CO2simulation(resolution)

	if filtertype is 'HiKF':
		kf = CO2_hikf_filter(resolution)
		def var(): return kf.var
	elif filtertype is 'KF':
		kf = CO2_kf_filter(resolution)
		def var(): return np.diag(kf.P)
	else:
		print "The filtertype does not exist"

	data, x = [], []
	x_kf, cov_kf = [], []
	
	for i in range(nsteps):
		#simulation
		data.append(CO2.move_and_sense())
		x.append(CO2.x)
		z = CO2.measurement()
		#filtering
		kf.predict()
		print var().shape
#     import pdb;pdb.set_trace()
		kf.update(z.reshape(len(z),1)) # use reshape to avoid broadcasting
		print var().shape
		#storing
		x_kf.append(kf.x)
		# import pdb;pdb.set_trace()
		cov_kf.append(var())

	x_kf = np.array(x_kf)
	cov_kf = np.array(cov_kf)
	return x_kf, cov_kf

def test_filter():
	x_dim = 1
	hikf = KalmanFilter(1,1)
	hikf.Q = 0.1*np.eye(1)
	hikf.F = np.eye(1)
	hikf.H = 5*np.eye(1)
	# hikf.QHT = Q * hikf.H
	hikf.P = 0.5*np.eye(x_dim)
	hikf.x = 0.5*np.ones(1)
	hikf.R = 2*np.eye(1)
	# hikf.sigma = Q
	hikf.var = 0.5*np.ones((x_dim,1))
	hikf.predict()
	hikf.update(4)
	return hikf.x, np.diag(hikf.P)

"""runCO2simulation(resolution,nsteps,filtertype)"""

if __name__ == '__main__':
	x, var = test_filter()
	print "x is %.4f equal to %.4f"% (x,0.7647)
	print "var is %.4f euqal to %.4f"% (var, 0.0706)
	# resolution = 'low'
	# x_kf, cov_kf = CO2_filter(resolution,2,'KF')
	# param = vco2.getImgParam(resolution)
	# vco2.plotCO2map(x_kf,param)
	# vco2.plotCO2map(cov_kf,param)
	# plt.show()