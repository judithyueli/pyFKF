from filterpy.kalman import KalmanFilter
from CO2simulation import CO2simulation
import common
import numpy as np
import visualizeCO2 as vco2
import matplotlib.pyplot as plt
from HiKF import HiKF
from numpy import dot, zeros, eye, isscalar

def CO2_kf_filter(CO2, param):
	"""filter matrices initialization for KF"""
	try:
		theta = param.theta
		Qparam = param.Qparam
	except AttributeError, e:
		raise e

	x_dim = CO2.x_dim
	z_dim = CO2.H_mtx.shape[0]
	# theta = (1.14,1e-5)
	kf = KalmanFilter(x_dim, z_dim)
	kf.P = np.zeros((x_dim,x_dim))
	kf.x = np.zeros((x_dim,1))
	kf.H = CO2.H_mtx
	kf.F = np.identity(x_dim)
	kf.R = theta[1]*kf.R
	grid = CO2.grid
	# create Q
	Q = common.getGCF(grid, Qparam[0], Qparam[1])
	kf.Q = theta[0]*theta[1]*Q
	return kf
	
def CO2_hikf_filter(CO2, param):
	"""HiKF filter matrices initialization"""
	try:
		theta = param.theta
		Qparam = param.Qparam
	except AttributeError, e:
		raise e
	
	x_dim = CO2.x_dim
	z_dim = CO2.H_mtx.shape[0]
	# theta = (1.14,1e-5)
	kf = HiKF(x_dim, z_dim)
	kf.x = np.zeros((x_dim,1))
	kf.H = CO2.H_mtx
	kf.F = np.identity(x_dim)
	kf.R = theta[1]*kf.R
	grid = CO2.grid
	# create Q
	Q = common.getGCF(grid, Qparam[0], Qparam[1])
	kf.PHT = np.zeros([x_dim, z_dim])
	kf.QHT = theta[0]*theta[1]*dot(Q,kf.H.T)
	kf.var = np.zeros((x_dim,1))
	kf.sigma = theta[0]*theta[1]
	return kf

def CO2_filter(CO2,param):
	"""main loop of filtering"""
	# Initialize filter matrices
	kf, var = select_filter(param, CO2)

	data, x = [], []
	x_kf, var_kf = [], []
	
	for i in range(param.nsteps):
		#simulation
		data.append(CO2.move_and_sense())
		x.append(CO2.x)
		z = CO2.extract_data()
		z = z[-1]
		#filtering
		kf.predict()
#     import pdb;pdb.set_trace()
		kf.update(z.reshape(len(z),1)) # use reshape to avoid broadcasting
		#storing
		x_kf.append(kf.x)
		# import pdb;pdb.set_trace()
		var_kf.append(var(kf))

	x_kf = np.array(x_kf)
	var_kf = np.array(var_kf)

	return kf, x_kf, var_kf

def CO2_filter_theta(theta):
	import param
	CO2 = CO2simulation(param)
	param.theta = theta
	kf, x_kf, var_kf = CO2_filter(CO2, param)
	return kf, x_kf, var_kf

def test_filter(param):
	"""Test filter using a scalar SSM"""
	x_dim = param.x_dim
	z_dim = param.z_dim
	hikf,var = select_filter(param)
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
	return hikf.x, var(hikf)

def select_filter(param, CO2 = None):
	"""initialize filter matrices for different filter types"""
	filtertype = param.filtertype
	if filtertype is 'KF':
		if CO2 is None:
			kf = KalmanFilter(param.x_dim, param.z_dim)
		else:
			kf = CO2_kf_filter(CO2, param)
		def var(obj): return np.diag(obj.P)
	elif filtertype is 'HiKF':
		if CO2 is None:
			kf = HiKF(x_dim, z_dim)
		else:
			kf = CO2_hikf_filter(CO2, param)
		def var(obj): return obj.var
	else:
		print "The filtertype does not exist"

	return kf, var

"""runCO2simulation(CO2,nsteps,filtertype)"""

if __name__ == '__main__':
	import param
	x, var = test_filter(param)
	print "x is %.4f equal to %.4f"% (x,0.7647)
	print "var is %.4f euqal to %.4f"% (var, 0.0706)
	# CO2 experiment
	# resolution = 'low'
	# CO2 = CO2simulation(resolution)

	# x_kf, cov_kf = CO2_filter(resolution,2,'KF')
	# param = vco2.getImgParam(resolution)
	# vco2.plotCO2map(x_kf,param)
	# vco2.plotCO2map(cov_kf,param)
	# plt.show()