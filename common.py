from __future__ import division # no integer division
import numpy as np

def getGCF(pt, power, corrlength):
	"""Compute the generalized Gaussian covariance matrix of the form:
		 $$Q = exp(h/l)^p$$
		 
		** input **
		 
		pt: numpy.array(x_dim, dim)
		 		location of each grid block
		 
		power: numpy.array(1)
		 		p in the equation above.
		 		p = 1: Exponential
		 		p = 2: Gaussian

		corrlength: numpy.array(1)
				correlation length parameter
	"""
	x_dim, dim = pt.shape[0], pt.shape[1]

	assert dim > 0 and dim <= 3
	assert x_dim > 0
	assert corrlength > 0
	assert power >= 0 

	# compute distance between two points
	h = np.zeros([x_dim, x_dim])
	for i in range(2):
		# import pdb;pdb.set_trace()
		[PT1, PT2] = np.meshgrid(pt[:,i], pt[:,i])
		h = h + (PT1 - PT2)**2

	h = np.sqrt(h)

	# compute the covariance matrix
	Q = np.exp(h/corrlength)**power

	return Q



