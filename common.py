from __future__ import division # no integer division
import numpy as np
from numpy import linalg as LA

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
	# Check the first argument pt
	x_dim = pt.shape[0]
	try:
		dim = pt.shape[1]
	except IndexError, e:
		dim = 1
	except:
		print "The first argument must be a numpy array"

	assert dim > 0 and dim <= 3
	assert x_dim > 0
	assert corrlength > 0
	assert power >= 0 

	# compute distance between two points
	h = np.zeros([x_dim, x_dim])
	for i in range(dim):
		# import pdb;pdb.set_trace()
		if dim == 1:
			[PT1, PT2] = np.meshgrid(pt, pt)
		else:
			[PT1, PT2] = np.meshgrid(pt[:,i], pt[:,i])
		h = h + (PT1 - PT2)**2

	h = np.sqrt(h)

	# compute the covariance matrix
	Q = np.exp(h/corrlength)**power

	return Q

def testFun():
	"""Test functions in common.py"""
	x1 = np.array([1,2,3])
	x2 = np.array([[1,2,3],
            [3,4,5]])
	p, l = 1, 0.5
	
	Q1 = getGCF(x1,p,l)
	Q2 = getQ(x1,l,p)
	Q3 = getGCF(x2,p,l)
	Q4 = getQ(x2,l,p)

	print Q1 == Q2
	print Q3 == Q4

def getQ(x,l,p):
		Q = np.zeros((x.shape[0],x.shape[0]))
		for i in range(len(x)):
			for j in range(len(x)):
				h = LA.norm(x[i]-x[j])
				Q[i][j] = np.exp(h/l)**p
		
		return Q

if __name__ == '__main__':
	testFun()
