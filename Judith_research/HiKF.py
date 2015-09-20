# from __future__ import (absolute_import, division, print_function,
												# unicode_literals)
import scipy.io
import numpy as np
import sys

# from filterpy.common import setter, setter_1d, setter_scalar, dot3
# from filterpy.stats import multivariate_gaussian
import numpy as np
from numpy import dot, zeros, eye, isscalar
import scipy.linalg as linalg
from scipy.stats import multivariate_normal

class HiKF(object):
	"""
		Implements the HiKF filter. The filter matrices are set outside the class.

		** Attributes **

		x : numpy.array(dim_x,1)
			state estimate vector

		PHT : numpy.array(dim_x, dim_z)
			cross-covariance estimate matrix

		R : numpy.array(dim_z, dim_z)
			measurement noise matrix

		Q : numpy.array(dim_x, dim_x)
			prossess noise matrix

		H : numpy.array(dim_z, dim_x)
			measurement matrix

		F : numpy.array(dim_x, dim_x)
			built-in as an identity matrix (random walk model)

		var: numpy.array(dim_x, 1)
			state variance vector

	"""

	def __init__(self, dim_x, dim_z, dim_u=0):
		"""
			**Parameters**

			dim_x : int
				number of state variables

			dim_z : int
				number of measurement

			dim_u : int (optional)
				size of the control input
		"""
		assert dim_x > 0
		assert dim_z > 0
		assert dim_u >= 0

		self.dim_x = dim_x
		self.dim_z = dim_z
		self.dim_u = dim_u

		self._x = zeros((dim_x,1)) # state
		self._PHT = eye(dim_x)       # uncertainty cross-covariance
		self._Q = eye(dim_x)       # process uncertainty
		self._B = 0                # control transition matrix
		self._F = eye(dim_x)       # state transition matrix
		self.H = 0                 # Measurement function
		self.R = eye(dim_z)        # state uncertainty
		self._alpha_sq = 1.        # fading memory control
		self._var = np.ones((dim_x,1))      # variance
		self._QHT = zeros((dim_x,dim_z)) # uncertainty cross-covariance
		self._sigma = np.ones((dim_x,1))             # process noise variance

		# gain and residual are computed during the innovation step. We
		# save them so that in case you want to inspect them for various
		# purposes
		self._K = 0 # kalman gain
		self._y = zeros((dim_z, 1))
		self._S = np.zeros((dim_z, dim_z)) # system uncertainty in measurement space

		# identity matrix. Do not alter this.
		self._I = np.eye(dim_x)

	def update(self, z, R=None, H=None):
		"""
				 Add a new measurement (z) to HiKF. If z is None, nothing is changed.

			**Parameters**
			
			z : np.array
					measurement for this update.
			
			R : np.array, scalar, or None
					Optionally provide R to override the measurement noise for this
					one call, otherwise  self.R will be used.
		"""
		if z is None:
				return

		if R is None:
				R = self.R
		elif isscalar(R):
				R = eye(self.dim_z) * R

		# rename for readability and a tiny extra bit of speed
		if H is None:
				H = self.H
		PHT = self._PHT
		x = self._x
		var = self._var

		# y = z - Hx
		# error (residual) between measurement and prediction
		if isscalar(z):
			self._y = z - dot(H, x)
		else:
			self._y = z - dot(H, x).reshape(z.shape)

		# S = HPH' + R
		# project system uncertainty into measurement space
		HPHT = dot(H, PHT)
		S = HPHT + R

		mean = np.array(dot(H, x)).flatten()
		flatz = np.array(z).flatten()

		self.likelihood = multivariate_normal.pdf(flatz, mean, cov=S)
		self.log_likelihood = multivariate_normal.logpdf(flatz, mean, cov=S)

		# K = PHT*inv(S)
		# map system uncertainty into kalman gain
		if isscalar(S):
			K = dot(PHT, 1.0/S)
		else:
			K = dot(PHT, linalg.inv(S))

		# x = x + Ky
		# predict new x with residual scaled by the kalman gain
		self._x = x + dot(K, self._y)

		# PHT = PHT - K*HPHT
		self._PHT = PHT - dot(K,HPHT)
		if len(var) is 1:  # a scalar or an array of size (1,1)
			self._var = var - K*PHT
		else:
			self._var = var - np.sum(K*PHT,axis = 1).reshape(var.shape)

		self._S = S
		self._K = K


	def predict(self, u=0):
		""" Predict next position.
		**Parameters**
		u : np.array
				Optional control vector. If non-zero, it is multiplied by B
				to create the control input into the system.
		"""
		# x = Fx + Bu, F = I
		self._x = self.x + dot(self._B, u) #potential broadcasting error

		# P = FPF' + Q
		# self._P = self._alpha_sq * dot3(self._F, self._P, self._F.T) + self._Q
		self._PHT = self._PHT + self._QHT
		if isscalar(self._var):
			self._var = self._var + self._sigma
		else:
			self._var = self._var + self._sigma.reshape(len(self._var),1)

	@property
	def QHT(self):
		""" Process uncertainty"""
		return self._QHT

	@QHT.setter
	def QHT(self, value):
		# self._Q = setter_scalar(value, self.dim_x)
		self._QHT = value

	@property
	def PHT(self):
		""" cross covariance matrix"""
		return self._PHT


	@PHT.setter
	def PHT(self, value):
		# self._P = setter_scalar(value, self.dim_x)
		self._PHT = value

	@property
	def var(self):
		""" variance of estimated states"""
		return self._var

	@var.setter
	def var(self, value):
		self._var = value

	@property
	def F(self):
		""" state transition matrix"""
		return self._F


	@F.setter
	def F(self, value):
		# self._F = setter(value, self.dim_x, self.dim_x)
		self._F = value

	@property
	def B(self):
		""" control transition matrix"""
		return self._B


	@B.setter
	def B(self, value):
		""" control transition matrix"""
		if np.isscalar(value):
				self._B = value
		else:
				# self._B = setter (value, self.dim_x, self.dim_u)
				self._B = value


	@property
	def x(self):
		""" filter state vector."""
		return self._x


	@x.setter
	def x(self, value):
		# self._x = setter_1d(value, self.dim_x)
		self._x = value

	@property
	def sigma(self):
		""" process noise variance."""
		return self._sigma

	@sigma.setter
	def sigma(self, value):
		# self._x = setter_1d(value, self.dim_x)
		if isscalar(value):
			self._sigma = value * self._sigma
		else:
			self._sigma = value

	@property
	def K(self):
		""" Kalman gain """
		return self._K


	@property
	def y(self):
		""" measurement residual (innovation) """
		return self._y

	@property
	def S(self):
		""" system uncertainty in measurement space """
		return self._S

if __name__ == '__main__':
	hikf = HiKF(1,1)
	Q = 0.1 * np.eye(1)
	hikf.F = 0.2*np.eye(1)
	hikf.H = 5*np.eye(1)
	hikf.QHT = np.dot(Q,hikf.H)
	hikf.PHT = 0.5* hikf.H
	hikf.x = 0.5*np.ones((1,1))
	hikf.R = 2*np.eye(1)
	hikf.sigma = np.diag(Q)
	hikf.var = 0.5*np.ones((1,1))
	hikf.predict()
	hikf.update(4)
	print "x is %.4f equal to %.4f"% (hikf.x,0.7647)
	print "var is %.4f euqal to %.4f"% (hikf.var, 0.0706)
	print "PHT is %.4f equal to %.4f"% (hikf.PHT, 0.3529)

