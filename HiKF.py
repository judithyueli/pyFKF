import scipy.io
import numpy as np
import sys

# from __future__ import (absolute_import, division, print_function,
												# unicode_literals)
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
		self._var = 1.             # variance
		self._QHT = zeros((dim_x,dim_z)) # uncertainty cross-covariance
		self.sigma = 0             # variance of Q

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
			self._y = z - dot(H, x)

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
			K = dot(PHT, linalg.inv(S))

			# x = x + Ky
			# predict new x with residual scaled by the kalman gain
			self._x = x + dot(K, self._y)

			# PHT = PHT - K*HPHT 
			self._PHT = PHT - dot(K,HPHT)
			# self._var = var + sum(K*PHT,2)

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
				self._x = self.x + dot(self._B, u)

				# P = FPF' + Q
				# self._P = self._alpha_sq * dot3(self._F, self._P, self._F.T) + self._Q
				self._PHT = self._PHT + self._QHT
				self._var = self._var + self.sigma
				