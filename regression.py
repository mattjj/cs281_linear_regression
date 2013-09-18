from __future__ import division
import numpy as np
from numpy.linalg import solve, cholesky, det, inv
from numpy.random import randn
from copy import deepcopy

from util import solve_psd

class LinearRegression(object):
    def __init__(self,mu_w,J_w):
        self.mu_w = mu_w
        self.J_w = J_w # NOTE: J_w can be zero, improper prior!

    def condition_on(self,y,X,Sigma_e=None):
        Sigma_e = np.eye(len(y)) if Sigma_e is None else Sigma_e
        X = np.atleast_2d(X)

        J_post = self.J_w + X.T.dot(solve(Sigma_e,X))
        mu_post = self.mu_w + solve(J_post,X.T.dot(solve(Sigma_e,y - X.dot(self.mu_w))))

        self.mu_w, self.J_w = mu_post, J_post
        return self

    def predict(self,X,Sigma_e):
        return X.dot(self.mu_w), np.diag(X.dot(solve(self.J_w,X.T)) + Sigma_e)

    def sample(self):
        d = len(self.mu_w)
        cho = cholesky(self.J_w)
        return self.mu_w + solve(cho.T,np.random.randn(d))

    def predict_sample(self,X,Sigma_e):
        return X.dot(self.sample()), np.diag(Sigma_e)

    def marginal_likelihood(self,y,X,Sigma_e):
        Sigma_y = X.dot(solve(self.J_w,X.T)) + Sigma_e
        mu_y = X.dot(self.mu_w)
        # gaussian log density on y ~ N(mu_y,Sigma_y)
        d = len(mu_y)
        cho = cholesky(Sigma_y)
        return -d/2*np.log(2*np.pi) - np.log(np.diag(cho)).sum() \
                - 1/2 * (y-mu_y).dot(solve(Sigma_y,(y-mu_y)))

    def __repr__(self):
        return '%s(post. mean: %s)' % (self.__class__.__name__, self.mu_w)

class PolyRegression(LinearRegression):
    '''
    models regressions of the form
    y = a_0 + a_1 x + a_2 x^2 ... a_deg x^deg
    '''
    def __init__(self,deg,mu_w=None,J_w=None):
        self.deg = deg

        mu_w = np.zeros(deg+1) if mu_w is None else mu_w
        J_w = np.zeros((deg+1,deg+1)) if J_w is None else J_w
        super(PolyRegression,self).__init__(mu_w,J_w)

    def condition_on(self,y,xs,Sigma_e=None):
        X = self._construct_X(xs)
        return super(PolyRegression,self).condition_on(y,X,Sigma_e)

    def marginal_likelihood(self,y,xs,Sigma_e):
        X = self._construct_X(xs)
        return super(PolyRegression,self).marginal_likelihood(y,X,Sigma_e)

    def predict(self,xs,sigma_es):
        X = self._construct_X(xs)
        Sigma_e = np.diag(np.atleast_1d(sigma_es))
        return super(PolyRegression,self).predict(X,Sigma_e)

    def predict_sample(self,xs,sigma_es):
        X = self._construct_X(xs)
        Sigma_e = np.diag(np.atleast_1d(sigma_es))
        return super(PolyRegression,self).predict_sample(X,Sigma_e)

    def _construct_X(self,xs):
        return np.vstack([x**np.arange(self.deg+1) for x in np.atleast_1d(xs)])

    def __repr__(self):
        return ' + '.join('%0.3g x^%d' % (w_i,i) for i,w_i in enumerate(self.mu_w))


class LinearRegressionInferNoise(object):
    def __init__(self,w_0,V_0,a_0,b_0):
        self.w_n = w_0
        self.V_n = V_0
        self.a_n = a_0
        self.b_n = b_0

    def condition_on(self,y,X):
        # See Eqs. 7.69-7.73 in Murphy
        y, X = np.atleast_1d(y), np.atleast_2d(X)
        w_0, V_0, a_0, b_0 = self.w_n, self.V_n, self.a_n, self.b_n
        self.V_n = inv(inv(V_0) + X.T.dot(X)) # NOTE: inv is numerically terrible
        self.w_n = self.V_n.dot(solve(V_0,w_0) + X.T.dot(y))
        assert self.w_n.ndim == 1
        self.a_n = a_0 + len(y)/2
        self.b_n = b_0 + 1./2*(w_0.dot(solve(V_0,w_0)) + y.dot(y)
                - self.w_n.dot(solve(self.V_n,self.w_n)))
        return self

    def predict(self,X):
        # see Eq. 7.76  and surrounding stuff in Murphy
        X = np.atleast_2d(X)
        return X.dot(self.w_n), self.b_n / self.a_n * np.diag(1 + X.dot(self.V_n).dot(X.T))

    def sample(self):
        sigmasq = 1./np.random.gamma(self.a_n,1./self.b_n)
        Sigma = self.V_n / sigmasq
        cho = cholesky(Sigma)
        d = cho.shape[0]
        return sigmasq, self.w_n + cho.dot(np.random.randn(d))

    def predict_sample(self,X):
        sigmasq, w = self.sample()
        return X.dot(w), np.diag(X.dot(sigmasq*self.V_n).dot(X.T) + sigmasq*np.eye(X.shape[0]))

    def marginal_likelihood(self,y,X):
        raise NotImplementedError # TODO

class PolyRegressionInferNoise(LinearRegressionInferNoise):
    def __init__(self,w_0,V_0,a_0,b_0):
        self.deg = len(w_0) - 1
        self.w_n = w_0
        self.V_n = V_0
        self.a_n = a_0
        self.b_n = b_0

    def condition_on(self,y,xs):
        X = self._construct_X(xs)
        return super(PolyRegressionInferNoise,self).condition_on(y,X)

    def marginal_likelihood(self,y,xs):
        X = self._construct_X(xs)
        return super(PolyRegressionInferNoise,self).marginal_likelihood(y,X)

    def predict(self,xs):
        X = self._construct_X(xs)
        return super(PolyRegressionInferNoise,self).predict(X)

    def predict_sample(self,xs):
        X = self._construct_X(xs)
        return super(PolyRegressionInferNoise,self).predict_sample(X)

    def _construct_X(self,xs):
        return np.vstack([x**np.arange(self.deg+1) for x in np.atleast_1d(xs)])

    def __repr__(self):
        return ' + '.join('%0.3g x^%d' % (w_i,i) for i,w_i in enumerate(self.w_n))

