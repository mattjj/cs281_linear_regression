from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

from regression import PolyRegression, PolyRegressionInferNoise

def interactive(deg=4,lmbda=0.,sigmasq=1.):
    plt.axes()
    plt.xlim([-10,10])
    plt.ylim([-10,10])

    a = PolyRegression(deg=deg,mu_w=zeros(deg+1),J_w=lmbda*eye(deg+1))
    t = np.linspace(-10,10,100)

    points = []
    while True:
        newpoint = plt.ginput()
        if newpoint == []:
            return a
        points += newpoint
        x,y = points[-1]

        plt.cla()
        plt.plot([p[0] for p in points],[p[1] for p in points],'kx')

        if len(points) >= deg+1 or lmbda != 0:
            if len(points) == deg+1 and lmbda == 0:
                a.condition_on(array([p[0] for p in points]),array([p[1] for p in points]),
                        sigmasq*np.eye(len(points)))
            elif len(points) > deg+1 or lmbda != 0:
                a.condition_on(y,x,sigmasq*np.eye(1))

            predictions, errors = a.predict(t,sigmasq)

            plt.plot(t,predictions,'b-')
            plt.plot(t,predictions+np.sqrt(errors),'b--')
            plt.plot(t,predictions-np.sqrt(errors),'b--')
            for i in range(3):
                predictions, _ = a.predict_sample(t,0.25)
                plt.plot(t,predictions,'r-',alpha=0.5)

        plt.xlim([-10,10])
        plt.ylim([-10,10])


def model_selection(q=lambda x: (x+4)*(x+1)*(x-2)*(x-7)/10,
        degrees=range(10),lmbda=25,sigma=100.,N=20):

    # generate the data
    xs = np.linspace(-10,10,N)
    ys = q(xs) + sigma*randn(N)

    # plot it
    plt.figure()
    plt.plot(xs,ys,'ro')
    plt.plot(xs,q(xs),'b-')

    models = \
        [PolyRegression(deg=d,J_w=lmbda*eye(d+1))
                for d in degrees]

    marg_likes = [m.marginal_likelihood(ys,xs,Sigma_e=sigma**2*np.eye(N)) for m in models]

    plt.figure()
    plt.plot(degrees,marg_likes)

    return marg_likes


def infernoise_interactive(deg=4,lmbda=0.):
    plt.axes()
    plt.xlim([-10,10])
    plt.ylim([-10,10])

    a = PolyRegressionInferNoise(w_0=zeros(deg+1),V_0=eye(deg+1),a_0=5,b_0=1.)
    t = np.linspace(-10,10,100)

    points = []
    while True:
        newpoint = plt.ginput()
        if newpoint == []:
            return a
        points += newpoint
        x,y = points[-1]

        plt.cla()
        plt.plot([p[0] for p in points],[p[1] for p in points],'kx')

        a.condition_on(y,x)

        predictions, errors = a.predict(t)

        plt.plot(t,predictions,'b-')
        plt.plot(t,predictions+np.sqrt(errors),'b--')
        plt.plot(t,predictions-np.sqrt(errors),'b--')
        for i in range(3):
            predictions, _ = a.predict_sample(t)
            plt.plot(t,predictions,'r-',alpha=0.5)

        plt.xlim([-10,10])
        plt.ylim([-10,10])

