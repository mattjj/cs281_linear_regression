# CS281: Linear Regression #

Run this code with `ipython --pylab` or use `from pylab import *`.

## Fitting an Affine Transformation in R^2 ##

```python
In [1]: A = randn(2,2)

In [2]: b = randn(2)

In [3]: X = randn(20,2)

In [4]: y = A.dot(X.T).T + b + randn(20,2)/5.

In [6]: solve(X.T.dot(X),X.T.dot(y)).T
Out[6]:
array([[-0.33567275,  1.44876796,  2.20848249],
       [-1.6441768 ,  0.46691108,  1.07798395]])

In [8]: A
Out[8]:
array([[-0.43447047,  1.46280608],
       [-1.62196681,  0.46863636]])

In [9]: b
Out[9]: array([ 2.21737423,  1.16437595])
```

## Polynomial Regression ##

```python
In [1]: x = np.linspace(-10,10,25)

In [2]: y = 3*x**2 - 10*x + 30 + 50.*randn(25)

In [3]: plot(x,y,'ro')
Out[3]: [<matplotlib.lines.Line2D at 0x10d5e4c10>]

In [4]: from regression import PolyRegression

In [5]: m = PolyRegression(deg=4)

In [6]: m.condition_on(y,x,50*eye(25))
Out[6]: 46.8 x^0 + -5.36 x^1 + 2.79 x^2 + -0.058 x^3 + 0.00115 x^4

In [7]: t = linspace(-10,10,100)

In [8]: means, errs = m.predict(t,50*eye(100))

In [9]: plot(t,means)
Out[9]: [<matplotlib.lines.Line2D at 0x111865850>]
```

## Interactive Demos ##

See `regression_experiments.py`, e.g.

```python
In [1]: import regression_experiments as r

In [2]: r.interactive(deg=4)
```

The blue solid curve is the mean of the posterior predictive distribution, and
blue dashed curves are +/- one standard deviation. Red curves are posterior
samples.

![Polynomial regression](http://www.mit.edu/~mattjj/github/cs281_linear_regression/figure_1.png)
