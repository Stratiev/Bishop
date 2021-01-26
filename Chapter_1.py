import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
mu = 0
sigma = 0.5
N = 10

# Polynomial fit
def poly(x, weights):
    M = len(weights)
    value = 0
    for i in range(M):
        value += weights[i] * x**i
    return value

def error(weights, x_train, t_train):
    error = 0
    for value, target in zip(x_train, t_train):
        error += (poly(value, weights) - target)**2
    return error/2

# Generate data
x_train = np.linspace(0, 1, num=N)
t_train = np.array([])
x_test = np.linspace(0, 1, num=N)
t_test = np.array([])
xax = np.linspace(0, 1, num=1000)
for i in range(N):
    t_train = np.append(t_train, np.sin(2*np.pi*x_train[i]) + np.random.normal(mu, sigma))
    t_test = np.append(t_test, np.sin(2*np.pi*x_test[i]) + np.random.normal(mu, sigma))
E_rms_train = np.array([])
E_rms_test = np.array([])

# Fit data using a polynomial model. 
param_num = 10
for i in range(1, param_num):
    weights = np.ones(i) 
    res = minimize(error, weights, args=(x_train, t_train), tol=1e-6)
    E_rms_train = np.append(E_rms_train, np.sqrt(2 * error(res.x, x_train, t_train)/len(x_train)))
    E_rms_test = np.append(E_rms_test, np.sqrt(2 * error(res.x, x_test, t_test)/len(x_test)))
    

# Here I've picked param_num = 4 as an instructive example for plotting.
weights = np.ones(4) 
res = minimize(error, weights, args=(x_train, t_train), tol=1e-6)

# Plot the data, the fit and the function, which generated the data.
plt.scatter(x_train ,t_train, label='Train set')
plt.plot(xax, np.sin(2*np.pi*xax), 'r--', label='True Value')
plt.plot(xax, poly(xax, res.x), 'b--', label='Fitted Value')
plt.legend()
plt.xlabel('x')
plt.ylabel('t')
plt.show()

# Plot the RMS for the test and training sets
xax = np.linspace(1, param_num, num=param_num-1)
plt.plot(xax, E_rms_train, label='Train set', marker='o', markersize=10)
plt.plot(xax, E_rms_test, label='Test set', marker='o', markersize=10)
plt.legend()
plt.xlabel('M')
plt.ylabel('E_rms')
plt.show()



