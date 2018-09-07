import numpy as np
import matplotlib.pyplot as plt

from vkikriging.kriging_v2 import Kriging, GEK
from vkikriging.sampling import halton, random_uniform

### 1. Example test-function (Genz function 1 - oscillatory)
dimension = 8
def f1(xi):
    return np.cos(np.pi + np.sum(9./5 * xi))
def df1(xi):
    return (-np.sin(np.pi + np.sum(9./5 * xi)) *
            np.ones(dimension) * 9./5)

### 2. Reconstruction with varying number of samples
k_mses, gek_mses = [], []
nsamples = [10,20,50,100,200,500,1000]

for nsample in nsamples:
    print(f'nsample = {nsample}')
    # 2a. Choose samples (pseudo-random) on [0,1]^d, and
    #     evaluate function and derivatives
    xi_samples = halton(nsample, dimension)
    y = np.array([f1(xi) for xi in xi_samples]) 
    dy = np.array([df1(xi) for xi in xi_samples])

    # 2b. Build the Kriging/GEK models
    kmodel = Kriging(xi_samples, y, sigma_y=0.01,
                     mu_x=0, sd_x=1, gamma=0.3)
    gekmodel = GEK(xi_samples, y, dy, sigma_y=0.01,
                   sigma_dy=0.01, mu_x=0, sd_x=1, gamma=0.3)
    
    # 2c. Use Kriging models to predict function at 1000
    #     random points
    xi_predict = random_uniform(1000, dimension)
    y_exact = np.array([f1(xi) for xi in xi_predict])
    k_predict = kmodel.predict(xi_predict, posterior_cov=None)
    gek_predict, _ = gekmodel.separate(
        gekmodel.predict(xi_predict, posterior_cov=None))

    # 2d. Compute mean-squared error of predition
    k_mses += [np.sum((k_predict - y_exact)**2) / nsample]
    gek_mses += [np.sum((gek_predict - y_exact)**2) / nsample]

### 3. Plot the convergence
plt.loglog(nsamples, k_mses, '-+b', label='Kriging')
plt.loglog(nsamples, gek_mses, '-+r', label='GEK')
plt.xlabel('number of samples')
plt.ylabel('mean-squared error')
plt.legend()
plt.show()
