import numpy as np
import matplotlib.pyplot as plt
from vkikriging.example1d import Example1d

# 1. Initialize the function to approxiate, samples
#    and Kriging paramters
ex1d = Example1d(np.sin, np.cos,
                 np.linspace(0, 10, 5),
                 gamma=1.0, sigma_d=0.001, sigma_dg=0.01,
                 xi_min=-5, xi_max=15)

# 2. Build the Kriging model, v1, v2 or v3.  For v1,
#    v2 GEK model is also built.
ex1d.build_surrogate_v1()

# 3. Plot the model, use any of all of the plot_* functions
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ex1d.plot_gek(ax)
ex1d.plot_posterior_samples_gek(ax)
ex1d.plot_reference(ax)
ex1d.plot_observed_gradients(ax, length=1.)
plt.show()
