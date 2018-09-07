from vkikriging.test_functions import Parabola
from vkikriging.sampling import sobol
from vkikriging.example2d import Example2d

# 1. Initialize the function to approxiate, samples
#    and Kriging paramters
f = Parabola()
xi = sobol(4, 2) * (f.xmax - f.xmin) + f.xmin
ex2d = Example2d(f, xi, gamma=1., sigma_d=0.01,
				 sigma_dg=0.01)

# 2. Build the Kriging model, v1, v2 or v3.
#    For v1, v2 GEK model is also built.
ex2d.build_surrogate_v1()

# 3. Plot the model
ex2d.plot_contours()
