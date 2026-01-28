import pymc as pm
import numpy as np

with pm.Model():
    x = pm.Normal("x", 0, 1)
    y = pm.Normal("y", x, 1, observed=np.array([0.1, -0.2, 0.05, 0.3]))
    idata = pm.sample(300, tune=300, chains=2, cores=1, progressbar=False)

print("ok", float(idata.posterior["x"].mean()))
