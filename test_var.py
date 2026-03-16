import sys
import numpy as np
sys.path.append("work/13_knockout_test")
from knockout_test import compute_gmsl_posterior, posterior_full

print("Computing GMSL posterior full:")
exp_mm, std_mm = compute_gmsl_posterior(posterior_full)
print(f"Exp: {exp_mm}, Std: {std_mm}")
