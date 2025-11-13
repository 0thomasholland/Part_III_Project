# %%

from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

# %%
# import variable_input_data_initial.pkl from ./output

output_data = pd.DataFrame(
    load(
        path.join(
            path.dirname(path.abspath(__file__)),
            "output",
            "variable_input_data_initial.pkl",
        ),
    ),
)


# %%
