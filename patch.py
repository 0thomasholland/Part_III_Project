import re

with open('work/99_plotting/06_simple_inversion_plots.py', 'r') as f:
    content = f.read()

replacement = """x_prior = np.linspace(
    prior_expectation - 6 * prior_std_dev,
    prior_expectation + 6 * prior_std_dev,
    1000,
)

xmin = min(
    GMSL_true - 6 * posterior_std_dev,
    posterior_expectation - 6 * posterior_std_dev,
    ssh_estimation_alt - 6 * ssh_std,
)
xmax = max(
    GMSL_true + 6 * posterior_std_dev,
    posterior_expectation + 6 * posterior_std_dev,
    ssh_estimation_alt + 6 * ssh_std,
)

x_post = np.linspace(xmin, xmax, 1000)
# %%"""

content = re.sub(
r'''x_prior = np\.linspace\(.*?x_prior = np\.linspace\(.*?\)
x_post = np\.linspace\(.*?\)
# %%

xmin = min\(.*?\)
xmax = max\(.*?\)''',
replacement,
content,
flags=re.DOTALL)

# Let's try simpler replacement
