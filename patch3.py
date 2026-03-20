import re

with open('work/99_plotting/06_simple_inversion_plots.py', 'r') as f:
    content = f.read()

content = content.replace('FIGURES_DIR = Path("figures")', 'FIGURES_DIR = Path("figures")\nFIGURES_DIR.mkdir(parents=True, exist_ok=True)')

with open('work/99_plotting/06_simple_inversion_plots.py', 'w') as f:
    f.write(content)

