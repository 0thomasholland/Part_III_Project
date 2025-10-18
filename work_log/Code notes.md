# Code notes


## 2025-10-17

For parallel processing, using joblib's Parallel and delayed is much simpler than multiprocessing.Pool. Just wrap the function call in delayed() and pass to Parallel.

Tried using ternary-plot but found mpternary worked much better.

