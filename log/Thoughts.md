# Thoughts log

## 2025-10-10 22:01

To what extent can the noise/artifacts that were filtered out by @coulson2022 be build into the output to see what the "real world" observable would be?

## 2025-10-17

For parallel processing, using joblib's Parallel and delayed is much simpler than multiprocessing.Pool. Just wrap the function call in delayed() and pass to Parallel.

Tried using ternary-plot but found mpternary worked much better.

## 2025-10-18

Build a load across multiple latitudes and then plot this against error at different satellite segments to see how error varies with load and latitude.

## 2025-10-26

Using ternary plots works well but it is important to provide contextual side plots that show a selection from the ternary location in actual space.

For example for melt contribution plots show a side plot of what each ternary space plot looks like on a map.

Similarly having bands of satalite latitude ranges showing what is being sampled by different plots can help contextualise the differences seen in distribution ternary plots.

## 2025-11-10 - Error exploration notes

- Variance shift, seems to decrease with increasing lmax till a point ~ =300
- This is reporducable with minimal code (ie just load field and solve SLQ)
- Mean values seem to form linear trend with lmax in log log space
- Similar is seen with mean movement error
- Variance shift seems to be sensible with trend towards a sensible number
- However dimensionalisation is causing issues with values being plotted
