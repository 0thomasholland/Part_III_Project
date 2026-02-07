# work

Directory where my work files / science occurs.

`major sources` = GIS/WAIS/EAIS

## folder structure

Folders are structured in terms of "questions":

- 00 - all ice field error: `with uniform/uniform distribution of ice thickness change across ice fields what is the error?`
- 01 - major sources: `how does the error vary across different ice sheets?`
- 02 - ice load bands: `how does changing the load latitude vs altimetry sampling latitude affect the error?`
- 03 - major source mixing: `how does mixing different major sources affect the error?`
- 04 - signal effects: `how do other signals (e.g. ODT, signal noise, etc) affect the error associated?`
- 05 - bayesian inversions: `what are bayesian inversions?`
- 06 - altimetry sampling: `how can altimeter data be sampled to generate a point field?`
- 07 - inversion: `how can we use baysian inversion with altimetry data`
- XX - time data: `how can time series be used to improve error estimates?`
- XX - other data: `can we use other data sources to improve accuracy?`

## order of work

Deterministic:
- 00 - all ice field error
- 01 - major sources in scalar fields
- 03 - looking at mixing of major sources

Gaussian framework:
- 00 - gaussian all ice field error
- [ ] 01 - major sources in gaussian fields
- [ ] 04 - adding in other signals (ODT, etc)
- 05 - learning about inversions using pygeoinf
- 06 - satellite altimetry sampling
- 07 - implementing inversions
