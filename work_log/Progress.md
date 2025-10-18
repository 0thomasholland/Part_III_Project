

# Plan for the project in outline

- Implementing the traditional method(s) for estimating sea level change from satellite altimetry observations, and investigating its accuracy. For global mean sea level, this method is just to spatially average sea surface height changes over the oceans.
  - Roughly first 2-4 weeks 
- Application of Bayesian inversion methods that incorporate sea level physics. First done for estimates at a single time. Comparing new methods against old.
  - Roughly second half of Michaelmas term
- Extension to consider time-dependent estimates.
  - Running over Christmas break
  - Consider building in feed forwards mechanisms (Kalman filter-like)
- Possible extension to consider:
  - Adding ice altimetry data and/or other data types
  - Effect of knocking out %’s of the networks (e.g. esp tidal guages → look at their average downtime)
  - Comparison of the data types (more data vs more types of data)
  - Consider if different data types have different abilities to resolve features (e.g. tidal gauges trade off within East and West Antarctic Ice … does this exist within altimetry data?)
  - Early Lent

# Weekly progress/goals


## Meeting 1: 13/Oct

- Start reading through the literature esp. focusing on:
  - Horton 2018 for general overview 
  - Lickley 2018 for sea level to surface
  - The Reciprocity Paper (TRP) §4.6 (and then the references within this)
- Look at implementing equivalent of eq. 81 from TRP to convert the SL change from fingerprint to SSH to then average this over the ocean for GMSL change


## Meeting 2: 17/Oct

- Discussion of scientific method
- Explore the error space for conventional methods 
  - Error for each ice sheet over satellite availability bands
  - Ternary space for major ice sheet contributions at satellite availability bands
- Future explore adding:
  - Noise based on satellite sampling
  - Ocean dynamic signals
- Writing useful scripts/functions that can be reused in the future
