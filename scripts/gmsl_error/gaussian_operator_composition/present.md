# Error quantification

## Test

I varied:

- ice_gmsl_target_std from 0.0005 to 0.5 with 8 values
- ice_length_scale from 0.05 to 0.7 mean sea floor radius with 4 values
- net_ice_thickness_change from 0 to 150 m with 8 values
- odt_length_scale from 0.001 to 0.1 mean sea floor radius with 3 values
- odt_standard_deviation from 0.0008 to 0.08 with 4 values
- altimetry_error_length_scale from 0.0005 to 0.05 mean sea floor radius with 3 values
- altimetry_error_amplitude from 0.0003 to 0.03 with 3 values

## Results

Testing for just linear interactions resulted in:

![alt text](observed_vs_predicted_error_mean.png)
![alt text](observed_vs_predicted_error_std.png)

With the following models:

```
Linear regression results for error_mean:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             error_mean   R-squared:                       0.606
Model:                            OLS   Adj. R-squared:                  0.606
Method:                 Least Squares   F-statistic:                 4.775e+04
Date:                Fri, 21 Nov 2025   Prob (F-statistic):               0.00
Time:                        10:12:29   Log-Likelihood:             5.2200e+05
No. Observations:              248832   AIC:                        -1.044e+06
Df Residuals:                  248823   BIC:                        -1.044e+06
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                           -0.1615      0.000   -446.802      0.000      -0.162      -0.161
ice_gmsl_target_std           -3.01e-15      0.000  -8.06e-12      1.000      -0.001       0.001
net_ice_thickness_change         0.0004    1.2e-06    366.223      0.000       0.000       0.000
odt_standard_deviation        3.722e-15      0.002   1.97e-12      1.000      -0.004       0.004
altimetry_error_amplitude     2.765e-15      0.004   6.23e-13      1.000      -0.009       0.009
altimetry_range                  0.0023   4.63e-06    497.871      0.000       0.002       0.002
ice_length_scale             -1.139e-17   3.69e-11  -3.09e-07      1.000   -7.23e-11    7.23e-11
odt_length_scale              -6.79e-19   2.09e-10  -3.25e-09      1.000    -4.1e-10     4.1e-10
altimetry_error_length_scale -8.338e-20   4.18e-10  -1.99e-10      1.000    -8.2e-10     8.2e-10
==============================================================================
Omnibus:                    13598.271   Durbin-Watson:                   0.966
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            17537.783
Skew:                          -0.536   Prob(JB):                         0.00
Kurtosis:                       3.737   Cond. No.                     2.11e+08
==============================================================================

```

```
Linear regression results for error_std:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              error_std   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 4.534e+08
Date:                Fri, 21 Nov 2025   Prob (F-statistic):               0.00
Time:                        10:12:29   Log-Likelihood:             1.2128e+06
No. Observations:              248832   AIC:                        -2.426e+06
Df Residuals:                  248823   BIC:                        -2.425e+06
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                            0.0051   2.25e-05    224.831      0.000       0.005       0.005
ice_gmsl_target_std              1.4001   2.32e-05   6.02e+04      0.000       1.400       1.400
net_ice_thickness_change      2.151e-14   7.44e-08   2.89e-07      1.000   -1.46e-07    1.46e-07
odt_standard_deviation           0.0035      0.000     29.457      0.000       0.003       0.004
altimetry_error_amplitude        0.0457      0.000    165.379      0.000       0.045       0.046
altimetry_range              -7.287e-05   2.88e-07   -252.955      0.000   -7.34e-05   -7.23e-05
ice_length_scale              4.046e-11    2.3e-12     17.608      0.000     3.6e-11     4.5e-11
odt_length_scale              6.817e-11    1.3e-11      5.234      0.000    4.26e-11    9.37e-11
altimetry_error_length_scale  2.277e-10   2.61e-11      8.742      0.000    1.77e-10    2.79e-10
==============================================================================
Omnibus:                    47418.951   Durbin-Watson:                   0.755
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           157721.887
Skew:                           0.962   Prob(JB):                         0.00
Kurtosis:                       6.392   Cond. No.                     2.11e+08
==============================================================================
```

These show that the model for error_std is fully described by the relationship:

$$
\sigma_{error} \propto \sigma_{ice\_gmsl\_target}
$$

while the model for error_mean is mostly described by (R²=0.606):
$$
\mu_{error} \propto \overline{\Delta ice\_thickness} + range\_altimetry
$$

The plots for the error_mean model looked like secondary interaction, and as such tested with interaction terms included:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             error_mean   R-squared:                       0.968
Model:                            OLS   Adj. R-squared:                  0.968
Method:                 Least Squares   F-statistic:                 1.247e+06
Date:                Fri, 21 Nov 2025   Prob (F-statistic):               0.00
Time:                        10:45:21   Log-Likelihood:             8.3376e+05
No. Observations:              248832   AIC:                        -1.668e+06
Df Residuals:                  248825   BIC:                        -1.667e+06
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
============================================================================================================
                                               coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                 1.421e-15      0.000   1.03e-11      1.000      -0.000       0.000
net_ice_thickness_change                    -0.0027   1.89e-06  -1414.413      0.000      -0.003      -0.003
altimetry_range                           -3.62e-17   1.91e-06   -1.9e-11      1.000   -3.74e-06    3.74e-06
net_ice_thickness_change:altimetry_range  4.439e-05   2.65e-08   1673.389      0.000    4.43e-05    4.44e-05
ice_gmsl_target_std                       6.706e-17      0.000   6.29e-13      1.000      -0.000       0.000
odt_standard_deviation                    2.502e-16      0.001   4.64e-13      1.000      -0.001       0.001
altimetry_error_amplitude                -1.298e-17      0.001  -1.02e-14      1.000      -0.002       0.002
==============================================================================
Omnibus:                    26235.648   Durbin-Watson:                   0.952
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           110553.147
Skew:                          -0.460   Prob(JB):                         0.00
Kurtosis:                       6.133   Cond. No.                     3.82e+05
==============================================================================
```

These show that including the interaction term between net_ice_thickness_change and altimetry_range results in a much better model (R²=0.968):
$$
\mu_{error} \propto \overline{\Delta ice\_thickness} \times range\_altimetry + \overline{\Delta ice\_thickness}
$$

Which resulted in this fit:

![alt text](observed_vs_predicted_error_mean_with_interactions.png)

And this predicted error field:

![alt text](predicted_error_mean_vs_ice_thickness_change_and_altimetry_range.png)

Which matches the true error field well:

![alt text](gmsl_error_over_ice_change_and_altimetry_range.png)
