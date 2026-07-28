Code and data accompanying:

> _Inferring activity from fluid flow in continuum models of active matter_ | 
> Aditya Mohapatra, Sagarika Adhikary, Rajesh Singh
> https://arxiv.org/abs/2607.20100

## `amh.py`
This simulates Active Model H for given parameter sets, extracts energy-spectrum E(k,t) from the simulated flow fields and train sa CNN to map those energy pectrum features to parameters.
Afterwards it evaluates the performance of the model on fresh held-out simulations.

## `an.py`
This simulates a model of Active Nematics for given parameter sets, extracts energy-spectrum E(k,t) from the simulated flow fields and train sa CNN to map those energy pectrum features to parameters.
Afterwards it evaluates the performance of the model on fresh held-out simulations.

## `model_selection.py`
Given a velocity field it compares and selects which model is more appropriate to the velocity field.

## `stats.ipynb`
Quantitative statistics for estimation of both Active Model H and Active Nematics.

## `model_selection_analysis.ipynb`
Quantitative statistics and visualization for model selection.
