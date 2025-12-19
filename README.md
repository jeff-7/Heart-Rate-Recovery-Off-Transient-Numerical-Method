## OFF-Transient Heart-Rate Fitting (Paper-Faithful)

This document describes a numerical fitting pipeline for heart-rate recovery data that reproduces the parameter estimation method of the reference paper. The analysis focuses exclusively on OFF-transient segments for visualization and performance evaluation.

The primary reported parameter is the per-trial demand parameter **D**, estimated using a global least-squares fit across all trials.

---

### Method Summary

Heart-rate data from multiple trials are preprocessed using a Fourier low-pass filter consistent with the reference methodology. A global least-squares optimization is performed across all trials, with shared model parameters and a per-trial demand parameter D.

Although both ON- and OFF-transient data are used during fitting, only OFF-transient segments are visualized and evaluated to characterize recovery dynamics.

---

### Input Data

The analysis requires an Excel workbook named *Heart Rate-Time.xlsx*.  
Each worksheet corresponds to one trial and must include:

- Time in seconds (`delta_t_s`)
- Heart rate in beats per minute (`bpm`)

Trial sheets are named using the convention *on-transient(i)* and *off-transient(i)*.

---

### Outputs

The script produces a multi-panel figure displaying raw heart-rate data, Fourier-filtered signals, and model predictions for each OFF-transient trial.

Fit quality is quantified using:
- L_data (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of determination (R²)

Metrics are reported per trial, pooled across all OFF-transient data points, and averaged across trials.

---

### Interpretation Notes

The reported demand parameters D are directly comparable across trials, as they are obtained using a single global fitting procedure. Pooled metrics reflect overall predictive performance, while mean per-trial metrics indicate consistency across experimental repetitions.

Internal model parameters are intentionally omitted from the output.

---

### Usage

The analysis is executed from the project directory using a single Python command. All outputs are generated automatically.
