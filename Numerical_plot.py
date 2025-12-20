#!/usr/bin/env python3
"""
Combine:
- D estimation EXACTLY like fit_paper_faithful.py (global fit across ALL trials, E=1 fixed)
- Off-transient ONLY figure like fit_off_transients_only.py
- Per-OFF-trial metrics (L_data=RMSE, MAE, R^2) + overall pooled + mean-of-trials
- Do NOT print A/B/C

Input:
- Heart Rate-Time.xlsx (sheets like on-transient(1), off-transient(1), ...)

Output:
- OFF_transients_fit.png
- Prints: D (off only) + metrics (off only)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ==============================
# Config (match paper script)
# ==============================
EXCEL_PATH = "Heart Rate-Time.xlsx"
FOURIER_CUTOFF_HZ = 0.05   # same as your paper-faithful run
EPS = 1e-4

# Least squares settings (match fit_paper_faithful.py)
LSQ_MAX_NFEV = 35
LSQ_LOSS = "soft_l1"
LSQ_F_SCALE = 0.1


# ==============================
# Utilities (same as paper script)
# ==============================
def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def fourier_lowpass_irregular(t: np.ndarray, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """
    EXACT same idea as fit_paper_faithful.py:
    interpolate -> FFT low-pass -> iFFT -> interpolate back.
    Uses np.interp, so output length matches input length (no early stopping).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    t0 = t.min()
    ts = t - t0

    # robust dt
    dts = np.diff(ts)
    dts = dts[dts > 0]
    dt = float(np.median(dts)) if len(dts) else 1.0

    # uniform grid spanning the full range
    tu = np.arange(ts.min(), ts.max() + dt, dt)
    xu = np.interp(tu, ts, x)

    n = len(xu)
    freqs = np.fft.rfftfreq(n, d=dt)
    X = np.fft.rfft(xu)
    X[freqs > cutoff_hz] = 0.0
    x_smooth_u = np.fft.irfft(X, n=n)

    # back to original times
    x_smooth = np.interp(ts, tu, x_smooth_u)
    return x_smooth


def hr_ode(hr, A, B, C, D):
    """
    Paper-faithful dynamics with E fixed to 1:
      d(hr)/dt = A * hr^B * (1-hr)^C * (D - hr)
    hr and D in (0,1).
    """
    hr = np.clip(hr, EPS, 1 - EPS)
    D = np.clip(D, EPS, 1 - EPS)
    return A * (hr ** B) * ((1 - hr) ** C) * (D - hr)


def simulate_segment(t, hr0, A, B, C, D, n_substeps: int = 2):
    """
    Same RK4 idea as fit_paper_faithful.py: integrate at observation times.
    """
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t, dtype=float)
    y[0] = float(np.clip(hr0, EPS, 1 - EPS))

    for i in range(len(t) - 1):
        dt_big = float(t[i + 1] - t[i])
        if dt_big <= 0:
            y[i + 1] = y[i]
            continue

        h = dt_big / n_substeps
        yi = y[i]

        for _ in range(n_substeps):
            k1 = hr_ode(yi, A, B, C, D)
            k2 = hr_ode(yi + 0.5 * h * k1, A, B, C, D)
            k3 = hr_ode(yi + 0.5 * h * k2, A, B, C, D)
            k4 = hr_ode(yi + h * k3, A, B, C, D)
            yi = yi + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            yi = float(np.clip(yi, EPS, 1 - EPS))

        y[i + 1] = yi

    return y


def pack_params(A, B, C, Ds):
    # log for positive A,B,C; logit for Ds in (0,1)
    return np.concatenate([
        np.array([np.log(A), np.log(B), np.log(C)], dtype=float),
        logit(np.array(Ds, dtype=float))
    ])

def unpack_params(theta, n_trials):
    theta = np.asarray(theta, dtype=float)
    logA, logB, logC = theta[:3]
    d_logits = theta[3:3 + n_trials]

    A = float(np.exp(logA))
    B = float(np.exp(logB))
    C = float(np.exp(logC))
    Ds = sigmoid(d_logits)
    Ds = np.clip(Ds, EPS, 1 - EPS)
    return A, B, C, Ds


# ==============================
# Data I/O + preprocessing
# ==============================
def read_trials(excel_path: str):
    xls = pd.ExcelFile(excel_path)
    trials = []
    for name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=name)
        if "delta_t_s" not in df.columns or "bpm" not in df.columns:
            raise ValueError(f"Sheet '{name}' must contain columns delta_t_s and bpm.")
        t = df["delta_t_s"].to_numpy(dtype=float)
        bpm = df["bpm"].to_numpy(dtype=float)

        idx = np.argsort(t)
        t = t[idx]
        bpm = bpm[idx]

        trials.append({"name": name, "t": t, "bpm": bpm})
    return trials


def normalize_all(trials):
    all_bpm = np.concatenate([tr["bpm"] for tr in trials])
    hr_min = float(np.min(all_bpm))
    hr_max = float(np.max(all_bpm))
    if hr_max <= hr_min:
        raise ValueError("hr_max <= hr_min; cannot normalize.")
    return hr_min, hr_max


def preprocess_trials(trials, hr_min, hr_max, cutoff_hz):
    processed = []
    for tr in trials:
        t = tr["t"]
        bpm = tr["bpm"]

        bpm_s = fourier_lowpass_irregular(t, bpm, cutoff_hz=cutoff_hz)

        hr_n = (bpm_s - hr_min) / (hr_max - hr_min)
        hr_n = np.clip(hr_n, EPS, 1 - EPS)

        ts = t - t.min()

        processed.append({
            "name": tr["name"],
            "t": ts,
            "bpm_raw": bpm,
            "bpm_smooth": bpm_s,
            "hr_n": hr_n,
        })
    return processed


def initial_guess(processed):
    Ds0 = []
    for seg in processed:
        hr = seg["hr_n"]
        k = max(3, int(0.1 * len(hr)))
        Ds0.append(float(np.median(hr[-k:])))
    Ds0 = np.clip(np.array(Ds0), 0.05, 0.95)

    A0 = 0.5
    B0 = 1.5
    C0 = 1.5
    return pack_params(A0, B0, C0, Ds0)


def build_residual(processed):
    n = len(processed)

    def residual(theta):
        A, B, C, Ds = unpack_params(theta, n)
        r = []
        for i, seg in enumerate(processed):
            t = seg["t"]
            hr = seg["hr_n"]
            D = float(Ds[i])

            y = simulate_segment(t, hr0=hr[0], A=A, B=B, C=C, D=D)
            if np.any(~np.isfinite(y)):
                r.append(np.full_like(hr, 1e3))
            else:
                r.append(y - hr)
        return np.concatenate(r)

    return residual


# ==============================
# Metrics
# ==============================
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true

    # L_data = RMSE (bpm)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return rmse, mae, r2, len(y_true)


# ==============================
# Main
# ==============================
def main():
    trials = read_trials(EXCEL_PATH)
    hr_min, hr_max = normalize_all(trials)
    processed = preprocess_trials(trials, hr_min, hr_max, cutoff_hz=FOURIER_CUTOFF_HZ)

    # --- Fit using the SAME method as fit_paper_faithful.py (global across ALL trials) ---
    theta0 = initial_guess(processed)
    res_fn = build_residual(processed)

    best = None
    rng = np.random.default_rng(0)
    for r in range(1):  # keep identical behavior
        theta_start = theta0 if r == 0 else theta0 + rng.normal(scale=0.15, size=theta0.shape)
        sol = least_squares(
            res_fn, theta_start,
            verbose=0,
            max_nfev=LSQ_MAX_NFEV,
            loss=LSQ_LOSS,
            f_scale=LSQ_F_SCALE
        )
        if best is None or sol.cost < best.cost:
            best = sol

    theta_hat = best.x
    A, B, C, Ds = unpack_params(theta_hat, len(processed))  # DO NOT print A/B/C

    # --- Select OFF only for outputs ---
    off = []
    for i, seg in enumerate(processed):
        if "off-transient" in seg["name"].lower():
            off.append((i, seg))

    if len(off) == 0:
        raise RuntimeError("No off-transient sheets found. Check sheet names in Excel.")

    # --- Print D (OFF only) ---
    print("\nPer-trial demand D (OFF only; computed from global fit across all trials):")
    for j, (i, seg) in enumerate(off, 1):
        Dn = float(Ds[i])
        Dbpm = hr_min + Dn * (hr_max - hr_min)
        print(f"  {seg['name']:<18}  D_n={Dn:.6f}   D_bpm≈{Dbpm:.3f}")

    # --- Plot OFF only ---
    n = len(off)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.8 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # --- Metrics (OFF only) ---
    per_trial = []
    y_true_all = []
    y_pred_all = []

    for ax, (i, seg) in zip(axes, off):
        t = seg["t"]
        bpm_raw = seg["bpm_raw"]
        bpm_s = seg["bpm_smooth"]
        hr = seg["hr_n"]
        D = float(Ds[i])

        y_hat_n = simulate_segment(t, hr0=hr[0], A=A, B=B, C=C, D=D)
        bpm_hat = hr_min + y_hat_n * (hr_max - hr_min)

        # metrics vs smoothed (paper-fitting target)
        rmse, mae, r2, N = metrics(bpm_s, bpm_hat)
        per_trial.append((seg["name"], rmse, mae, r2, N))

        y_true_all.append(bpm_s)
        y_pred_all.append(bpm_hat)

        ax.plot(t, bpm_raw, label="raw bpm", alpha=0.6)
        ax.plot(t, bpm_s, label="Fourier low-pass bpm")
        ax.plot(t, bpm_hat, label="model fit bpm")
        ax.set_title(seg["name"])
        ax.set_xlabel("t (s, shifted to segment start)")
        ax.set_ylabel("bpm")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig("OFF_transients_fit.png", dpi=400, bbox_inches="tight")
    plt.show()

    # --- Print metrics table ---
    print("\nFit quality metrics (OFF only; compared to Fourier low-pass bpm):")
    print("  (L_data = RMSE in bpm)")
    print(f"{'Trial':<18} {'L_data(RMSE)':>12} {'MAE':>10} {'R^2':>10} {'N':>6}")
    for name, rmse, mae, r2, N in per_trial:
        r2_str = f"{r2:.4f}" if np.isfinite(r2) else "nan"
        print(f"{name:<18} {rmse:>12.3f} {mae:>10.3f} {r2_str:>10} {N:>6d}")

    # overall pooled (concatenate all off points)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    rmse_all, mae_all, r2_all, N_all = metrics(y_true_all, y_pred_all)

    # mean-of-trials (unweighted)
    rmse_mean = float(np.mean([x[1] for x in per_trial]))
    mae_mean  = float(np.mean([x[2] for x in per_trial]))
    r2_vals   = [x[3] for x in per_trial if np.isfinite(x[3])]
    r2_mean   = float(np.mean(r2_vals)) if len(r2_vals) else float("nan")

    r2_all_str = f"{r2_all:.4f}" if np.isfinite(r2_all) else "nan"
    r2_mean_str = f"{r2_mean:.4f}" if np.isfinite(r2_mean) else "nan"

    print("\nOverall (OFF only):")
    print(f"  Pooled over all OFF points:   L_data(RMSE)={rmse_all:.3f}  MAE={mae_all:.3f}  R^2={r2_all_str}  N={N_all}")
    print(f"  Mean of per-trial metrics:    L_data(RMSE)={rmse_mean:.3f}  MAE={mae_mean:.3f}  R^2={r2_mean_str}")

    # Note: we intentionally do NOT print A/B/C.


if __name__ == "__main__":
    main()
