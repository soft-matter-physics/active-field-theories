"""
Active Model H:
Recover params from simulated flow time-series.

1. Simulate Active Model H for parameter sets
2. Extract spatiotemporal features: log-energy-spectrum E(k,t)
3. Train a CNN to map those features to parameters
4. Evaluate on fresh held-out simulations
"""


# 1. Imports


import numpy as np
import scipy.fft as sp_fft
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import joblib
import os
import datetime


# 2. Configuration

# Output paths
SAVE_DIR  = "amh_float32_32"
os.makedirs(SAVE_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Simulation grid & timen3
SIM_NG   = 32      # spatial grid: NG x NG points
SIM_NT   = 4000    # total integration steps
SIM_DT   = 0.005   # time-step size
SIM_H    = 1.0     # spatial grid spacing
SIM_NF   = 100      # number of snapshots saved per simulation
SIM_PHI0 = 0.2     # mean value of the order-parameter phi

# parameter ranges

KT_MIN, KT_MAX = -1.2,  -0.1   # activity coefficient k_tilde  (always negative)
KT_DEAD        =  0.05           # enforce |k_tilde| > KT_DEAD to stay in active regime
KP_MIN, KP_MAX =  0.8,   1.2    # kappa 
B_MIN,  B_MAX  =  0.25,  1.25   # b     

# Sampling strategy
# Mix a regular grid (with jitter) and purely random draws for broad coverage.
GRID_N_KT   = 10    # grid resolution along k_tilde axis
GRID_N_KP   = 10    # grid resolution along kappa axis
GRID_N_B    = 10    # grid resolution along b axis
JITTER_FRAC = 0.2   # jitter magnitude as a fraction of axis range (Gaussian sigma)
P_RANDOM    = 0.3   # probability of a fully random draw instead of a grid point

# Feature extraction
N_FRAMES = 100   # number of evenly-spaced time frames used to build features

# Training
N_SAMPLES    = 500   # total simulations to generate
N_WORKERS    = 10     # parallel workers for data generation
TRAIN_FRAC   = 0.85   # fraction used for training  (remainder = validation)
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 200

# Evaluation
N_TESTS = 100   # number of fresh test simulations for final reporting



# 3. Simulator


class FPS_Numpy:
    """
    Parameters are passed as a dict `p` with keys:
        Ng, Nt, dt, h, a, b, kp, k_tilde, Nf
    """

    def __init__(self, p):
        self.Ng      = p['Ng']
        self.Nt      = p['Nt']
        self.dt      = np.float32(p['dt'])
        self.h       = np.float32(p['h'])
        self.a       = np.float32(p['a'])
        self.b       = np.float32(p['b'])
        self.kp      = np.float32(p['kp'])
        self.k_tilde = np.float32(p['k_tilde'])
        self.Nf      = p.get('Nf', 50)

        # Wavenumber arrays (float32)
        q = (np.fft.fftfreq(self.Ng) * (2 * np.pi / self.h)).astype(np.float32)
        self.qx, self.qy = np.meshgrid(q, q, indexing='ij')   # (Ng, Ng)
        self.q2 = (self.qx**2 + self.qy**2).astype(np.float32)

        # Spectral derivative operators (complex64)
        self.Dx = (1j * self.qx).astype(np.complex64)
        self.Dy = (1j * self.qy).astype(np.complex64)

        # Inverse Laplacian  (zero mode set to 0 to avoid division by zero)
        q2_safe  = np.where(self.q2 == 0, np.float32(1e-16), self.q2)
        self.iq2 = np.where(
            self.q2 > 0, np.float32(1) / q2_safe, np.float32(0)
        ).astype(np.float32)

        # Projection helpers  q_alpha / |q|^2
        self.iQ2x = (self.qx * self.iq2).astype(np.float32)
        self.iQ2y = (self.qy * self.iq2).astype(np.float32)

        # Exponential factor for the linear part  e^{L * dt}
        self.eA = np.exp(
            -self.q2 * (self.a + self.kp * self.q2) * self.dt
        ).astype(np.float32)

        # Coefficient for the nonlinear (cubic) term
        self.q2b = (self.q2 * self.b).astype(np.float32)

    def flow(self, u_hat):
        """
        Given phi_hat in Fourier space, return physical-space velocity (vx, vy)
        satisfying the Stokes equation.
        """
        ux = sp_fft.ifft2(self.Dx * u_hat)
        uy = sp_fft.ifft2(self.Dy * u_hat)

        # Active stress tensor components (Fourier space)
        sxx = np.float32(0.5) * sp_fft.fft2(uy**2 - ux**2)
        sxy =                  -sp_fft.fft2(ux * uy)

        # Divergence of stress (force)
        fqx = self.Dx * sxx + self.Dy * sxy
        fqy = self.Dx * sxy - self.Dy * sxx

        # Project onto divergence-free subspace (incompressibility constraint)
        fac = fqx * self.iQ2x + fqy * self.iQ2y
        vx  = sp_fft.ifft2(self.iq2 * fqx - fac * self.iQ2x)
        vy  = sp_fft.ifft2(self.iq2 * fqy - fac * self.iQ2y)

        return np.real(vx).astype(np.float32), np.real(vy).astype(np.float32)

    def step(self, u_hat):
        """
        Advance phi_hat by one time-step
        """
        uc       = sp_fft.ifft2(u_hat)
        vxg, vyg = self.flow(u_hat)
        ux       = sp_fft.ifft2(self.Dx * u_hat)
        uy       = sp_fft.ifft2(self.Dy * u_hat)

        advection = -self.k_tilde * sp_fft.fft2(vxg * ux + vyg * uy)
        nonlinear = -self.q2b     * sp_fft.fft2(uc**3)
        N_hat     = advection + nonlinear

        L          = -self.q2 * (self.a + self.kp * self.q2)
        denom      = np.where(L == 0, np.float32(1.0), L)
        propagator = np.where(L == 0, self.dt, (self.eA - np.float32(1)) / denom)

        return u_hat * self.eA + N_hat * propagator

    def run_full(self):
        """
        Integrate from t=0 to t=Nt*dt.
        Snapshots are saved only from the last 20% of the run

        Returns
        -------
        vx_ts, vy_ts, phi_ts : float32 arrays of shape (Nf, Ng, Ng)
        """

        # initial condition: small Gaussian perturbation around phi0.

        u = (SIM_PHI0 * np.ones((self.Ng, self.Ng))
             + np.float32(0.05) * np.random.randn(self.Ng, self.Ng)
             ).astype(np.float32)
        u_hat = sp_fft.fft2(u)   # complex64

        vx_ts, vy_ts, phi_ts = [], [], []

        # Only save snapshots from the last (1 - LATE_FRAC) of the run
        LATE_FRAC  = 0.8
        start_step = int(self.Nt * LATE_FRAC)
        late_steps = self.Nt - start_step
        save_every = max(1, late_steps // self.Nf)

        for i in range(self.Nt):
            vxg, vyg = self.flow(u_hat)
            if i >= start_step and (i - start_step) % save_every == 0:
                vx_ts.append((vxg * self.k_tilde).astype(np.float32))
                vy_ts.append((vyg * self.k_tilde).astype(np.float32))
                phi_ts.append(np.real(sp_fft.ifft2(u_hat)).astype(np.float32))
            u_hat = self.step(u_hat)

        return (
            np.array(vx_ts[:self.Nf],  dtype=np.float32),
            np.array(vy_ts[:self.Nf],  dtype=np.float32),
            np.array(phi_ts[:self.Nf], dtype=np.float32),
        )



# 4. Feature Extraction


def extract_spatiotemporal_features(vx_ts, vy_ts):
    """
    Parameters
    ----------
    vx_ts, vy_ts : (Nt, Ng, Ng) float32

    Returns
    -------
    Xc : (1, N_FRAMES, Ng//2) float32  — log energy-spectrum 
    """

    Ng    = vx_ts.shape[1]
    max_r = Ng // 2

    # Sub-sample N_FRAMES evenly-spaced frames
    indices = np.linspace(0, len(vx_ts) - 1, N_FRAMES, dtype=int)
    vx_sub  = vx_ts[indices]   # (N_FRAMES, Ng, Ng)
    vy_sub  = vy_ts[indices]

    # Pre-compute integer radial bin index for every pixel (same for all frames)
    x = np.arange(Ng) - Ng // 2
    X, Y = np.meshgrid(x, x)
    r    = np.sqrt(X**2 + Y**2).astype(int)   # (Ng, Ng)
    nr   = np.bincount(r.ravel())              # pixel count per radial bin

    # Radially-averaged energy spectrum for each frame
    spec_t = np.zeros((N_FRAMES, max_r), dtype=np.float32)
    for t in range(N_FRAMES):
        vxh = np.fft.fftshift(np.fft.fft2(vx_sub[t]))
        vyh = np.fft.fftshift(np.fft.fft2(vy_sub[t]))
        E   = np.float32(0.5) * (np.abs(vxh)**2 + np.abs(vyh)**2)
        # np.bincount requires float64 weights; cast result back to float32
        tbin      = np.bincount(r.ravel(), E.ravel().astype(np.float64))
        spec_t[t] = (tbin/nr)[:max_r].astype(np.float32)

    # Add channel dimension for CNN input: (1, N_FRAMES, Ng//2)
    Xc = np.expand_dims(
        np.log10(spec_t + np.float32(1e-16)), axis=0
    ).astype(np.float32)

    return Xc



# 5. Data Generation


def _sample_kt():
    """Draw k_tilde uniformly from [KT_MIN, KT_MAX], rejecting |k_tilde| < KT_DEAD."""
    kt = np.random.uniform(KT_MIN, KT_MAX)
    while abs(kt) < KT_DEAD:
        kt = np.random.uniform(KT_MIN, KT_MAX)
    return np.float32(kt)


def _sample_parameters(idx):
    """
    Return (kt, kp, b) as float32.
    With probability P_RANDOM: fully random draw.
    Otherwise: jittered grid point for good parameter-space coverage.
    """

    if np.random.rand() < P_RANDOM:
        return (
            _sample_kt(),
            np.float32(np.random.uniform(KP_MIN, KP_MAX)),
            np.float32(np.random.uniform(B_MIN,  B_MAX)),
        )

    # Map flat index to 3-D grid index
    i    = idx % (GRID_N_KT * GRID_N_KP * GRID_N_B)
    i_kt = i  // (GRID_N_KP * GRID_N_B)
    i_kp = (i //  GRID_N_B) % GRID_N_KP
    i_b  =  i  %  GRID_N_B

    kt = np.linspace(KT_MIN, KT_MAX, GRID_N_KT)[i_kt]
    kp = np.linspace(KP_MIN, KP_MAX, GRID_N_KP)[i_kp]
    b  = np.linspace(B_MIN,  B_MAX,  GRID_N_B )[i_b]

    # Add Gaussian jitter so the model sees values between grid lines
    kt += JITTER_FRAC * (KT_MAX - KT_MIN) * np.random.randn()
    kp += JITTER_FRAC * (KP_MAX - KP_MIN) * np.random.randn()
    b  += JITTER_FRAC * (B_MAX  - B_MIN)  * np.random.randn()

    # Clip to valid ranges and enforce the dead-zone constraint
    kt = np.clip(kt, KT_MIN, KT_MAX)
    if abs(kt) < KT_DEAD:
        kt = np.sign(kt) * KT_DEAD if kt != 0 else KT_DEAD

    return (
        np.float32(kt),
        np.float32(np.clip(kp, KP_MIN, KP_MAX)),
        np.float32(np.clip(b,  B_MIN,  B_MAX)),
    )


def _run_sim(kt, kp, b):
    """Run one simulation; return (None, None, None) if it diverges."""
    p = dict(
        Ng=SIM_NG, Nt=SIM_NT, dt=SIM_DT, h=SIM_H,
        a=-float(b),      # a = -b in Active Model H
        b=float(b),
        kp=float(kp),
        k_tilde=float(kt),
        Nf=SIM_NF,
    )
    vx, vy, phi = FPS_Numpy(p).run_full()
    if not (np.isfinite(vx).all() and np.isfinite(vy).all()
            and np.isfinite(phi).all()):
        return None, None, None
    return vx, vy, phi


def _generate_one(idx):
    """
    Worker function called by each multiprocessing process.
    Returns None if the simulation diverged.
    """
    kt, kp, b   = _sample_parameters(idx)
    vx, vy, phi = _run_sim(kt, kp, b)
    if vx is None:
        return None

    Xc = extract_spatiotemporal_features(vx, vy)

    # Predict in log-space for better-conditioned regression
    y = np.array([np.log10(abs(kt)), np.log10(b), np.log10(kp)],
                 dtype=np.float32)
    return Xc, y, vx, vy, phi, kt, kp, b


def generate_and_save_data(n_samples=N_SAMPLES, n_workers=N_WORKERS):
    """Generate `n_samples` simulations in parallel and save to disk."""
    print(f"Generating {n_samples} samples with {n_workers} workers ...")

    with Pool(n_workers) as pool:
        out = list(tqdm(
            pool.imap_unordered(_generate_one, range(n_samples)),
            total=n_samples,
        ))

    out = [o for o in out if o is not None]
    print(f"  {len(out)} / {n_samples} simulations succeeded.")

    Xc_list, y_list, rows = [], [], []
    for Xc, y, vx, vy, phi, kt, kp, b in out:
        Xc_list.append(Xc)
        y_list.append(y)
        # rows.append(dict(kt=kt, kp=kp, b=b, vx=vx, vy=vy, phi=phi))

    Xc_arr = np.array(Xc_list, dtype=np.float32)   # (N, 1, N_FRAMES, Ng//2)
    y_arr  = np.array(y_list,  dtype=np.float32)   # (N, 3)

    np.savez_compressed(f"{SAVE_DIR}/dataset_{TIMESTAMP}.npz",
                        Xc=Xc_arr, y=y_arr)
    # pd.DataFrame(rows).to_pickle(f"{SAVE_DIR}/groundtruth_df_{TIMESTAMP}.pkl")
    # print(f"Saved to {SAVE_DIR}/")
    return Xc_arr, y_arr



# 6. Model
#
# CNN that maps the log-energy-spectrum image
# (batch, 1, N_FRAMES, Ng/2) to  (batch, 3) predicted log10 parameters


# class ActiveMatterNet2DNOET(nn.Module):
#     """
#     Input  : (batch, 1, N_FRAMES, Ng//2)  log energy-spectrum image
#     Output : (batch, 3)                   [log10|k_tilde|, log10 b, log10 kappa]
#     """

#     def __init__(self, n_frames=N_FRAMES, n_bins=SIM_NG // 2):
#         super().__init__()

#         # Two conv blocks, each halving spatial dims via MaxPool
#         self.cnn = nn.Sequential(
#             # Block 1: 1 -> 32 channels
#             nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool2d(2),          # output: (N_FRAMES/2, n_bins/2)
#             # Block 2: 32 -> 64 channels
#             nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool2d(2),          # output: (N_FRAMES/4, n_bins/4)
#             nn.Flatten(),
#         )

#         cnn_out_dim = 64 * (n_frames // 4) * (n_bins // 4)

#         # MLP head -> 3 outputs
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_out_dim, 128), nn.LeakyReLU(0.1),
#             nn.Linear(128,          64), nn.LeakyReLU(0.1),
#             nn.Linear(64,            3),
#         )

#     def forward(self, xc):
#         return self.fc(self.cnn(xc))

class ActiveMatterNet2DNOET(nn.Module):
    def __init__(self, n_frames=N_FRAMES, n_bins=SIM_NG // 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        cnn_dim = 64 * (n_frames // 4) * (n_bins // 4)
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64),      nn.LeakyReLU(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, xc):
        return self.fc(self.cnn(xc))



# 7. Training


def train_model(Xc, y, epochs=EPOCHS):
    """
    Train the model on (Xc, y) and save weights + scaler to disk.
    """
    # Standardise targets to zero mean / unit variance
    scaler   = StandardScaler()
    y_scaled = scaler.fit_transform(y).astype(np.float32)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_{TIMESTAMP}.pkl")

    dataset = TensorDataset(
        torch.from_numpy(Xc),          # already float32, no copy
        torch.from_numpy(y_scaled),
    )
    n_train  = int(TRAIN_FRAC * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model   = ActiveMatterNet2DNOET()
    opt     = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for ep in range(epochs):
        # Training pass
        model.train()
        train_loss = 0.0
        for xc, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xc), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Validation pass (no gradients needed)
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                loss_fn(model(xc), yb).item() for xc, yb in val_dl
            ])
        val_losses.append(val_loss)

        if ep % 10 == 0:
            print(f"Epoch {ep:03d} | Train {train_loss:.4e} | Val {val_loss:.4e}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/model_{TIMESTAMP}.pth")
    joblib.dump({"train": train_losses, "val": val_losses},
                f"{SAVE_DIR}/history_{TIMESTAMP}.pkl")

    return model, scaler



# 8. Evaluation


def test_pipeline(model, scaler, n_tests=N_TESTS):
    """
    Run `n_tests` fresh simulations, predict parameters, and report MAPE.
    Saves results DataFrame to disk and returns it.
    """
    model.eval()
    rows = []

    for _ in tqdm(range(n_tests), desc="Testing"):
        kt = _sample_kt()
        kp = np.float32(np.random.uniform(KP_MIN, KP_MAX))
        b  = np.float32(np.random.uniform(B_MIN,  B_MAX))

        vx, vy, phi = _run_sim(kt, kp, b)
        if vx is None:
            continue   # skip diverged runs

        Xc = extract_spatiotemporal_features(vx, vy)
        with torch.no_grad():
            pred_scaled = model(
                torch.from_numpy(Xc).unsqueeze(0)   # add batch dim
            ).numpy()

        # Invert StandardScaler, then convert from log10 back to linear scale
        pred    = scaler.inverse_transform(pred_scaled)[0]
        kt_pred = float(10 ** pred[0])
        b_pred  = float(10 ** pred[1])
        kp_pred = float(10 ** pred[2])

        rows.append(dict(
            kt=abs(float(kt)), kt_pred=kt_pred,
            kp=float(kp),      kp_pred=kp_pred,
            b=float(b),        b_pred=b_pred,
            vx=vx, vy=vy, phi=phi,
        ))

    df = pd.DataFrame(rows)



    print("\n── Test results ──────────────────────────────")
    for true_col, pred_col in [("kt", "kt_pred"), ("b", "b_pred"), ("kp", "kp_pred")]:
        mape = (np.abs(df[true_col] - df[pred_col]) / np.abs(df[true_col]) * 100).mean()
        print(f"  MAPE {true_col:>3s}: {mape:.2f}%")

    df.to_pickle(f"{SAVE_DIR}/results_df_{TIMESTAMP}.pkl")
    return df



# 9. Main Call


if __name__ == "__main__":
    # Step 1: Generate (or load) training data
    Xc, y = generate_and_save_data()

    # To load a previously saved dataset instead, comment the line above and uncomment below:

    #   data = np.load(f"{SAVE_DIR}/dataset_XXXXXXXX_XXXXXX.npz")
    #   Xc, y = data["Xc"], data["y"]

    # Step 2: Train 
    model, scaler = train_model(Xc, y)

    # To load a pre-trained checkpoint instead:
    #   model = ActiveMatterNet2DNOET()
    #   model.load_state_dict(torch.load(f"{SAVE_DIR}/model_XXXXXXXX_XXXXXX.pth"))
    #   model.eval()
    #   scaler = joblib.load(f"{SAVE_DIR}/scaler_XXXXXXXX_XXXXXX.pkl")

    # Step 3: Evaluate 
    results_df = test_pipeline(model, scaler, n_tests=N_TESTS)
