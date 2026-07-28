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


SAVE_DIR  = "an_psi_results"
os.makedirs(SAVE_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#Fixed physical constants (same as force-projection file) 
ETA   = 1.0   # shear viscosity
GAMMA = 1.0   # rotational viscosity  

#Simulation grid & time 
SIM_NG = 32
SIM_NT = 4000
SIM_DT = 0.005
SIM_H  = 1.0    # grid spacing (domain size = Ng*H)
SIM_NF = 100     # number of saved snapshots

#Free parameter ranges 
ZETA_MIN,  ZETA_MAX  = -1.2, -0.1   # contractile only (zeta < 0)
ZETA_DEAD            =  0.05
KAPPA_MIN, KAPPA_MAX =  0.5,  2.0

#Sampling 
GRID_N_ZETA  = 15
GRID_N_KAPPA = 15
JITTER_FRAC  = 0.2
P_RANDOM     = 0.3

#Features 
N_FRAMES = 100

#Training 
N_SAMPLES    = 500
N_WORKERS    = 10
TRAIN_FRAC   = 0.85
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 200

#Evaluation 
N_TESTS = 100



# 3. Simulator  (psi formulation)


class FPS_Nematic_Psi:
    """
    Pseudo-spectral ETD1 integrator using the stream-function approach.

    dict `p` keys: Ng, Nt, dt, h, kappa, zeta, eta, Nf
    (gamma fixed to GAMMA = 1.0 globally)
    """

    def __init__(self, p):
        self.Ng    = p['Ng']
        self.Nt    = p['Nt']
        self.dt    = np.float32(p['dt'])
        self.h     = np.float32(p.get('h', 1.0))
        self.Nf    = p.get('Nf', 50)

        kappa = np.float32(p['kappa'])
        zeta  = np.float32(p['zeta'])
        eta   = np.float32(p.get('eta', ETA))

        # Pre-computed dimensional coefficients
        self.kbe = kappa / (2.0 * eta)   # kappa/(2*eta): elastic part of psi eq.
        self.zbe = zeta  / eta           # zeta/eta:      active  part of psi eq.
        self.kG  = kappa * GAMMA         # kappa*Gamma:   elastic relaxation of theta

        # Wavenumber grid  
        fack = 2.0 * np.pi / self.h
        kx   = (np.fft.fftfreq(self.Ng) * fack).astype(np.float32)
        ky   = (np.fft.fftfreq(self.Ng) * fack).astype(np.float32)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing='ij')
        self.kx2 = self.kx ** 2
        self.ky2 = self.ky ** 2
        self.ksq = (self.kx2 + self.ky2).astype(np.float32)

        #  1/q^4  
        k4   = self.ksq ** 2
        k4_s = np.where(k4 < 1e-6, np.float32(1e10), k4)
        self.ik4 = (np.float32(1.0) / k4_s).astype(np.float32)
        # where q=0 the 1/q^4 blows up; we set those modes to 0 explicitly
        self.ik4[self.ksq == 0] = np.float32(0.0)

        # 2/3 dealiasing filter  (applied to advection nonlinearity only)
        kmax_x = np.max(np.abs(self.kx))
        kmax_y = np.max(np.abs(self.ky))
        flt = np.ones((self.Ng, self.Ng), dtype=np.float32)
        flt[np.abs(self.kx) > kmax_x * 2.0 / 3.0] = 0.0
        flt[np.abs(self.ky) > kmax_y * 2.0 / 3.0] = 0.0
        self.dAl = flt

        # Linear ETD operator for theta:  L = -kG * q^2 = -kappa * q^2
        self.L_theta = (-self.kG * self.ksq).astype(np.float32)
        self.eA      = np.exp(self.L_theta * self.dt).astype(np.float32)

    # 
    # _psi_hat: solve the biharmonic equation for psi in Fourier space
    # 
    def _psi_hat(self, theta_hat, theta):
        """
        Compute psi_hat from the current theta field.
        """
        sin2t_hat = sp_fft.fft2(np.sin(2.0 * theta))
        cos2t_hat = sp_fft.fft2(np.cos(2.0 * theta))

        return (- self.kbe * theta_hat
                + self.zbe * self.ik4
                  * (np.float32(0.5) * (self.kx2 - self.ky2) * sin2t_hat
                     - self.kx * self.ky * cos2t_hat))

    # 
    # step: one ETD1 advance of theta
    # 
    def step(self, theta_hat):
        """
        update theta_hat(t+dt) 
        """
        theta = np.real(sp_fft.ifft2(theta_hat)).astype(np.float32)

        psi_hat = self._psi_hat(theta_hat, theta)

        # velocity from stream function
        # vx = d_y psi  --> iky * psi_hat
        # vy = -d_x psi --> -ikx * psi_hat
        # (using c_x = d_x psi, c_y = d_y psi as in snippet to make the
        #  advection derivation transparent)
        c_x = np.real(sp_fft.ifft2(1j * self.kx * psi_hat)).astype(np.float32)  # = d_x psi = -vy
        c_y = np.real(sp_fft.ifft2(1j * self.ky * psi_hat)).astype(np.float32)  # = d_y psi = +vx

        # gradient of theta
        u_x = np.real(sp_fft.ifft2(1j * self.kx * theta_hat)).astype(np.float32)  # d_x theta
        u_y = np.real(sp_fft.ifft2(1j * self.ky * theta_hat)).astype(np.float32)  # d_y theta

        # advection:  c_x*(d_y theta) - c_y*(d_x theta) = -(v . grad theta)
        # dealiasing applied to the nonlinear (quadratic) term only
        advection = self.dAl * (c_x * u_y - c_y * u_x)

        # rotation: omega = -lap(psi)
        # In Fourier: -lap(psi) = q^2 * psi_hat  (since lap -> -q^2)
        # coefficient +1 matches the paper's d_t theta = ... + (d_x vy - d_y vx)
        omega = np.real(sp_fft.ifft2(self.ksq * psi_hat)).astype(np.float32)

        N     = advection + omega
        N_hat = sp_fft.fft2(N)

        # ETD1 propagator (same as force-projection file)
        denom      = np.where(self.L_theta == 0, np.float32(1.0), self.L_theta)
        propagator = np.where(self.L_theta == 0, self.dt,
                               (self.eA - np.float32(1.0)) / denom)

        return theta_hat * self.eA + N_hat * propagator, psi_hat

    # 
    # run_full: integrate and return late-time snapshots
    # 
    def run_full(self):
        """
        Returns vx_ts, vy_ts, theta_ts of shape (Nf, Ng, Ng), float32.
        Snapshots from the last 20% of the run (late-time active state).
        """
        # Small random perturbation around theta=0 (unstable aligned state)
        theta     = (np.float32(0.05)
                     * np.random.randn(self.Ng, self.Ng).astype(np.float32))
        theta_hat = sp_fft.fft2(theta)

        vx_ts, vy_ts, theta_ts = [], [], []

        LATE_FRAC  = 0.8
        start_step = int(self.Nt * LATE_FRAC)
        late_steps = self.Nt - start_step
        save_every = max(1, late_steps // self.Nf)

        for i in range(self.Nt):
            theta_hat, psi_hat = self.step(theta_hat)

            if i >= start_step and (i - start_step) % save_every == 0:
                # velocity from the most recently computed psi
                vx = np.real(sp_fft.ifft2( 1j * self.ky * psi_hat)).astype(np.float32)
                vy = np.real(sp_fft.ifft2(-1j * self.kx * psi_hat)).astype(np.float32)
                theta_snap = np.real(sp_fft.ifft2(theta_hat)).astype(np.float32)
                vx_ts.append(vx)
                vy_ts.append(vy)
                theta_ts.append(theta_snap)

        return (
            np.array(vx_ts[:self.Nf],    dtype=np.float32),
            np.array(vy_ts[:self.Nf],    dtype=np.float32),
            np.array(theta_ts[:self.Nf], dtype=np.float32),
        )



# 4. Feature Extraction  (identical to force-projection file)


def extract_spatiotemporal_features(vx_ts, vy_ts):
    """
    log10 radially-averaged kinetic energy spectrum E(k,t).

    Returns Xc : (1, N_FRAMES, Ng//2) float32
    """
    Ng    = vx_ts.shape[1]
    max_r = Ng // 2

    indices = np.linspace(0, len(vx_ts) - 1, N_FRAMES, dtype=int)
    vx_sub  = vx_ts[indices]
    vy_sub  = vy_ts[indices]

    x    = np.arange(Ng) - Ng // 2
    X, Y = np.meshgrid(x, x)
    r    = np.sqrt(X**2 + Y**2).astype(int)
    nr   = np.bincount(r.ravel())

    spec_t = np.zeros((N_FRAMES, max_r), dtype=np.float32)
    for t in range(N_FRAMES):
        vxh = np.fft.fftshift(np.fft.fft2(vx_sub[t]))
        vyh = np.fft.fftshift(np.fft.fft2(vy_sub[t]))
        E   = np.float32(0.5) * (np.abs(vxh)**2 + np.abs(vyh)**2)
        tbin      = np.bincount(r.ravel(), E.ravel().astype(np.float64))
        spec_t[t] = (tbin / nr)[:max_r].astype(np.float32)

    return np.expand_dims(
        np.log10(spec_t + np.float32(1e-16)), axis=0
    ).astype(np.float32)   # (1, N_FRAMES, max_r)



# 5. Data Generation


def _sample_zeta():
    z = np.random.uniform(ZETA_MIN, ZETA_MAX)
    while abs(z) < ZETA_DEAD:
        z = np.random.uniform(ZETA_MIN, ZETA_MAX)
    return np.float32(z)


def _sample_parameters(idx):
    if np.random.rand() < P_RANDOM:
        return (
            _sample_zeta(),
            np.float32(np.random.uniform(KAPPA_MIN, KAPPA_MAX)),
        )
    i_z = idx  % GRID_N_ZETA
    i_k = (idx // GRID_N_ZETA) % GRID_N_KAPPA

    zeta  = np.linspace(ZETA_MIN,  ZETA_MAX,  GRID_N_ZETA )[i_z]
    kappa = np.linspace(KAPPA_MIN, KAPPA_MAX, GRID_N_KAPPA)[i_k]

    zeta  += JITTER_FRAC * (ZETA_MAX  - ZETA_MIN)  * np.random.randn()
    kappa += JITTER_FRAC * (KAPPA_MAX - KAPPA_MIN) * np.random.randn()
    zeta   = np.clip(zeta, ZETA_MIN, ZETA_MAX)
    kappa  = np.clip(kappa, KAPPA_MIN, KAPPA_MAX)
    if abs(zeta) < ZETA_DEAD:
        zeta = -ZETA_DEAD

    return np.float32(zeta), np.float32(kappa)


def _run_sim(zeta, kappa):
    p = dict(Ng=SIM_NG, Nt=SIM_NT, dt=SIM_DT, h=SIM_H,
             kappa=float(kappa), zeta=float(zeta), eta=ETA, Nf=SIM_NF)
    vx, vy, theta = FPS_Nematic_Psi(p).run_full()
    if not (np.isfinite(vx).all() and np.isfinite(vy).all()
            and np.isfinite(theta).all()):
        return None, None, None
    return vx, vy, theta


def _generate_one(idx):
    zeta, kappa = _sample_parameters(idx)
    vx, vy, theta = _run_sim(zeta, kappa)
    if vx is None:
        return None
    Xc = extract_spatiotemporal_features(vx, vy)
    y  = np.array([np.log10(abs(zeta)), np.log10(kappa)], dtype=np.float32)
    return Xc, y, zeta, kappa


def generate_and_save_data(n_samples=N_SAMPLES, n_workers=N_WORKERS):
    print(f"Generating {n_samples} samples with {n_workers} workers ...")
    with Pool(n_workers) as pool:
        out = list(tqdm(
            pool.imap_unordered(_generate_one, range(n_samples)),
            total=n_samples,
        ))
    out = [o for o in out if o is not None]
    print(f"  {len(out)} / {n_samples} simulations succeeded.")

    Xc_arr = np.array([o[0] for o in out], dtype=np.float32)  # (N, 1, N_FRAMES, Ng//2)
    y_arr  = np.array([o[1] for o in out], dtype=np.float32)  # (N, 2)
    np.savez_compressed(f"{SAVE_DIR}/dataset_{TIMESTAMP}.npz", Xc=Xc_arr, y=y_arr)
    return Xc_arr, y_arr



# 6. Model


class ActiveNematicNet(nn.Module):
    """
    Input  : (batch, 1, N_FRAMES, Ng//2)  log10 kinetic-energy spectrum
    Output : (batch, 2)                   [log10|zeta|, log10 kappa]
    """
    def __init__(self, n_frames=N_FRAMES, n_bins=SIM_NG // 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        cnn_dim = 64 * (n_frames // 4) * (n_bins // 4)
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64),      nn.LeakyReLU(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, xc):
        return self.fc(self.cnn(xc))



# 7. Training


def train_model(Xc, y, epochs=EPOCHS):
    scaler   = StandardScaler()
    y_scaled = scaler.fit_transform(y).astype(np.float32)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_{TIMESTAMP}.pkl")

    dataset  = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(y_scaled))
    n_train  = int(TRAIN_FRAC * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model   = ActiveNematicNet()
    opt     = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for ep in range(epochs):
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

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_fn(model(xc), yb).item() for xc, yb in val_dl])
        val_losses.append(val_loss)

        if ep % 10 == 0:
            print(f"Epoch {ep:03d} | Train {train_loss:.4e} | Val {val_loss:.4e}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/model_{TIMESTAMP}.pth")
    joblib.dump({"train": train_losses, "val": val_losses},
                f"{SAVE_DIR}/history_{TIMESTAMP}.pkl")
    return model, scaler




# 8. Evaluation


def test_pipeline(model, scaler, n_tests=N_TESTS):
    model.eval()
    rows = []
    for _ in tqdm(range(n_tests), desc="Testing"):
        zeta  = _sample_zeta()
        kappa = np.float32(np.random.uniform(KAPPA_MIN, KAPPA_MAX))
        vx, vy, theta = _run_sim(zeta, kappa)
        if vx is None:
            continue
        Xc = extract_spatiotemporal_features(vx, vy)
        with torch.no_grad():
            pred = scaler.inverse_transform(
                model(torch.from_numpy(Xc).unsqueeze(0)).numpy()
            )[0]
        rows.append(dict(
            zeta=abs(float(zeta)),   zeta_pred=float(10**pred[0]),
            kappa=float(kappa),      kappa_pred=float(10**pred[1]),
            vx=vx, vy=vy, theta=theta,
        ))

    df = pd.DataFrame(rows)
    print("\n── Test results ──────────────────────────────")
    for tc, pc in [("zeta", "zeta_pred"), ("kappa", "kappa_pred")]:
        mape = (np.abs(df[tc] - df[pc]) / np.abs(df[tc]) * 100).mean()
        print(f"  MAPE {tc:>5s}: {mape:.2f}%")
    df.to_pickle(f"{SAVE_DIR}/results_df_{TIMESTAMP}.pkl")
    return df



# 9. Main


if __name__ == "__main__":
    Xc, y = generate_and_save_data()
    model, scaler = train_model(Xc, y)
    results_df = test_pipeline(model, scaler, n_tests=N_TESTS)