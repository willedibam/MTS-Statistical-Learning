# generators.py
import numpy as np
from numpy.random import default_rng
from scipy.stats import zscore

# Global RNG (used only when no rng is provided explicitly)
_rng_global = default_rng(12345)

# ---------- helpers -----------------------------------------------------
def _ring_laplacian(M: int, k: int = 2, w: float = 1.0) -> np.ndarray:
    """Ring Laplacian with k-neighbour symmetric coupling and weight w."""
    A = np.zeros((M, M))
    for i in range(M):
        for d in range(1, k+1):
            A[i, (i+d) % M] = w
            A[i, (i-d) % M] = w
    D = np.diag(A.sum(axis=1))
    return D - A

def _ring_directed_adj(M: int, stride: int = 1, w: float = 1.0) -> np.ndarray:
    """Unidirectional ring adjacency: i → (i+stride). Useful to inject direction."""
    A = np.zeros((M, M))
    for i in range(M):
        A[i, (i+stride) % M] = w
    return A

# ===================== 0) VAR(1) =======================================
def gen_var(M: int = 20, T: int = 2000, phi: float = 1.0, coupling: float = 0.8,
            noise_std: float = 0.1, transients: int = 50, rng=None) -> np.ndarray:
    """
    Stable VAR(1): x_t = A x_{t-1} + ε_t, A ≈ phi*I + coupling*(off-diag)/M.
    Baseline linear–Gaussian.
    """
    rng = rng or _rng_global
    A = np.eye(M) * phi + (coupling/M)*(np.ones((M, M)) - np.eye(M))
    # Keep spectral radius < 1
    ev = np.linalg.eigvals(A); sr = np.max(np.abs(ev))
    if sr >= 0.98: A = A / (1.05 * sr)
    steps = transients + T
    X = np.zeros((steps, M), float)
    eps = rng.normal(0.0, noise_std, size=(steps, M))
    for t in range(1, steps):
        X[t] = A @ X[t-1] + eps[t]
    return zscore(X[transients:], axis=0)

# ===================== 1) OU network ===================================
def gen_ou_network(M=10, T=2000, dt=0.01, alpha=1.0, k=1, w=0.3,
                   sigma=0.5, transients=500, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    L = _ring_laplacian(M, k=k, w=w); beta = 1.0
    steps = transients + T
    X = np.zeros((steps, M))
    for t in range(1, steps):
        dW = rng.normal(scale=np.sqrt(dt), size=M)
        drift = (-alpha * X[t-1] - beta * (L @ X[t-1])) * dt
        X[t] = X[t-1] + drift + sigma * dW
    return zscore(X[transients:], axis=0)

# ===================== 2) Kuramoto =====================================
def gen_kuramoto(M=10, T=5000, dt=0.002, K=1.5, k=1, w=1.0,
                 omega_mean=2*np.pi*0.1, omega_std=0.01, eta=0.0,
                 transients=2000, rng=None, output="sin", directed: bool = False) -> np.ndarray:
    rng = rng or _rng_global
    A = (-_ring_laplacian(M, k=k, w=w)); np.fill_diagonal(A, 0)   # symmetric by default
    if directed:
        A = _ring_directed_adj(M, stride=1, w=w)                   # optional direction
    theta = rng.uniform(0, 2*np.pi, size=M)
    omega = rng.normal(omega_mean, omega_std, size=M)
    steps = transients + T
    Y = np.zeros((steps, M))
    for t in range(steps):
        Y[t] = np.sin(theta) if output == "sin" else (np.cos(theta) if output == "cos" else theta)
        # mean-field coupling per node
        S = np.sin(theta[None, :] - theta[:, None])     # S[i,j] = sin(theta_j - theta_i)
        coupling_term = K * (A * S).sum(axis=1)         # sum_j A_ij * sin(theta_j - theta_i)
        noise = eta * np.sqrt(dt) * rng.normal(size=M)
        theta = np.mod(theta + (omega + coupling_term) * dt + noise, 2*np.pi)
    return zscore(Y[transients:], axis=0)

# ===================== 3) Stuart–Landau ================================
def gen_stuart_landau(M=10, T=4000, dt=0.002, lam=0.2, omega=2*np.pi*0.08,
                      g=0.15, k=1, w=0.5, sigma=0.01, transients=1500,
                      rng=None, output="real", directed: bool = False) -> np.ndarray:
    rng = rng or _rng_global
    # coupling: symmetric Laplacian by default; optional directed difference
    if directed:
        A = _ring_directed_adj(M, stride=1, w=w)
        coupling_op = lambda z: g * (A @ z - z)  # simple directed diff
    else:
        L = _ring_laplacian(M, k=k, w=w)
        coupling_op = lambda z: -g * (L @ z)
    z = rng.normal(scale=0.1, size=M) + 1j*rng.normal(scale=0.1, size=M)
    steps = transients + T; Y = np.zeros((steps, M))
    iomega = 1j * omega
    for t in range(steps):
        Y[t] = z.real if output == "real" else (z.imag if output == "imag" else np.abs(z))
        drift = ((lam + iomega) - (np.abs(z)**2)) * z + coupling_op(z)
        dW = (rng.normal(scale=np.sqrt(dt), size=M) + 1j*rng.normal(scale=np.sqrt(dt), size=M))
        z = z + drift * dt + sigma * dW
    return zscore(Y[transients:], axis=0)

# ===================== 4) Lorenz–96 ====================================
def gen_lorenz96(M=10, T=3000, dt=0.01, F=8.0, transients=1000, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    def f(x):
        xp1 = np.roll(x, -1); xm1 = np.roll(x, 1); xm2 = np.roll(x, 2)
        return (xp1 - xm2) * xm1 - x + F
    steps = transients + T
    X = rng.normal(size=M); Y = np.zeros((steps, M))
    for t in range(steps):
        Y[t] = X
        k1 = f(X); k2 = f(X + 0.5*dt*k1); k3 = f(X + 0.5*dt*k2); k4 = f(X + dt*k3)
        X = X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return zscore(Y[transients:], axis=0)

# ===================== 5) Rössler (coupled) ============================
def gen_rossler_coupled(M=6, T=6000, dt=0.01, a=0.2, b=0.2, c=5.7,
                        eps=0.05, k=1, w=1.0, transients=2000,
                        rng=None, output="x") -> np.ndarray:
    rng = rng or _rng_global
    L = _ring_laplacian(M, k=k, w=w) * eps
    X = rng.normal(scale=0.1, size=M)
    Y = rng.normal(scale=0.1, size=M)
    Z = rng.normal(scale=0.1, size=M)
    steps = transients + T; out = np.zeros((steps, M))
    def fx(x,y,z): return -y - z
    def fy(x,y,z): return x + a*y
    def fz(x,y,z): return b + z*(x - c)
    for t in range(steps):
        out[t] = {"x": X, "y": Y, "z": Z}[output]
        cx = -(L @ X)  # diffusive coupling on x
        k1x, k1y, k1z = fx(X,Y,Z)+cx, fy(X,Y,Z), fz(X,Y,Z)
        k2x, k2y, k2z = fx(X+0.5*dt*k1x, Y+0.5*dt*k1y, Z+0.5*dt*k1z)+cx, fy(X+0.5*dt*k1x, Y+0.5*dt*k1y, Z+0.5*dt*k1z), fz(X+0.5*dt*k1x, Y+0.5*dt*k1y, Z+0.5*dt*k1z)
        k3x, k3y, k3z = fx(X+0.5*dt*k2x, Y+0.5*dt*k2y, Z+0.5*dt*k2z)+cx, fy(X+0.5*dt*k2x, Y+0.5*dt*k2y, Z+0.5*dt*k2z), fz(X+0.5*dt*k2x, Y+0.5*dt*k2y, Z+0.5*dt*k2z)
        k4x, k4y, k4z = fx(X+dt*k3x, Y+dt*k3y, Z+dt*k3z)+cx, fy(X+dt*k3x, Y+dt*k3y, Z+dt*k3z), fz(X+dt*k3x, Y+dt*k3y, Z+dt*k3z)
        X += (dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
        Y += (dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
        Z += (dt/6)*(k1z + 2*k2z + 2*k3z + k4z)
    return zscore(out[transients:], axis=0)

# ===================== 6) CML logistic =================================
def gen_cml_logistic(M=64, T=2000, r=3.8, eps=0.2, transients=500, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    X = rng.uniform(low=0.1, high=0.9, size=M)
    def f(x): return r * x * (1 - x)
    Y = np.zeros((transients + T, M))
    for t in range(transients + T):
        Y[t] = X
        X = (1 - eps) * f(X) + 0.5 * eps * (f(np.roll(X, 1)) + f(np.roll(X, -1)))
    return zscore(Y[transients:], axis=0)

# ===================== 7) OU heavy-tail ================================
def gen_ou_heavytail(M=10, T=2000, dt=0.01, alpha=1.0, sigma=0.5, df=2.5,
                     transients=500, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    steps = transients + T
    X = np.zeros((steps, M))
    for t in range(1, steps):
        dW = rng.standard_t(df, size=M) * np.sqrt(dt)
        X[t] = X[t-1] + (-alpha * X[t-1]) * dt + sigma * dW
    return zscore(X[transients:], axis=0)

# ===================== 8) GBM returns ==================================
def gen_gbm_returns(M=10, T=3000, dt=1/252, mu=0.05, sigma=0.2, rho=0.6, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    f = rng.normal(scale=np.sqrt(dt), size=T)
    idio = rng.normal(scale=np.sqrt(dt), size=(T, M))
    shocks = np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * idio
    rts = (mu - 0.5*sigma**2)*dt + sigma * shocks  # log-returns
    return zscore(rts, axis=0)

# ===================== 9) Time-warp clones =============================
def gen_timewarp_clones(M=10, T=2000, base_freq=0.03, warp_strength=0.15,
                        noise=0.05, K=8, transients=0, rng=None) -> np.ndarray:
    rng = rng or _rng_global
    steps = transients + T
    t = np.arange(steps)
    base = np.sin(2*np.pi*base_freq * t) + rng.normal(0, noise, size=steps)
    X = np.zeros((steps, M))
    knots = np.linspace(0, steps - 1, K, dtype=int)
    seg = (steps - 1) / (K - 1) if K > 1 else steps - 1
    jitter_sd = warp_strength * seg
    for m in range(M):
        jitter = rng.normal(0, jitter_sd, size=K)
        warped = np.clip(np.maximum.accumulate(knots.astype(float) + jitter), 0, steps - 1)
        tau = np.interp(t, knots, warped)
        X[:, m] = np.interp(tau, t, base) + rng.normal(0, noise, size=steps)
    return zscore(X[transients:], axis=0)
