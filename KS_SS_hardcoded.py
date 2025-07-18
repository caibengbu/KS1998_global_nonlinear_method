import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import scipy
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def tauchen(rho: float, mu: float, sigma2: float, n: int, m: float = 3.0) -> tuple[NDArray, NDArray]:
    """Tauchen (1986) discretization of AR(1) process."""
    sigma_e = np.sqrt(sigma2)
    sigma_x = np.sqrt(sigma2 / (1 - rho ** 2))
    z_std = np.linspace(-m * sigma_x, m * sigma_x, n)
    z = mu + z_std
    P = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                P[i, j] = scipy.stats.norm.cdf((z[0] - rho * z[i] + (z[1] - z[0]) / 2) / sigma_e)
            elif j == n - 1:
                P[i, j] = 1 - scipy.stats.norm.cdf((z[-1] - rho * z[i] - (z[-1] - z[-2]) / 2) / sigma_e)
            else:
                P[i, j] = (
                    scipy.stats.norm.cdf((z[j] - rho * z[i] + (z[1] - z[0]) / 2) / sigma_e)
                    - scipy.stats.norm.cdf((z[j] - rho * z[i] - (z[1] - z[0]) / 2) / sigma_e)
                )
    return P, z

def ia_index(a_idx: int, z_idx: int, na: int) -> int:
    """Flattened index for state (a,z) when z runs fastest."""
    return z_idx * na + a_idx

@dataclass
class ModelParams:
    alpha: float = 0.33
    beta: float = 0.99
    delta: float = 0.025
    rho: float = 0.90
    sigma: float = 0.05
    frisch: float = 1.00
    eta: float = 7.60
    mu: float = 0.60
    num_grid_z: int = 7
    num_grid_a: int = 100
    grid_a_min: float = 1.0
    grid_a_max: float = 200.0

@dataclass
class Tolerances:
    labor: float = 1e-10
    ge: float = 1e-8
    pfi: float = 1e-10
    hist: float = 1e-10

@dataclass
class Weights:
    old1: float = 0.9
    old2: float = 0.9
    old3: float = 0.9
    old4: float = 0.9

class KrusellSmithModelSteadyState:
    """Krusell-Smith model with steady state solution."""
    def __init__(self, params: ModelParams, tol: Tolerances, weights: Weights, verbose: bool = True):
        self.p = params
        self.tol = tol
        self.weights = weights
        self.verbose = verbose
        self._setup_grids()
        self._initialize()

    def _setup_grids(self):
        self.trans_z, self.grid_z_raw = tauchen(self.p.rho, 0.0, self.p.sigma ** 2, self.p.num_grid_z)
        self.grid_z = np.exp(self.grid_z_raw)
        x = np.linspace(0, 0.5, self.p.num_grid_a)
        y = x ** 7 / x.max() ** 7
        self.grid_a = self.p.grid_a_min + (self.p.grid_a_max - self.p.grid_a_min) * y
        self.mgrida = np.repeat(self.grid_a[:, None], self.p.num_grid_z, axis=1)
        self.mgridz = np.repeat(self.grid_z[None, :], self.p.num_grid_a, axis=0)

    def _initialize(self):
        self.K = 7.50
        self.supply_L = 0.33
        self.current_dist = np.ones((self.p.num_grid_a, self.p.num_grid_z)) / (self.p.num_grid_a * self.p.num_grid_z)
        self.mpol_c = 0.01 * self.mgrida.copy()
        self.mpol_aprime = np.ones_like(self.mpol_c)
        self.mpol_aprime_new = np.zeros_like(self.mpol_c)
        self.mlambda = np.zeros_like(self.mpol_c)
        self.mlambda_new = np.zeros_like(self.mpol_c)
        self.sol_dir = Path("WIP_ks1998endolaborfrischadjcost_ss.npz")
        if self.sol_dir.exists():
            self._load_state()

    def _load_state(self):
        data = np.load(self.sol_dir)
        self.K = float(data["K"])
        self.supply_L = float(data["supplyL"])
        self.current_dist = data["currentdist"]
        self.mpol_c = data["mpolc"]
        self.mpol_aprime = data["mpolaprime"]
        self.mlambda = data["mlambda"]
        print("→ Resumed from saved state.")

    def solve(self):
        t0 = time.time()
        error = 10.0
        iter_ge = 1
        while error > self.tol.ge:
            r, w = self._compute_prices()
            self._update_policy_functions(r, w)
            self._update_stationary_distribution()
            error = self._update_aggregates(r, w)
            if self.verbose and ((iter_ge - 1) % 200 == 0 or error <= self.tol.ge):
                self._report(iter_ge, error, r, w)
            iter_ge += 1
        self._save_final(t0)

    def _compute_prices(self):
        r = self.p.alpha * (self.K / self.supply_L) ** (self.p.alpha - 1) - self.p.delta
        w = (1 - self.p.alpha) * (self.K / self.supply_L) ** self.p.alpha
        return r, w

    def _update_policy_functions(self, r, w):
        mexpectation = np.zeros_like(self.mgrida)
        for izprime, zprime in enumerate(self.grid_z):
            interp_func = interp1d(self.grid_a, self.mpol_aprime[:, izprime], kind="linear", fill_value="extrapolate")
            mpol_aprimeprime = interp_func(self.mpol_aprime)
            psi2 = -(self.p.mu / 2) * ((mpol_aprimeprime / self.mpol_aprime) ** 2 - 1)
            mprime = (
                (1 + r) * self.mpol_aprime
                - (self.p.mu / 2) * ((mpol_aprimeprime - self.mpol_aprime) / self.mpol_aprime) ** 2 * self.mpol_aprime
                - mpol_aprimeprime
            ) / (w * zprime)
            nprime = (-self.p.eta * mprime + np.sqrt((self.p.eta * mprime) ** 2 + 4 * self.p.eta)) / (2 * self.p.eta)
            cprime = (
                w * zprime * nprime
                + (1 + r) * self.mpol_aprime
                - mpol_aprimeprime
                - (self.p.mu / 2) * ((mpol_aprimeprime - self.mpol_aprime) / self.mpol_aprime) ** 2 * self.mpol_aprime
            )
            cprime = np.maximum(cprime, 1e-10)
            muprime = 1.0 / cprime
            mexpectation += np.repeat(self.trans_z[:, izprime][None, :], self.p.num_grid_a, axis=0) * (
                1 + r - psi2
            ) * muprime
        mexpectation *= self.p.beta
        c = (1 + self.p.mu * (self.mpol_aprime - self.mgrida) / self.mgrida) / (mexpectation + self.mlambda)
        mpol_n = (w * self.mgridz / (self.p.eta * c)) ** self.p.frisch
        self.mlambda_new = (
            (1 + self.p.mu * (self.mpol_aprime - self.mgrida) / self.mgrida)
            / (
                w * self.mgridz * mpol_n
                + (1 + r) * self.mgrida
                - (self.p.mu / 2) * ((self.mpol_aprime - self.mgrida) / self.mgrida) ** 2 * self.mgrida
                - self.mpol_aprime
            )
            - mexpectation
        )
        self.mpol_aprime_new = (
            w * self.mgridz * mpol_n
            + (1 + r) * self.grid_a[:, None]
            - c
            - (self.p.mu / 2) * ((self.mpol_aprime - self.mgrida) / self.mgrida) ** 2 * self.mgrida
        )
        # Borrowing constraint
        self.mlambda_new[self.mpol_aprime_new > self.p.grid_a_min] = 0.0
        self.mpol_aprime_new = np.where(self.mpol_aprime_new <= self.p.grid_a_min, self.p.grid_a_min, self.mpol_aprime_new)
        self.mpol_c = c
        self.mpol_n = mpol_n

    def _update_stationary_distribution(self):
        # Eigenvector method (sparse)
        na, nz = self.p.num_grid_a, self.p.num_grid_z
        n_states = na * nz
        rows, cols, data = [], [], []
        for idx in range(n_states):
            ia = idx % na
            iz = idx // na
            nexta = self.mpol_aprime_new[ia, iz]
            ub = np.sum(self.grid_a < nexta)
            ub = np.clip(ub, 1, na - 1)
            lb = ub - 1
            weightlb = (self.grid_a[ub] - nexta) / (self.grid_a[ub] - self.grid_a[lb])
            weightlb = np.clip(weightlb, 0.0, 1.0)
            weightub = 1.0 - weightlb
            for izp in range(nz):
                target_lb = ia_index(lb, izp, na)
                target_ub = ia_index(ub, izp, na)
                rows.extend([idx, idx])
                cols.extend([target_lb, target_ub])
                data.extend([weightlb * self.trans_z[iz, izp], weightub * self.trans_z[iz, izp]])
        P_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
        eigval, eigvec = eigs(P_sparse.T, k=1, which="LM")
        eigvec = np.real(eigvec[:, 0])
        eigvec /= eigvec.sum()
        self.current_dist = eigvec.reshape(nz, na).T
        self.current_dist = np.maximum(self.current_dist, 0.0)
    
    def _update_distribution(self):
        # Update the current distribution based on the new policy function
        na, nz = self.p.num_grid_a, self.p.num_grid_z
        new_dist = np.zeros_like(self.current_dist)
        # Vectorized update using numpy broadcasting
        ia = np.arange(na)[:, None]
        iz = np.arange(nz)[None, :]
        nexta = self.mpol_aprime_new
        ub = np.sum(self.grid_a[None, :] < nexta[..., None], axis=2)
        ub = np.clip(ub, 1, na - 1)
        lb = ub - 1
        grid_a_lb = self.grid_a[lb]
        grid_a_ub = self.grid_a[ub]
        weightlb = (grid_a_ub - nexta) / (grid_a_ub - grid_a_lb)
        weightlb = np.clip(weightlb, 0.0, 1.0)
        weightub = 1.0 - weightlb

        # Transition probabilities: shape (nz, nz)
        trans = self.trans_z

        # For each (ia, iz), distribute mass to (lb, izp) and (ub, izp)
        new_dist = np.zeros_like(self.current_dist)
        for izp in range(nz):
            # lb and ub indices for all (ia, iz)
            np.add.at(new_dist[:, izp], lb, self.current_dist * weightlb * trans[iz, izp])
            np.add.at(new_dist[:, izp], ub, self.current_dist * weightub * trans[iz, izp])

        return new_dist

    def _update_aggregates(self, r, w):
        marginal_dist_a = self.current_dist.sum(axis=1)
        endo_K = np.dot(self.grid_a, marginal_dist_a)
        Lambda = np.sum(self.mlambda * self.current_dist)
        endo_Lambda = np.sum(self.mlambda_new * self.current_dist)
        endo_supply_L = np.sum(self.current_dist * ((w * self.mgridz / (self.p.eta * self.mpol_c)) ** self.p.frisch) * self.mgridz)
        error = np.mean(np.abs([endo_K - self.K, endo_supply_L - self.supply_L, Lambda - endo_Lambda]))
        self.K = self.weights.old1 * self.K + (1 - self.weights.old1) * endo_K
        self.supply_L = self.weights.old2 * self.supply_L + (1 - self.weights.old2) * endo_supply_L
        self.mlambda = self.weights.old3 * self.mlambda + (1 - self.weights.old3) * self.mlambda_new
        self.mpol_aprime = self.weights.old4 * self.mpol_aprime + (1 - self.weights.old4) * self.mpol_aprime_new
        return error

    def _report(self, iter_ge, error, r, w):
        print(f"\n─ Iteration {iter_ge} ──────────────────────────────")
        print(f"max error           : {error: .3e}")
        print(f"capital rent (r)    : {r: .12f}")
        print(f"wage (w)            : {w: .12f}")
        print(f"aggregate capital   : {self.K: .12f}")
        print(f"aggregate labor     : {self.supply_L: .12f}")
        self._plot_distribution()
        self._plot_policy()
        self._save_state()

    def _plot_distribution(self):
        plt.figure(figsize=(7, 5))
        for z_idx in range(self.p.num_grid_z):
            plt.plot(self.grid_a, self.current_dist[:, z_idx], label=f"z_{z_idx+1}")
        plt.legend()
        plt.title("Stationary wealth distribution by productivity state")
        plt.tight_layout()
        plt.savefig("dist_ss_hardcoded.jpg")
        plt.close()

    def _plot_policy(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.grid_a, self.mpol_aprime[:, 0], label="Lowest z")
        plt.plot(self.grid_a, self.mpol_aprime[:, -1], label="Highest z")
        plt.legend(loc="lower right")
        plt.title("Savings policy")
        plt.tight_layout()
        plt.savefig("policy_hardcoded.jpg")
        plt.close()

    def _save_state(self):
        np.savez(
            self.sol_dir,
            K=self.K,
            supplyL=self.supply_L,
            currentdist=self.current_dist,
            mpolc=self.mpol_c,
            mpolaprime=self.mpol_aprime,
            mlambda=self.mlambda,
            grid_a=self.grid_a,
            grid_z=self.grid_z,
            mgridz=self.mgridz,
            mgrida=self.mgrida,
        )
        print("(WIP state saved.)")

    def _save_final(self, t0):
        final_dir = Path("ks1998endolaborfrischadjcost_ss.npz")
        np.savez(
            final_dir,
            K=self.K,
            supplyL=self.supply_L,
            currentdist=self.current_dist,
            mpolc=self.mpol_c,
            mpolaprime=self.mpol_aprime,
            mlambda=self.mlambda,
            grid_a=self.grid_a,
            grid_z=self.grid_z,
            mgridz=self.mgridz,
            mgrida=self.mgrida,
        )
        print("\n✓ Stationary equilibrium solved & saved →", final_dir)
        print("Runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))

if __name__ == "__main__":
    params = ModelParams()
    tol = Tolerances()
    weights = Weights()
    model = KrusellSmithModelSteadyState(params, tol, weights, verbose=True)
    model.solve()
