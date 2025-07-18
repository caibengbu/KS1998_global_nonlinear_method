# Placeholder for the backward-consistency simulation
# Krusell and Smith (1998) DSGE model with endogenous labor and adjustment costs

import KS_SS_hardcoded 
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.signal import detrend
import bisect
import numba as nb
from scipy.interpolate import interp1d 

@nb.njit(parallel=True)
def accumulate_mass_numba(grid_a, trans_z, mpolaprime, moving_dist):
    na, nz, nt = mpolaprime.shape
    moving_dist_new = np.zeros((na, nz, nt))

    for it in nb.prange(nt):
        for iz in nb.prange(nz):
            for ia in nb.prange(na):
                nexta = mpolaprime[ia, iz, it]
                # Faster and supported: searchsorted
                ub = np.searchsorted(grid_a, nexta)
                ub = min(max(ub, 1), na - 1)
                lb = ub - 1

                a_lb = grid_a[lb]
                a_ub = grid_a[ub]
                denom = a_ub - a_lb
                weightlb = (a_ub - nexta) / denom
                weightlb = min(max(weightlb, 0.0), 1.0)
                weightub = 1.0 - weightlb

                mass = moving_dist[ia, iz, it]
                for jz in nb.prange(nz):
                    moving_dist_new[lb, jz, it] += weightlb * mass * trans_z[iz, jz]
                    moving_dist_new[ub, jz, it] += weightub * mass * trans_z[iz, jz]

    return moving_dist_new

@dataclass
class ShockParams:
    A_vals: NDArray = field(default_factory=lambda: np.array([0.99, 1.01])) # Two-state Markov process for technology shocks
    A_trans: NDArray = field(default_factory=lambda: np.array([[0.875, 0.125], [0.125, 0.875]])) # Transition matrix for the Markov process
    T: int = 2001
    burnin: int = 500
    seed: int = 1337

@dataclass
class AlgoParams:
    verbose: bool = True
    tol: float = 1e-6
    w1: float = 0.95
    w2: float = 0.95
    w3: float = 0.95
    w4: float = 0.95

class KrusellSmithSimulation:
    def __init__(self, ss: KS_SS_hardcoded.KrusellSmithModelSteadyState, shock: ShockParams, params: AlgoParams):
        self.params = params
        self.shock = shock
        self.ss = ss
        self._init_sim_path()
        self._init_equilibrium_objects()

    def _init_sim_path(self):
        np.random.seed(self.shock.seed)
        self.pathlength = self.shock.T + self.shock.burnin
        self.tsimpath = np.zeros(self.pathlength, dtype=int)
        self.tsimpath[0] = 0
        for t in range(1, self.pathlength):
            self.tsimpath[t] = np.random.choice(len(self.shock.A_vals), p=self.shock.A_trans[self.tsimpath[t-1]])
        self.iA = self.tsimpath

    def _init_equilibrium_objects(self):
        # aggregates 
        self.tK = self.ss.K + 1e-4 * np.random.randn(self.pathlength)
        self.tL = self.ss.supply_L * np.ones(self.pathlength)
        self.tY = np.zeros(self.pathlength)
        self.tC = np.zeros(self.pathlength)
        self.tlambda = np.sum(self.ss.mlambda * self.ss.current_dist) * np.ones(self.pathlength)

        # declare equilibrium objects
        self.mpolc = np.tile(self.ss.mpol_c[:,:,np.newaxis], (1,1,self.pathlength))  # Consumption policy
        self.mpolaprime = np.tile(self.ss.mpol_aprime[:,:,np.newaxis], (1,1,self.pathlength))
        self.mlambda = np.tile(self.ss.mlambda[:,:,np.newaxis], (1,1,self.pathlength))  # Lagrange multiplier
        self.mpoln = np.tile(self.ss.mpol_n[:,:,np.newaxis], (1,1,self.pathlength))  # Labor policy
        self.moving_dist = np.tile(self.ss.current_dist[:,:,np.newaxis], (1,1,self.pathlength)) # Initial distribution of agents


    def simulate(self):
        """Run the iterative simulation until market clearing conditions converge."""
        alpha = self.ss.p.alpha
        delta = self.ss.p.delta
        pnum_a = len(self.ss.grid_a)
        pnum_z = len(self.ss.grid_z)
        iA = self.iA
        # i_future = np.roll(np.arange(0,self.pathlength), -1) 
        i_trans = np.arange(0, self.pathlength)  # Transition indices
        i_future = np.roll(i_trans, -1)
        # i_future[i_future >= self.pathlength] = self.pathlength - 1  # Ensure we don't go out of bounds
        future_shock = self.tsimpath[i_future]  # Future shock states for each period
        vA = self.shock.A_vals[iA]  # Aggregate shock values
        # probability of realized shocks
        realized_probs = np.array([self.shock.A_trans[iA[t], future_shock[t]] for t in range(self.pathlength)])
        error2 = 1.0  # Initialize error for convergence check

        r = alpha * vA * (self.tK[i_trans] / self.tL) ** (alpha - 1) - delta  # Initial interest rate
        w = (1 - alpha) * vA * (self.tK[i_trans] / self.tL) ** alpha  # Initial wage rate
        tK = self.tK[i_trans]
        while error2 > self.params.tol:
            tKprime = self.tK[i_future]
            mexpectation = np.zeros((pnum_a, pnum_z, self.pathlength))
            for iAprime in range(len(self.shock.A_vals)):
                needs_interpolate = iAprime != future_shock
                Aprime = self.shock.A_vals[iAprime]
                candidate = tK[iA == iAprime]
                candidate_location = np.where(self.tsimpath == iAprime)[0]
                # candidate = candidate[(candidate_location >= self.shock.burnin) & (candidate_location < (self.shock.T + self.shock.burnin))]
                # candidate_location = candidate_location[(candidate_location >= self.shock.burnin) & (candidate_location < (self.shock.T + self.shock.burnin))]
                sorted_idx = np.argsort(candidate)
                candidate = candidate[sorted_idx]
                candidate_location = candidate_location[sorted_idx]

                KHigh_loc = np.sum(candidate[:,None] < tKprime, axis=0)
                KLow_loc = KHigh_loc - 1
                hit_lower_bound = KHigh_loc <= 0
                hit_upper_bound = KHigh_loc >= len(candidate)
                KLow_loc[hit_lower_bound] = 0
                KHigh_loc[hit_lower_bound] = 1
                KLow_loc[hit_upper_bound] = len(candidate) - 2
                KHigh_loc[hit_upper_bound] = len(candidate) - 1
                weight_low = np.zeros_like(KHigh_loc, dtype=float)
                weight_high = np.zeros_like(KHigh_loc, dtype=float)
                weight_low[hit_lower_bound] = 1.0
                weight_high[hit_upper_bound] = 1.0
                weight_low[~hit_lower_bound & ~hit_upper_bound] = ((candidate[KHigh_loc] - tKprime) / (candidate[KHigh_loc] - candidate[KLow_loc]))[~hit_lower_bound & ~hit_upper_bound]
                weight_high[~hit_lower_bound & ~hit_upper_bound] = 1.0 - weight_low[~hit_lower_bound & ~hit_upper_bound]
                
                K2Lprimelow = tKprime/self.tL[candidate_location[KLow_loc]]
                rprimelow = alpha * Aprime * (K2Lprimelow) ** (alpha - 1) - delta
                wprimelow = (1 - alpha) * Aprime * (K2Lprimelow ** alpha)
                K2Lprimehigh = tKprime/self.tL[candidate_location[KHigh_loc]]
                rprimehigh = alpha * Aprime * (K2Lprimehigh) ** (alpha - 1) - delta
                wprimehigh = (1 - alpha) * Aprime * (K2Lprimehigh ** alpha)
                rprime     = weight_low*rprimelow + weight_high*rprimehigh
                wprime     = weight_low*wprimelow + weight_high*wprimehigh

                for izprime in range(pnum_z):
                    zprime = self.ss.grid_z[izprime]
                    mpolaprime_temp = weight_low[None, :] * self.mpolaprime[:, izprime, candidate_location[KLow_loc]] + weight_high[None, :] * self.mpolaprime[:, izprime, candidate_location[KHigh_loc]]
                    mpolaprimeprime = np.zeros((pnum_a, pnum_z, pnum_z, self.pathlength))

                    for t in range(self.pathlength):
                        f = interp1d(self.ss.grid_a, mpolaprime_temp[:,t], kind='linear', fill_value='extrapolate')
                        mpolaprimeprime[:,:,izprime,t] = f(self.mpolaprime[:,:,t]) # mpolaprimeprime(a, z, z', t)

                    psi2 = - self.ss.p.mu / 2 * ((mpolaprimeprime[:,:,izprime,:] / self.mpolaprime) ** 2 - 1.0)
                    mprime = ((1+rprime)*self.mpolaprime
                               - (self.ss.p.mu/2)*((mpolaprimeprime[:,:,izprime,:]-self.mpolaprime)/self.mpolaprime)**2*self.mpolaprime 
                               - mpolaprimeprime[:,:,izprime,:])/(wprime*zprime)
                    nprime = (-self.ss.p.eta*mprime + np.sqrt((self.ss.p.eta*mprime)**2+4*self.ss.p.eta))/(2*self.ss.p.eta)

                    cprime = wprime*zprime*nprime + (1+rprime)*self.mpolaprime - mpolaprimeprime[:,:,izprime,:] - (self.ss.p.mu/2)*((mpolaprimeprime[:,:,izprime,:]-self.mpolaprime)/self.mpolaprime)**2*self.mpolaprime
                    cprime[cprime<=0] = 1e-10

                    muprime = 1/cprime
                    mexpectation = mexpectation + (1+rprime-psi2) * muprime * self.ss.trans_z[:, izprime][None, :, None] * self.shock.A_trans[iA, iAprime]
            
            mexpectation = self.ss.p.beta * mexpectation
            mexpectation = (mexpectation + self.mlambda) / (1 + self.ss.p.mu * (self.mpolaprime - self.ss.grid_a[:, None, None]) / self.ss.grid_a[:, None, None])
            c = 1 / mexpectation
            n = (w[None, None, :] * self.ss.grid_z[None, :, None] / (self.ss.p.eta * c)) ** self.ss.p.frisch
            mlambda_newtemp = 1 / self.mpolc - mexpectation
            mpolaprime_newtemp = w[None, None, :] * self.ss.grid_z[None, :, None] * n + (1 + r[None, None, :]) * self.ss.grid_a[:, None, None] - c - (self.ss.p.mu / 2) * ((self.mpolaprime - self.ss.grid_a[:, None, None]) / self.ss.grid_a[:, None, None]) ** 2 * self.ss.grid_a[:, None, None]
            mlambda_newtemp[mpolaprime_newtemp>self.ss.grid_a.min()] = 0.0
            c = np.where(mpolaprime_newtemp <= self.ss.grid_a.min(), c + mpolaprime_newtemp - self.ss.grid_a.min(), c)
            mpolaprime_newtemp = np.clip(mpolaprime_newtemp, self.ss.grid_a.min(), None)  # Enforce borrowing constraint

            # update
            mpolaprime_new = mpolaprime_newtemp
            mlambda_new = mlambda_newtemp
            mpolc_new = c
            mpoln = n
            moving_dist = self.moving_dist[:, :, i_trans]
            moving_dist_new = accumulate_mass_numba(
                self.ss.grid_a,
                self.ss.trans_z,
                mpolaprime_new,
                moving_dist
            )
            self.moving_dist[:, :, i_future] = moving_dist_new  # Update distribution for the next period
            moving_dist = self.moving_dist[:, :, i_trans]
            tK_new = np.sum(np.sum(self.moving_dist, 1) * self.ss.grid_a[:, None], 0) 
            tL_new = np.einsum('ijt,ijt->t', moving_dist, mpoln * self.ss.grid_z[None, :, None])
            tY_new = vA * tK_new[i_trans] ** alpha * tL_new ** (1 - alpha)
            tC_new = np.einsum('ijt,ijt->t', moving_dist, mpolc_new)
            tlambda_new = np.einsum('ijt,ijt->t', moving_dist, mlambda_new)
            error2 = np.max(np.abs(self.tK - tK_new)) + np.max(np.abs(self.tL - tL_new)) + np.max(np.abs(self.tlambda - tlambda_new))
            if self.params.verbose:
                print(f"Error: {error2:.6f}, K: {np.mean(tK_new):.4f}, L: {np.mean(tL_new):.4f}, Lambda: {np.mean(tlambda_new):.4f}")
                plt.plot(self.tK, label='Capital')
                plt.plot(tK_new, 'k:', label='New Capital')
                plt.legend()
                plt.savefig(Path(__file__).parent / 'capital_convergence.png')
                plt.clf()
            self.tK = self.params.w1 * self.tK + (1-self.params.w1) * tK_new
            self.tL = self.params.w2 * self.tL + (1-self.params.w2) * tL_new
            self.mlambda = self.params.w3 * self.mlambda + (1-self.params.w3) * mlambda_new
            self.mpolaprime = self.params.w4 * self.mpolaprime + (1-self.params.w4) * mpolaprime_new
            


    def report(self):
        print("\nRaw time series stats")
        for label, series in zip(['Output', 'Investment', 'Consumption'], [self.tY, self.tI, self.tC]):
            print(f"{label}: Mean={np.mean(np.log(series)):.4f}, Std={np.std(np.log(series)):.4f}, Skew={skew(np.log(series)):.4f}")

        print("\nHP-filtered stats (approximate using detrend)")
        for label, series in zip(['Output', 'Investment', 'Consumption'], [self.tY, self.tI, self.tC]):
            hp_filtered = detrend(np.log(series), type='constant')
            print(f"{label}: Std={np.std(hp_filtered):.4f}, Skew={skew(hp_filtered):.4f}")




if __name__ == "__main__":
    shock = ShockParams()
    params = KS_SS_hardcoded.ModelParams()
    tol = KS_SS_hardcoded.Tolerances()
    weights = KS_SS_hardcoded.Weights()
    steady_state = KS_SS_hardcoded.KrusellSmithModelSteadyState(params, tol, weights, verbose=True)
    steady_state.solve()
    algo_params = AlgoParams()
    sim = KrusellSmithSimulation(steady_state, shock, algo_params)
    sim.simulate()
    sim.report()
