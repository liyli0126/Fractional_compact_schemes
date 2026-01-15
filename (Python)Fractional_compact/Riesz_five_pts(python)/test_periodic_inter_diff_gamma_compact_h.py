import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add the current directory to path to import solve_redu_inter_five_h
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_redu_inter_five_h import solve_redu_inter_five_h


def test_periodic_inter_diff_gamma_compact_h(M=None, k_wave=None, gamma_array=None):
    """
    Scan gamma for a fixed M to test the compact periodic scheme
    on [0, 1) using u(x) = sin(2*pi*k*x).
    
    Input:
        M: fixed number of grid points in [0,1) (default: 64)
        k_wave: wave number (default: 1)
        gamma_array: vector of gamma values in (0,2] (default: 0.1:0.1:2.0)
    """
    
    if M is None:
        M = 64
    if k_wave is None:
        k_wave = 1
    if gamma_array is None:
        gamma_array = np.arange(0.1, 2.1, 0.1)
    
    M = int(np.round(M))
    if M <= 0:
        raise ValueError('M must be a positive integer.')
    
    L = 1  # domain length
    h = L / M
    
    # Convert gamma_array to 1D array
    gamma_array = np.asarray(gamma_array).flatten()
    n_gamma = len(gamma_array)
    
    # Allocate error arrays
    L1_err = np.zeros(n_gamma)
    L2_err = np.zeros(n_gamma)
    Lmax_err = np.zeros(n_gamma)
    relL1_err = np.zeros(n_gamma)
    relL2_err = np.zeros(n_gamma)
    relLmax_err = np.zeros(n_gamma)
    
    for i in range(n_gamma):
        gamma = gamma_array[i]
        if gamma <= 0 or gamma > 2:
            warnings.warn(f'gamma out of (0,2] range: {gamma:.2f}', UserWarning)
            L1_err[i] = np.nan
            L2_err[i] = np.nan
            Lmax_err[i] = np.nan
            relL1_err[i] = np.nan
            relL2_err[i] = np.nan
            relLmax_err[i] = np.nan
            continue
        
        print(f'  gamma = {gamma:.2f}')
        c_int, _, _ = solve_redu_inter_five_h(gamma, h, None, False)
        (L1_err[i], L2_err[i], Lmax_err[i],
         relL1_err[i], relL2_err[i], relLmax_err[i]) = \
            compute_single_test(gamma, k_wave, M, L, c_int, return_all=False)
    
    # |Lmax| vs gamma
    valid = ~np.isnan(Lmax_err)
    
    plt.figure(figsize=(8, 4))
    plt.plot(gamma_array[valid], Lmax_err[valid], 'ro-',
             linewidth=1.5, markerfacecolor='r')
    plt.xlabel('γ')
    plt.ylabel('L_∞ Error (Absolute)')
    plt.title(f'Absolute L_∞ Error vs γ (M = {M})')
    plt.grid(True)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    
    # Relative Lmax vs gamma
    plt.figure(figsize=(8, 4))
    plt.plot(gamma_array[valid], relLmax_err[valid], 'bs-',
             linewidth=1.5, markerfacecolor='b')
    plt.xlabel('γ')
    plt.ylabel('Relative L_∞ Error')
    plt.title(f'Relative L_∞ Error vs γ (M = {M})')
    plt.grid(True)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    
    # gamma-error table
    print()
    print('=' * 70)
    print(f'         ERROR vs GAMMA (M = {M}, h = {h:.3e})')
    print('=' * 70)
    print(f'{"gamma":<8} {"L1":<12} {"L2":<12} {"Lmax":<12} {"Rel L1":<12} {"Rel L2":<12} {"Rel Lmax":<12}')
    print('-' * 70)
    for i in range(n_gamma):
        if not np.isnan(Lmax_err[i]):
            print(f'{gamma_array[i]:.2f}     {L1_err[i]:<12.4e} {L2_err[i]:<12.4e} {Lmax_err[i]:<12.4e} '
                  f'{relL1_err[i]:<12.4e} {relL2_err[i]:<12.4e} {relLmax_err[i]:<12.4e}')
    
    # exact vs numerical for extreme gamma values
    if n_gamma >= 1:
        gamma_extremes = [gamma_array[0], gamma_array[-1]]
        for k, gamma in enumerate(gamma_extremes):
            if gamma <= 0 or gamma > 2:
                continue
            
            c_int, _, _ = solve_redu_inter_five_h(gamma, h, None, False)
            (_, _, _, _, _, _, x, exact_deriv, d_num, _) = \
                compute_single_test(gamma, k_wave, M, L, c_int, return_all=True)
            
            # extend to x=1 via periodicity for plotting
            x_ext = np.concatenate([x, [1.0]])
            exact_ext = np.concatenate([exact_deriv, [exact_deriv[0]]])
            d_num_ext = np.concatenate([d_num, [d_num[0]]])
            
            plt.figure(figsize=(8, 4))
            plt.plot(x_ext, exact_ext, 'b-', linewidth=2, label='Exact')
            plt.plot(x_ext, d_num_ext, 'r--o', linewidth=2.5, markersize=4,
                    markevery=max(1, len(x_ext) // 20), markerfacecolor='r',
                    label='Numerical')
            plt.title(f'γ = {gamma:.1f} (M = {M}, k = {k_wave:.2f})')
            plt.xlabel('x')
            plt.ylabel('D^γ u(x)')
            plt.legend(loc='best')
            plt.grid(True)
            plt.xlim([0, 1])
            plt.tight_layout()


def compute_periodic_test_function(gamma, k_wave, x):
    """
    Compute test function and exact derivative for periodic domain.
    
    Input:
        gamma: fractional derivative order
        k_wave: wave number
        x: grid points
    
    Output:
        u: test function u(x) = sin(2π k_wave x)
        exact_deriv: exact derivative D^gamma u = -|2π k_wave|^gamma * sin(2π k_wave x)
    """
    u = np.sin(2 * np.pi * k_wave * x)
    exact_deriv = -np.abs(2 * np.pi * k_wave) ** gamma * np.sin(2 * np.pi * k_wave * x)
    return u, exact_deriv


def compute_single_test(gamma, k_wave, M, L, c_int, return_all=False):
    """
    Compute single test with error metrics.
    
    Input:
        gamma: fractional derivative order
        k_wave: wave number
        M: number of grid points
        L: domain length
        c_int: coefficients [a1, a2, a3, alpha, beta]
        return_all: if True, return all outputs including x, exact_deriv, d_num, abs_err
    
    Output:
        If return_all=False:
            L1, L2, Linf, relL1, relL2, relLinf
        If return_all=True:
            L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err
    """
    h = L / M
    x = np.arange(M) * h  # periodic grid: [0, 1)
    u, exact_deriv = compute_periodic_test_function(gamma, k_wave, x)
    
    # Extract coefficients from c_int = [a1, a2, a3, alpha, beta]
    a1 = c_int[0]
    a2 = c_int[1]
    a3 = c_int[2]
    alpha = c_int[3]
    beta = c_int[4]
    a4 = -2 * (a1 + a2 + a3)
    
    # Build RHS
    RHS = np.zeros(M)
    for j in range(M):  # Python 0-based indexing
        # Periodic indexing: MATLAB mod(j-4, M) + 1 -> Python (j - 3) % M
        j_m3 = (j - 3) % M  # j-3
        j_m2 = (j - 2) % M  # j-2
        j_m1 = (j - 1) % M  # j-1
        j_p1 = (j + 1) % M  # j+1
        j_p2 = (j + 2) % M  # j+2
        j_p3 = (j + 3) % M  # j+3
        
        RHS[j] = (1.0 / (h ** gamma)) * (
            a1 * u[j_m3] + a2 * u[j_m2] + a3 * u[j_m1] +
            a4 * u[j] +
            a3 * u[j_p1] + a2 * u[j_p2] + a1 * u[j_p3]
        )
    
    # Build LHS matrix A
    A = sp.lil_matrix((M, M))
    for j in range(M):  # Python 0-based indexing
        A[j, j] = 1
        j_m1 = (j - 1) % M
        j_p1 = (j + 1) % M
        j_m2 = (j - 2) % M
        j_p2 = (j + 2) % M
        A[j, j_m1] = alpha
        A[j, j_p1] = alpha
        A[j, j_m2] = beta
        A[j, j_p2] = beta
    
    # Convert to CSR format for efficient solving
    A = A.tocsr()
    
    # Solve A * d_num = RHS
    d_num = spsolve(A, RHS)
    
    # Errors
    err = d_num - exact_deriv
    abs_err = np.abs(err)
    L1 = h * np.sum(abs_err)
    L2 = np.sqrt(h * np.sum(err ** 2))
    Linf = np.max(abs_err)
    
    exact_L1 = h * np.sum(np.abs(exact_deriv))
    exact_L2 = np.sqrt(h * np.sum(exact_deriv ** 2))
    exact_Linf = np.max(np.abs(exact_deriv))
    
    eps_val = np.finfo(float).eps
    relL1 = L1 / (exact_L1 + eps_val)
    relL2 = L2 / (exact_L2 + eps_val)
    relLinf = Linf / (exact_Linf + eps_val)
    
    if return_all:
        return L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err
    else:
        return L1, L2, Linf, relL1, relL2, relLinf


if __name__ == '__main__':
    # Example usage
    M = 64
    k_wave = 1
    gamma_array = np.arange(0.1, 2.1, 0.1)
    test_periodic_inter_diff_gamma_compact_h(M, k_wave, gamma_array)
    plt.show()
