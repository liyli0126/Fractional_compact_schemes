import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add the current directory to path to import boundary scheme functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_redu_inter_h import solve_redu_inter_h
from solve_redu_inter_bdy_i1_h import solve_redu_inter_bdy_i1_h
from solve_redu_inter_bdy_i2_h import solve_redu_inter_bdy_i2_h
from solve_redu_inter_bdy_i3_h import solve_redu_inter_bdy_i3_h
from solve_redu_inter_bdy_iN_h import solve_redu_inter_bdy_iN_h
from solve_redu_inter_bdy_iN_1_h import solve_redu_inter_bdy_iN_1_h
from solve_redu_inter_bdy_iN1_h import solve_redu_inter_bdy_iN1_h


def test_exp_inter_bdy_diff_gamma_compact_h(N=None, gamma_array=None):
    """
    Scan gamma for a fixed N to test the compact non-periodic scheme
    on [-10, 1] using u(x) = exp(lambda x).
    
    Input:
        N: fixed number of intervals (h = L/N, num_points = N+1) (default: 64)
        gamma_array: vector of gamma values in (0,2] (default: 0.1:0.1:2.0)
    """
    
    if N is None:
        N = 64
    if gamma_array is None:
        gamma_array = np.arange(0.1, 2.1, 0.1)
    
    N = int(np.round(N))
    if N <= 0:
        raise ValueError('N must be a positive integer.')
    
    L = 11  # Domain length: [-10, 1]
    h = L / N
    
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
        (L1_err[i], L2_err[i], Lmax_err[i],
         relL1_err[i], relL2_err[i], relLmax_err[i]) = \
            compute_single_test(gamma, N, L)
    
    # |Lmax| vs gamma
    valid = ~np.isnan(Lmax_err)
    
    plt.figure(figsize=(8, 4))
    plt.plot(gamma_array[valid], Lmax_err[valid], 'ro-',
             linewidth=1.5, markerfacecolor='r')
    plt.xlabel('γ', fontsize=12)
    plt.ylabel('L_∞ Error (Absolute)', fontsize=12)
    plt.title(f'Absolute L_∞ Error vs γ (N = {N})', fontsize=12)
    plt.grid(True)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    
    # Relative Lmax vs gamma
    plt.figure(figsize=(8, 4))
    plt.plot(gamma_array[valid], relLmax_err[valid], 'bs-',
             linewidth=1.5, markerfacecolor='b')
    plt.xlabel('γ', fontsize=12)
    plt.ylabel('L_∞ Error (Relative)', fontsize=12)
    plt.title(f'Relative L_∞ Error vs γ (N = {N})', fontsize=12)
    plt.grid(True)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    
    # gamma-error table
    print()
    print('=' * 70)
    print(f'         ERROR vs GAMMA (N = {N}, h = {h:.3e})')
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
            
            (_, _, _, _, _, _, x, exact_deriv, d_num, _) = \
                compute_single_test(gamma, N, L, return_all=True)
            
            plt.figure(figsize=(8, 4))
            plt.plot(x, exact_deriv, 'b-', linewidth=2, label='Exact')
            plt.plot(x, d_num, 'r--o', markersize=3, markerfacecolor='r',
                     markevery=max(1, len(x) // 20), label='Numerical')
            plt.title(f'γ = {gamma:.1f} (N = {N})')
            plt.xlabel('x')
            plt.ylabel('D^γ u(x)')
            plt.legend(loc='best')
            plt.grid(True)
            plt.xlim([-10, 1])
            plt.tight_layout()
    
    plt.show()


def compute_exp_test_function(gamma, x):
    """
    Compute test function and exact derivative for exponential function.
    
    Input:
        gamma: fractional derivative order
        x: grid points
    
    Output:
        u: test function u(x) = exp(lambda * x)
        exact_deriv: exact derivative D^gamma u = lambda^gamma * exp(lambda * x)
    """
    lambda_val = 1
    u = np.exp(lambda_val * x)
    exact_deriv = lambda_val ** gamma * np.exp(lambda_val * x)
    return u, exact_deriv


def compute_single_test(gamma, N, L, return_all=False):
    """
    Compute single test with error metrics for non-periodic boundary scheme.
    
    Input:
        gamma: fractional derivative order
        N: number of intervals
        L: domain length
        return_all: if True, return all outputs including x, exact_deriv, d_num, abs_err
    
    Output:
        If return_all=False:
            L1, L2, Linf, relL1, relL2, relLinf
        If return_all=True:
            L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err
    """
    h = L / N
    num_points = N + 1  # Total grid points: x_0 to x_N
    x = -10 + np.arange(num_points) * h  # num_points points: x[0]=x_0, x[N]=x_N
    
    u, exact_deriv = compute_exp_test_function(gamma, x)
    
    # Coefficients
    c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
    c_i0, _, _, _ = solve_redu_inter_bdy_i1_h(gamma, h, None, False)
    c_i1, _, _, _ = solve_redu_inter_bdy_i2_h(gamma, h, None, False)
    c_i2, _, _, _ = solve_redu_inter_bdy_i3_h(gamma, h, None, False)
    c_iN_1, _, _, _ = solve_redu_inter_bdy_iN_h(gamma, h, None, False)
    if N >= 6:
        c_iN_2, _, _, _ = solve_redu_inter_bdy_iN_1_h(gamma, h, None, False)
    else:
        c_iN_2 = None
    c_iN, _, _, _ = solve_redu_inter_bdy_iN1_h(gamma, h, None, False)
    
    # Internal 7-point coefficients
    a1i = c_int[0]
    a2i = c_int[1]
    a3i = c_int[2]
    a5i = c_int[3]
    a6i = c_int[4]
    a7i = c_int[5]
    a4i = -(a1i + a2i + a3i + a5i + a6i + a7i)
    alphai = c_int[6]
    betai = c_int[7]
    
    # Left boundary coefficients (7-point) - 0-based indexing
    # idx=0 (x_0)
    a2_1 = c_i0[0]
    a3_1 = c_i0[1]
    a4_1 = c_i0[2]
    a5_1 = c_i0[3]
    a6_1 = c_i0[4]
    a7_1 = c_i0[5]
    alpha1 = c_i0[6]
    beta1 = c_i0[7]
    a1_1 = -(a2_1 + a3_1 + a4_1 + a5_1 + a6_1 + a7_1)
    
    # idx=1 (x_1)
    a1_2 = c_i1[0]
    a3_2 = c_i1[1]
    a4_2 = c_i1[2]
    a5_2 = c_i1[3]
    a6_2 = c_i1[4]
    a7_2 = c_i1[5]
    alpha2 = c_i1[6]
    beta2 = c_i1[7]
    a2_2 = -(a1_2 + a3_2 + a4_2 + a5_2 + a6_2 + a7_2)
    
    # idx=2 (x_2)
    a1_3 = c_i2[0]
    a2_3 = c_i2[1]
    a4_3 = c_i2[2]
    a5_3 = c_i2[3]
    a6_3 = c_i2[4]
    a7_3 = c_i2[5]
    alpha3 = c_i2[6]
    beta3 = c_i2[7]
    a3_3 = -(a1_3 + a2_3 + a4_3 + a5_3 + a6_3 + a7_3)
    
    # Right boundary coefficients (7-point) - 0-based indexing
    # idx=N (x_N)
    a1_N1 = c_iN[0]
    a2_N1 = c_iN[1]
    a3_N1 = c_iN[2]
    a4_N1 = c_iN[3]
    a5_N1 = c_iN[4]
    a6_N1 = c_iN[5]
    alphaN1 = c_iN[6]
    betaN1 = c_iN[7]
    a7_N1 = -(a1_N1 + a2_N1 + a3_N1 + a4_N1 + a5_N1 + a6_N1)
    
    # idx=N-1 (x_{N-1}) - 7-point format
    # c_iN_1 returns [a1, a2, a3, a4, a5, a7, alpha, beta] (a6 eliminated)
    a1_N = c_iN_1[0]
    a2_N = c_iN_1[1]
    a3_N = c_iN_1[2]
    a4_N = c_iN_1[3]
    a5_N = c_iN_1[4]
    a7_N = c_iN_1[5]
    alphaN = c_iN_1[6]
    betaN = c_iN_1[7]
    a6_N = -(a1_N + a2_N + a3_N + a4_N + a5_N + a7_N)
    
    # idx=N-2 (x_{N-2}) - 7-point format
    if c_iN_2 is not None:
        a1_N_1 = c_iN_2[0]
        a2_N_1 = c_iN_2[1]
        a3_N_1 = c_iN_2[2]
        a4_N_1 = c_iN_2[3]
        a6_N_1 = c_iN_2[4]
        a7_N_1 = c_iN_2[5]
        alphaN_1 = c_iN_2[6]
        betaN_1 = c_iN_2[7]
        a5_N_1 = -(a1_N_1 + a2_N_1 + a3_N_1 + a4_N_1 + a6_N_1 + a7_N_1)
    else:
        # Fallback for small N
        a1_N_1 = a2_N_1 = a3_N_1 = a4_N_1 = a6_N_1 = a7_N_1 = 0
        alphaN_1 = betaN_1 = 0
        a5_N_1 = 0
    
    # RHS matrix B
    B = sp.lil_matrix((num_points, num_points))
    for j in range(num_points):
        if j == 0:
            # idx=0 (x_0) - 7-point format
            B[j, j] = a1_1
            if num_points >= 2:
                B[j, j + 1] = a2_1
            if num_points >= 3:
                B[j, j + 2] = a3_1
            if num_points >= 4:
                B[j, j + 3] = a4_1
            if num_points >= 5:
                B[j, j + 4] = a5_1
            if num_points >= 6:
                B[j, j + 5] = a6_1
            if num_points >= 7:
                B[j, j + 6] = a7_1
        elif j == 1:
            # idx=1 (x_1)
            if num_points >= 2:
                B[j, j - 1] = a1_2
            B[j, j] = a2_2
            if num_points >= 3:
                B[j, j + 1] = a3_2
            if num_points >= 4:
                B[j, j + 2] = a4_2
            if num_points >= 5:
                B[j, j + 3] = a5_2
            if num_points >= 6:
                B[j, j + 4] = a6_2
            if num_points >= 7:
                B[j, j + 5] = a7_2
        elif j == 2:
            # idx=2 (x_2)
            if num_points >= 2:
                B[j, j - 2] = a1_3
            if num_points >= 2:
                B[j, j - 1] = a2_3
            B[j, j] = a3_3
            if num_points >= 4:
                B[j, j + 1] = a4_3
            if num_points >= 5:
                B[j, j + 2] = a5_3
            if num_points >= 6:
                B[j, j + 3] = a6_3
            if num_points >= 7:
                B[j, j + 4] = a7_3
        elif j == num_points - 1:
            # idx=N (x_N) - 7-point format
            if num_points >= 7:
                B[j, j - 6] = a1_N1
            if num_points >= 6:
                B[j, j - 5] = a2_N1
            if num_points >= 5:
                B[j, j - 4] = a3_N1
            if num_points >= 4:
                B[j, j - 3] = a4_N1
            if num_points >= 3:
                B[j, j - 2] = a5_N1
            if num_points >= 2:
                B[j, j - 1] = a6_N1
            B[j, j] = a7_N1
        elif j == num_points - 2:
            # idx=N-1 (x_{N-1}) - 7-point format
            if num_points >= 7:
                B[j, j - 5] = a1_N
            if num_points >= 6:
                B[j, j - 4] = a2_N
            if num_points >= 5:
                B[j, j - 3] = a3_N
            if num_points >= 4:
                B[j, j - 2] = a4_N
            if num_points >= 3:
                B[j, j - 1] = a5_N
            B[j, j] = a6_N
            if num_points >= 2:
                B[j, j + 1] = a7_N
        elif j == num_points - 3:
            # idx=N-2 (x_{N-2}) - 7-point format
            if num_points >= 7:
                B[j, j - 4] = a1_N_1
            if num_points >= 6:
                B[j, j - 3] = a2_N_1
            if num_points >= 5:
                B[j, j - 2] = a3_N_1
            if num_points >= 4:
                B[j, j - 1] = a4_N_1
            B[j, j] = a5_N_1
            if num_points >= 3:
                B[j, j + 1] = a6_N_1
            if num_points >= 2:
                B[j, j + 2] = a7_N_1
        else:
            # Internal points (idx=3 to idx=N-4)
            if j >= 3:
                B[j, j - 3] = a1i
            if j >= 2:
                B[j, j - 2] = a2i
            if j >= 1:
                B[j, j - 1] = a3i
            B[j, j] = a4i
            if j <= num_points - 2:
                B[j, j + 1] = a5i
            if j <= num_points - 3:
                B[j, j + 2] = a6i
            if j <= num_points - 4:
                B[j, j + 3] = a7i
    
    RHS = (1.0 / (h ** gamma)) * (B @ u)
    
    # LHS matrix A
    A = sp.lil_matrix((num_points, num_points))
    for j in range(num_points):
        if j == 0:
            # idx=0 (x_0)
            A[j, j] = 1
            if num_points >= 2:
                A[j, j + 1] = alpha1
            if num_points >= 3:
                A[j, j + 2] = beta1
        elif j == 1:
            # idx=1 (x_1)
            if num_points >= 2:
                A[j, j - 1] = alpha2
            A[j, j] = 1
            if num_points >= 3:
                A[j, j + 1] = alpha2
            if num_points >= 4:
                A[j, j + 2] = beta2
        elif j == 2:
            # idx=2 (x_2)
            if num_points >= 3:
                A[j, j - 2] = beta3
            if num_points >= 2:
                A[j, j - 1] = alpha3
            A[j, j] = 1
            if num_points >= 4:
                A[j, j + 1] = alpha3
            if num_points >= 5:
                A[j, j + 2] = beta3
        elif j == num_points - 1:
            # idx=N (x_N)
            if num_points >= 3:
                A[j, j - 2] = betaN1
            if num_points >= 2:
                A[j, j - 1] = alphaN1
            A[j, j] = 1
        elif j == num_points - 2:
            # idx=N-1 (x_{N-1})
            if num_points >= 3:
                A[j, j - 2] = betaN
            if num_points >= 2:
                A[j, j - 1] = alphaN
            A[j, j] = 1
            if j + 1 < num_points:
                A[j, j + 1] = alphaN
        elif j == num_points - 3:
            # idx=N-2 (x_{N-2}) - boundary format
            if num_points >= 4:
                A[j, j - 2] = betaN_1
            if num_points >= 3:
                A[j, j - 1] = alphaN_1
            A[j, j] = 1
            if j + 1 < num_points:
                A[j, j + 1] = alphaN_1
            if j + 2 < num_points:
                A[j, j + 2] = betaN_1
        else:
            # Internal points (idx=3 to idx=N-4)
            A[j, j - 2] = betai
            A[j, j - 1] = alphai
        A[j, j] = 1
            A[j, j + 1] = alphai
            A[j, j + 2] = betai
    
    # Convert to CSR format for efficient solving
    A = A.tocsr()
    B = B.tocsr()
    
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
    N = 64
    gamma_array = np.arange(0.1, 2.1, 0.1)
    test_exp_inter_bdy_diff_gamma_compact_h(N, gamma_array)
