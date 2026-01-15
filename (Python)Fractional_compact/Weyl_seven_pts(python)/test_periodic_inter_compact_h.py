import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add the current directory to path to import solve_redu_inter_h
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_redu_inter_h import solve_redu_inter_h


def test_periodic_inter_compact_h(gamma, k_wave=None, M_array=None):
    """
    Test the compact scheme for the Weyl derivative
    on periodic domain [0, 1) using u(x) = sin(2*pi*k*x)
    
    Input:
        gamma: fractional derivative order (0 < gamma <= 2)
        k_wave: wave number (default: 1)
        M_array: scalar or vector of grid points (number of points in [0,1)) (default: 64)
    """
    
    if k_wave is None:
        k_wave = 1
    if M_array is None:
        M_array = 64
    
    M_array = np.asarray(M_array)
    if M_array.ndim == 0:  # scalar
        M_array = np.array([M_array])
        single_test = True
    else:  # array
        single_test = False
    
    L = 1  # domain length
    
    if single_test:
        M = M_array[0]
        h = L / M
        c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, True)
        
        (L1_error, L2_error, max_error, _, _, _, 
         x, exact_deriv, d_num, _) = compute_single_test(gamma, k_wave, M, L, c_int, return_all=True)
        
        print(f'Testing Weyl Derivative (gamma={gamma:.2f}) on u(x)=sin(2π*{k_wave:.2f}*x) with M={M} points')
        print(f'L1={L1_error:.4e}, L2={L2_error:.4e}, Lmax={max_error:.4e}')
        
        # Plot (extend to x=1 via periodicity)
        x_ext = np.concatenate([x, [1.0]])
        exact_ext = np.concatenate([exact_deriv, [exact_deriv[0]]])
        d_num_ext = np.concatenate([d_num, [d_num[0]]])
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_ext, exact_ext, 'b-', linewidth=2, label='Exact Derivative')
        plt.plot(x_ext, d_num_ext, 'r--o', linewidth=2.5, markersize=4,
                markevery=max(1, len(x_ext) // 20), markerfacecolor='r', 
                label='Numerical Derivative')
        plt.xlabel('x')
        plt.ylabel('D^γ u(x)')
        plt.title(f'Periodic Test: gamma={gamma:.2f}, k={k_wave:.2f}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim([0, 1])
        plt.tight_layout()
        
    else:
        # Convergence test
        n = len(M_array)
        L1_err = np.zeros(n)
        L2_err = np.zeros(n)
        Linf_err = np.zeros(n)
        relL1_err = np.zeros(n)
        relL2_err = np.zeros(n)
        relLinf_err = np.zeros(n)
        h_arr = np.zeros(n)
        
        print(f'Testing gamma={gamma:.2f}, k={k_wave:.2f}')
        for i in range(n):
            M = M_array[i]
            h = L / M
            h_arr[i] = h
            c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
            (L1_err[i], L2_err[i], Linf_err[i],
             relL1_err[i], relL2_err[i], relLinf_err[i]) = \
                compute_single_test(gamma, k_wave, M, L, c_int, return_all=False)
        
        # Output results table (absolute and relative errors, no rate)
        print('=' * 110)
        print(' ' * 40 + 'RESULTS TABLE')
        print('=' * 110)
        print(f'{"M":<6} {"h":<10} {"L1 Error":<12} {"L2 Error":<12} {"Max Error":<12} '
              f'{"Rel L1":<12} {"Rel L2":<12} {"Rel Max":<12}')
        print('-' * 110)
        for i in range(n):
            print(f'{M_array[i]:<6} {h_arr[i]:<10.3e} {L1_err[i]:<12.4e} {L2_err[i]:<12.4e} '
                  f'{Linf_err[i]:<12.4e} {relL1_err[i]:<12.4e} {relL2_err[i]:<12.4e} '
                  f'{relLinf_err[i]:<12.4e}')
        
        # Final comparison plot (largest M)
        M_plot = M_array[-1]
        h = L / M_plot
        c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
        (_, _, _, _, _, _, x, exact_deriv, d_num, _) = \
            compute_single_test(gamma, k_wave, M_plot, L, c_int, return_all=True)
        
        x_ext = np.concatenate([x, [1.0]])
        exact_ext = np.concatenate([exact_deriv, [exact_deriv[0]]])
        d_num_ext = np.concatenate([d_num, [d_num[0]]])
        
        M_fine = 1000
        x_fine = np.linspace(0, 1, M_fine)
        _, exact_fine = compute_periodic_test_function(gamma, k_wave, x_fine)
        exact_fine[-1] = exact_fine[0]  # enforce periodicity
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_fine, exact_fine, 'b-', linewidth=2.5, label='Exact')
        plt.plot(x_ext, d_num_ext, 'r--o', linewidth=2.5, markersize=4,
                markevery=max(1, len(x_ext) // 20), markerfacecolor='r', 
                label='Numerical')
        plt.xlabel('x')
        plt.ylabel('D^γ u(x)')
        plt.title(f'Exact vs Numerical (M={M_plot}, γ={gamma:.2f})')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim([0, 1])
        plt.tight_layout()


def compute_periodic_test_function(gamma, k_wave, x):
    """
    Compute test function and exact derivative for periodic domain (Weyl derivative).
    
    Input:
        gamma: fractional derivative order
        k_wave: wave number
        x: grid points
    
    Output:
        u: test function u(x) = sin(2π k_wave x)
        exact_deriv: exact Weyl derivative D^gamma u = (2π k_wave)^gamma * sin(2π k_wave x + πγ/2)
    """
    u = np.sin(2 * np.pi * k_wave * x)
    exact_deriv = (2 * np.pi * k_wave) ** gamma * np.sin(2 * np.pi * k_wave * x + np.pi * gamma / 2)
    return u, exact_deriv


def compute_single_test(gamma, k_wave, M, L, c_int, return_all=False):
    """
    Compute single test with error metrics for Weyl derivative.
    
    Input:
        gamma: fractional derivative order
        k_wave: wave number
        M: number of grid points
        L: domain length
        c_int: coefficients [a1, a2, a3, a5, a6, a7, alpha, beta]
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
    
    # Extract coefficients from c_int = [a1, a2, a3, a5, a6, a7, alpha, beta]
    a1 = c_int[0]
    a2 = c_int[1]
    a3 = c_int[2]
    a5 = c_int[3]
    a6 = c_int[4]
    a7 = c_int[5]
    alpha = c_int[6]
    beta = c_int[7]
    a4 = -(a1 + a2 + a3 + a5 + a6 + a7)
    
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
            a5 * u[j_p1] + a6 * u[j_p2] + a7 * u[j_p3]
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
    gamma = 1.5
    k_wave = 1
    M_array = 64
    test_periodic_inter_compact_h(gamma, k_wave, M_array)
    plt.show()
