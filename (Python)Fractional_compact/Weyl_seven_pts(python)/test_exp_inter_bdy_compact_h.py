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


def test_exp_inter_bdy_compact_h(gamma, N_array=None):
    """
    Test the compact scheme for the Weyl derivative
    on [-20, 1] using u(x) = exp(lambda x) with compact boundary.
    
    Input:
        gamma: fractional derivative order (0 < gamma <= 2)
        N_array: scalar or vector of interval numbers (h = L/N, num_points = N+1)
                 (default: 64)
    """
    
    if N_array is None:
        N_array = 64
    
    N_array = np.asarray(N_array)
    if N_array.ndim == 0:
        N_array = np.array([N_array])
        single_test = True
    else:
        single_test = False
    
    L = 21  # Domain length (from -20 to 1)
    
    if single_test:
        N = int(N_array[0])
        h = L / N
        
        result = compute_single_test(gamma, N, L, return_all=True)
        L1_error, L2_error, max_error, _, _, _, x, exact_deriv, d_num, _, point_errors = result
        
        print(f'Testing Weyl Derivative (gamma={gamma:.2f}) on u(x)=e^{{λx}} with N={N} intervals ({N+1} points)')
        print(f'L1={L1_error:.4e}, L2={L2_error:.4e}, L∞={max_error:.4e}')
        
        # Print errors at specific boundary and interior points
        if point_errors is not None and len(point_errors['indices']) > 0:
            print()
            print('=' * 100)
            print('                    ERROR ANALYSIS AT BOUNDARY AND INTERIOR POINTS')
            print('=' * 100)
            print(f'{"Index":<8} {"x":<12} {"Exact":<15} {"Numerical":<15} {"Error":<15} {"Rel Error":<15}')
            print('-' * 100)
            for i in range(len(point_errors['indices'])):
                idx = point_errors['indices'][i]
                print(f'{idx:<8} {x[idx]:<12.6f} {exact_deriv[idx]:<15.6e} {d_num[idx]:<15.6e} '
                      f'{point_errors["errors"][i]:<15.6e} {point_errors["rel_errors"][i]:<15.6e}')
            print('=' * 100)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, exact_deriv, 'b-', linewidth=2, label='Exact Derivative')
        plt.plot(x, d_num, 'r--o', linewidth=2.5, markersize=4,
                markevery=max(1, len(x) // 5), markerfacecolor='r', label='Numerical Derivative')
        plt.xlabel('x')
        plt.ylabel('D^γ u(x)')
        plt.title(f'Boundary Test: γ={gamma:.2f}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim([-20, 1])
        plt.tight_layout()
        plt.show()
        
    else:
        # Convergence test: multiple N, output error table
        n = len(N_array)
        L1_err = np.zeros(n)
        L2_err = np.zeros(n)
        Lmax_err = np.zeros(n)
        relL1_err = np.zeros(n)
        relL2_err = np.zeros(n)
        relLmax_err = np.zeros(n)
        h_arr = np.zeros(n)
        
        for i in range(n):
            N = int(N_array[i])
            h = L / N
            h_arr[i] = h
            (L1_err[i], L2_err[i], Lmax_err[i],
             relL1_err[i], relL2_err[i], relLmax_err[i]) = \
                compute_single_test(gamma, N, L, return_all=False)
        
        # Output results table
        print()
        print(f'{"N":<6} {"h":<10} {"L1":<12} {"L2":<12} {"Lmax":<12} {"Rel L1":<12} {"Rel L2":<12} {"Rel Lmax":<12}')
        for i in range(n):
            print(f'{int(N_array[i]):<6} {h_arr[i]:<10.3e} {L1_err[i]:<12.4e} {L2_err[i]:<12.4e} {Lmax_err[i]:<12.4e} '
                  f'{relL1_err[i]:<12.4e} {relL2_err[i]:<12.4e} {relLmax_err[i]:<12.4e}')
        
        # Final comparison plot (largest N)
        N_plot = int(N_array[-1])
        h = L / N_plot
        result = compute_single_test(gamma, N_plot, L, return_all=True)
        _, _, _, _, _, _, x, exact_deriv, d_num, _, point_errors = result
        
        # Print errors at specific boundary and interior points for convergence test
        if point_errors is not None and len(point_errors['indices']) > 0:
            print()
            print('=' * 100)
            print(f'                    ERROR ANALYSIS AT BOUNDARY AND INTERIOR POINTS (N={N_plot})')
            print('=' * 100)
            print(f'{"Index":<8} {"x":<12} {"Exact":<15} {"Numerical":<15} {"Error":<15} {"Rel Error":<15}')
            print('-' * 100)
            for i in range(len(point_errors['indices'])):
                idx = point_errors['indices'][i]
                print(f'{idx:<8} {x[idx]:<12.6f} {exact_deriv[idx]:<15.6e} {d_num[idx]:<15.6e} '
                      f'{point_errors["errors"][i]:<15.6e} {point_errors["rel_errors"][i]:<15.6e}')
            print('=' * 100)
        
        N_fine = 1000
        x_fine = np.linspace(-20, 1, N_fine + 1)
        _, exact_fine = compute_exp_test_function(gamma, x_fine)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_fine, exact_fine, 'b-', linewidth=2.5, label='Exact Derivative')
        plt.plot(x, d_num, 'r--o', linewidth=2.5, markersize=4,
                markevery=max(1, len(x) // 20), markerfacecolor='r', label='Numerical Derivative')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('Fractional Derivative Value', fontsize=14)
        plt.xlim([-20, 1])
        plt.title(f'Exact vs Numerical Derivative: γ={gamma:.2f}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tick_params(labelsize=12)
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
        return_all: if True, return all outputs including x, exact_deriv, d_num, abs_err, point_errors
    
    Output:
        If return_all=False:
            L1, L2, Linf, relL1, relL2, relLinf
        If return_all=True:
            L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err, point_errors
    """
    h = L / N
    
    # Check if h <= 1 (required by solve_redu_inter_h)
    if h > 1:
        raise ValueError(f'Grid spacing h = {h:.6f} exceeds 1. Please use N >= L (N >= {L}) to ensure h <= 1.')
    
    num_points = N + 1  # Total grid points: x_0 to x_N
    x = -20 + np.arange(num_points) * h  # num_points points on [-20, 1]
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
    # c_iN_2 returns [a1, a2, a3, a5, a6, a7, alpha, beta] (a4 eliminated)
    # Note: Variable names match MATLAB code (a4_N_1 actually stores a5, a5_N_1 stores a4)
    if c_iN_2 is not None:
        a1_N_1 = c_iN_2[0]  # a1
        a2_N_1 = c_iN_2[1]  # a2
        a3_N_1 = c_iN_2[2]  # a3
        a4_N_1 = c_iN_2[3]  # a5 (stored as a4_N_1 to match MATLAB naming)
        a6_N_1 = c_iN_2[4]  # a6
        a7_N_1 = c_iN_2[5]  # a7
        alphaN_1 = c_iN_2[6]
        betaN_1 = c_iN_2[7]
        a5_N_1 = -(a1_N_1 + a2_N_1 + a3_N_1 + a4_N_1 + a6_N_1 + a7_N_1)  # a4 (eliminated)
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
    
    # Compute errors at specific boundary and interior points
    point_errors = None
    if return_all:
        point_errors = {
            'indices': [],
            'errors': [],
            'rel_errors': [],
            'labels': []
        }
        
        if N >= 6:
            # Left boundary points: indices 0, 1, 2 (x_0, x_1, x_2)
            left_indices = [0, 1, 2]
            left_labels = ['x_0 (left)', 'x_1 (left)', 'x_2 (left)']
            
            # Right boundary points: indices num_points-3, num_points-2, num_points-1
            # (x_{N-2}, x_{N-1}, x_N)
            right_indices = [num_points - 3, num_points - 2, num_points - 1]
            right_labels = ['x_{N-2} (right)', 'x_{N-1} (right)', 'x_N (right)']
            
            # Interior point: middle point
            interior_idx = (num_points - 1) // 2
            interior_label = f'x_{interior_idx} (interior)'
            
            # Combine all indices: 6 boundary points + 1 interior point = 7 points total
            all_indices = left_indices + [interior_idx] + right_indices
            all_labels = left_labels + [interior_label] + right_labels
            
            # Compute errors
            for i, idx in enumerate(all_indices):
                if 0 <= idx < num_points:
                    point_errors['indices'].append(idx)
                    point_errors['errors'].append(abs_err[idx])
                    point_errors['rel_errors'].append(abs_err[idx] / (np.abs(exact_deriv[idx]) + eps_val))
                    point_errors['labels'].append(all_labels[i])
        elif N >= 3:
            # If N < 6, we still have left boundary points 0, 1, 2 and right boundary points
            left_indices = [0, 1, 2]
            left_labels = ['x_0 (left)', 'x_1 (left)', 'x_2 (left)']
            
            right_indices = [num_points - 2, num_points - 1]
            right_labels = [f'x_{N-1} (right)', f'x_{N} (right)']
            
            # Interior point if possible
            if num_points >= 5:
                interior_idx = (num_points - 1) // 2
                interior_label = f'x_{interior_idx} (interior)'
                all_indices = left_indices + [interior_idx] + right_indices
                all_labels = left_labels + [interior_label] + right_labels
            else:
                all_indices = left_indices + right_indices
                all_labels = left_labels + right_labels
            
            # Compute errors
            for i, idx in enumerate(all_indices):
                if 0 <= idx < num_points:
                    point_errors['indices'].append(idx)
                    point_errors['errors'].append(abs_err[idx])
                    point_errors['rel_errors'].append(abs_err[idx] / (np.abs(exact_deriv[idx]) + eps_val))
                    point_errors['labels'].append(all_labels[i])
        elif num_points >= 2:
            # Very few points: just use available boundary points
            available_indices = [0, num_points - 1]
            available_labels = ['x_0 (left)', f'x_{N} (right)']
            
            for i, idx in enumerate(available_indices):
                point_errors['indices'].append(idx)
                point_errors['errors'].append(abs_err[idx])
                point_errors['rel_errors'].append(abs_err[idx] / (np.abs(exact_deriv[idx]) + eps_val))
                point_errors['labels'].append(available_labels[i])
    
    if return_all:
        return L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err, point_errors
    else:
        return L1, L2, Linf, relL1, relL2, relLinf


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    N_array = 64
    test_exp_inter_bdy_compact_h(gamma, N_array)
    
    # Convergence test
    # N_array = np.array([32, 64, 128, 256])
    # test_exp_inter_bdy_compact_h(gamma, N_array)
