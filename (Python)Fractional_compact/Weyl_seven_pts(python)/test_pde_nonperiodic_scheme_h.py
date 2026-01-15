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


def test_pde_with_nonperiodic_schemes_h(gamma, k_wave=None, M=None):
    """
    Test 1D PDE: D^gamma u + u = f with periodic boundary conditions for Weyl derivative.
    
    Input:
        gamma: fractional derivative order (e.g., 0.5, 1.0, 1.5)
        k_wave: wave number for test function u(x) = sin(2π k_wave x) (default: 1)
        M: number of grid points (x_0, x_1, ..., x_{M-1}) covering [0, 1) with periodicity
           - Can be a scalar (single test) or a vector/array (multiple tests)
    """
    
    # Set default k_wave if not provided
    if k_wave is None:
        k_wave = 1
    
    if M is None:
        raise ValueError('M must be provided')
    
    # Determine if single test or multiple tests
    M_array = np.asarray(M)
    if M_array.ndim == 0:  # scalar
        M_array = np.array([int(np.round(M_array))])
        single_test = True
    else:  # array
        M_array = np.round(M_array.flatten()).astype(int)
        single_test = False
        # Ensure all M values are positive integers
        if np.any(M_array <= 0) or np.any(np.abs(M_array - np.round(M_array)) > 1e-10):
            raise ValueError('All values in M_array must be positive integers')
    
    if single_test:
        # Single test: formatted output
        M = M_array[0]
        h = 1.0 / M
        
        # Get coefficients using h-based auto-selection
        c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
        
        (L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax,
         x, u_exact, u_num) = solve_single_pde_periodic(gamma, M, c_int, k_wave)
        
        # Plot exact vs numerical solution
        plt.figure(figsize=(8, 6))
        plt.plot(x, u_exact, 'b-', linewidth=2.5, label='Exact Solution')
        plt.plot(x, u_num, 'r--o', linewidth=2.5, markersize=4,
                markevery=max(1, len(x) // 20), markerfacecolor='r',
                label='Numerical Solution')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('u(x)', fontsize=14)
        plt.xlim([0, 1])
        plt.title(f'Exact vs Numerical Solution: u(x)=sin(2π*{k_wave:.2f}*x), γ={gamma:.2f}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        
        # Plot error
        err = u_num - u_exact
        plt.figure(figsize=(8, 6))
        plt.plot(x, err, 'r-', linewidth=2.5)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('Error (u_num - u_exact)', fontsize=14)
        plt.xlim([0, 1])
        plt.title(f'Error: u_num - u_exact, γ={gamma:.2f}, M={M}',
                 fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        
        # Output results in formatted table
        print('\n=== Results ===')
        print(f'Testing PDE (gamma={gamma:.2f}) on u(x)=sin(2*pi*{k_wave:.2f}*x) with periodic boundary conditions')
        print('=' * 110)
        print(' ' * 40 + 'RESULTS TABLE')
        print('=' * 110)
        print(f'{"M":<6} {"h":<12} {"L1 Error":<12} {"L2 Error":<12} {"L_max Error":<12} '
              f'{"Rel L1":<12} {"Rel L2":<12} {"Rel L_max":<12}')
        print('-' * 110)
        print(f'{M:<6} {h:<12.6e} {L1_err:<12.4e} {L2_err:<12.4e} {Lmax_err:<12.4e} '
              f'{rel_L1:<12.4e} {rel_L2:<12.4e} {rel_Lmax:<12.4e}')
        print('=' * 110)
        
    else:
        # Multiple tests: run for each M and create table
        n_tests = len(M_array)
        h_array = np.zeros(n_tests)
        L1_errors = np.zeros(n_tests)
        L2_errors = np.zeros(n_tests)
        Lmax_errors = np.zeros(n_tests)
        relative_L1_errors = np.zeros(n_tests)
        relative_L2_errors = np.zeros(n_tests)
        relative_Lmax_errors = np.zeros(n_tests)
        
        print(f'Testing PDE (gamma={gamma:.2f}) on u(x)=sin(2*pi*{k_wave:.2f}*x) with periodic boundary conditions')
        print('Grid points: ', end='')
        print(' '.join(map(str, M_array)))
        print()
        
        # Run tests for each M
        for i in range(n_tests):
            M = M_array[i]
            h = 1.0 / M
            h_array[i] = h
            
            # Get coefficients using h-based auto-selection for each M
            c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
            
            (L1_errors[i], L2_errors[i], Lmax_errors[i],
             relative_L1_errors[i], relative_L2_errors[i], relative_Lmax_errors[i],
             x_plot, u_exact_plot, u_num_plot) = solve_single_pde_periodic(gamma, M, c_int, k_wave)
            
            # Plot for last test
            if i == n_tests - 1:
                # Plot exact vs numerical solution
                plt.figure(figsize=(8, 6))
                plt.plot(x_plot, u_exact_plot, 'b-', linewidth=2.5, label='Exact Solution')
                plt.plot(x_plot, u_num_plot, 'r--o', linewidth=2.5, markersize=4,
                        markevery=max(1, len(x_plot) // 20), markerfacecolor='r',
                        label='Numerical Solution')
                plt.xlabel('x', fontsize=14)
                plt.ylabel('u(x)', fontsize=14)
                plt.xlim([0, 1])
                plt.title(f'Exact vs Numerical Solution: u(x)=sin(2π*{k_wave:.2f}*x), γ={gamma:.2f}, N={M}',
                         fontsize=14, fontweight='bold')
                plt.legend(loc='best', fontsize=12)
                plt.grid(True)
                plt.tick_params(labelsize=12)
                plt.tight_layout()
                
                # Plot error
                err_plot = u_num_plot - u_exact_plot
                plt.figure(figsize=(8, 6))
                plt.plot(x_plot, err_plot, 'r-', linewidth=2.5)
                plt.xlabel('x', fontsize=14)
                plt.ylabel('Error (u_num - u_exact)', fontsize=14)
                plt.xlim([0, 1])
                plt.title(f'Error: u_num - u_exact, γ={gamma:.2f}, N={M}',
                         fontsize=14, fontweight='bold')
                plt.grid(True)
                plt.tick_params(labelsize=12)
                plt.tight_layout()
        
        # Output results table
        print('=' * 110)
        print(' ' * 40 + 'RESULTS TABLE')
        print('=' * 110)
        print(f'{"M":<6} {"h":<12} {"L1 Error":<12} {"L2 Error":<12} {"L_max Error":<12} '
              f'{"Rel L1":<12} {"Rel L2":<12} {"Rel L_max":<12}')
        print('-' * 110)
        for i in range(n_tests):
            print(f'{M_array[i]:<6} {h_array[i]:<12.6e} {L1_errors[i]:<12.4e} {L2_errors[i]:<12.4e} '
                  f'{Lmax_errors[i]:<12.4e} {relative_L1_errors[i]:<12.4e} '
                  f'{relative_L2_errors[i]:<12.4e} {relative_Lmax_errors[i]:<12.4e}')
        print('=' * 110)


def solve_single_pde_periodic(gamma, M, c_int, k_wave=1):
    """
    Solve single 1D PDE with periodic boundary conditions for Weyl derivative.
    
    PDE: D^gamma u + u = f
    where f = D^gamma u_exact + u_exact
    and u_exact = sin(2π k_wave x)
    
    Input:
        gamma: fractional derivative order
        M: number of grid points
        c_int: coefficients [a1, a2, a3, a5, a6, a7, alpha, beta]
        k_wave: wave number (default: 1)
    
    Output:
        L1_err, L2_err, Lmax_err: absolute errors
        rel_L1, rel_L2, rel_Lmax: relative errors
        x: grid points
        u_exact, u_num: solution arrays
    """
    
    h = 1.0 / M
    # Grid points: x = (0:M-1) * h to cover [0, 1) (periodic, exclude x=1)
    x = np.arange(M) * h
    
    # Build global LHS matrix A (for d) and RHS matrix B (for u) with sparse matrix
    A_mat = sp.lil_matrix((M, M))  # compact: usually 3-5 diagonals
    B_mat = sp.lil_matrix((M, M))  # FD: up to 7-point stencil
    
    # Exact RHS of PDE:
    # For Weyl derivative: D^gamma[sin(2πkx)] = (2πk)^gamma * sin(2πkx + πγ/2)
    u_exact = np.sin(2 * np.pi * k_wave * x)
    d_exact = (2 * np.pi * k_wave) ** gamma * np.sin(2 * np.pi * k_wave * x + np.pi * gamma / 2)
    f_exact = d_exact + u_exact
    
    # Fill rows using periodic indexing
    for j in range(M):  # Python index (0-based)
        # All points use internal scheme with periodic indexing
        A_row, B_row = build_row_internal_periodic(M, h, gamma, c_int, j)
        A_mat[j, :] = A_row
        B_mat[j, :] = B_row
    
    # Convert to CSR format for efficient operations
    A_mat = A_mat.tocsr()
    B_mat = B_mat.tocsr()
    
    # Solve (B + A) * u = A * f
    rhs = A_mat @ f_exact
    
    # Transfer to dense matrix, and the system (B_mat + A_mat) * u = rhs is well-conditioned
    B_mat_full = B_mat.toarray()
    A_mat_full = A_mat.toarray()
    B_mat_modified = B_mat_full + A_mat_full
    
    # Check condition number and rank of modified B_mat
    cond_B = np.linalg.cond(B_mat_modified)
    
    # Solve (B_mat + A_mat) * u = rhs
    if cond_B > 1e10:
        warnings.warn(f'(B_mat + A_mat) is ill-conditioned (cond={cond_B:e}, M={M}); results may be unstable',
                     UserWarning)
    
    u_num = np.linalg.solve(B_mat_modified, rhs)
    
    err = u_num - u_exact
    
    # Compute errors
    L1_err = h * np.sum(np.abs(err))
    L2_err = np.sqrt(h * np.sum(err ** 2))
    Lmax_err = np.max(np.abs(err))
    
    # Compute relative errors
    u_exact_L1_norm = h * np.sum(np.abs(u_exact))
    u_exact_L2_norm = np.sqrt(h * np.sum(u_exact ** 2))
    u_exact_Lmax_norm = np.max(np.abs(u_exact))
    
    # Calculate relative errors (avoid division by zero)
    eps_val = np.finfo(float).eps
    if u_exact_L1_norm > eps_val:
        rel_L1 = L1_err / u_exact_L1_norm
    else:
        rel_L1 = L1_err  # If exact solution is zero, use absolute error
    
    if u_exact_L2_norm > eps_val:
        rel_L2 = L2_err / u_exact_L2_norm
    else:
        rel_L2 = L2_err  # If exact solution is zero, use absolute error
    
    if u_exact_Lmax_norm > eps_val:
        rel_Lmax = Lmax_err / u_exact_Lmax_norm
    else:
        rel_Lmax = Lmax_err  # If exact solution is zero, use absolute error
    
    return (L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax,
            x, u_exact, u_num)


def build_row_internal_periodic(M, h, gamma, coeffs, j):
    """
    Build row for internal points with periodic boundary conditions.
    Weyl derivative format: non-symmetric 7-point scheme with 8 parameters
    
    Input:
        M: number of grid points
        h: grid spacing
        gamma: fractional derivative order
        coeffs: [a1, a2, a3, a5, a6, a7, alpha, beta] from solve_redu_inter_h
        j: row index (0-based in Python)
    
    Output:
        A_row: sparse row vector for LHS matrix A
        B_row: sparse row vector for RHS matrix B
    """
    # coeffs = [a1, a2, a3, a5, a6, a7, alpha, beta] from solve_redu_inter_h
    # Weyl derivative: non-symmetric format
    # LHS: d_j + alpha*(d_{j-1}+d_{j+1}) + beta*(d_{j-2}+d_{j+2})
    # RHS: (1/h^gamma) * [a1*u_{j-3} + a2*u_{j-2} + a3*u_{j-1} + a4*u_j + a5*u_{j+1} + a6*u_{j+2} + a7*u_{j+3}]
    # where a4 = -(a1 + a2 + a3 + a5 + a6 + a7)
    
    a1 = coeffs[0]
    a2 = coeffs[1]
    a3 = coeffs[2]
    a5 = coeffs[3]
    a6 = coeffs[4]
    a7 = coeffs[5]
    alpha = coeffs[6]
    beta = coeffs[7]
    a4 = -(a1 + a2 + a3 + a5 + a6 + a7)
    
    A_row = sp.lil_matrix((1, M))
    B_row = sp.lil_matrix((1, M))
    
    # Use periodic indexing to find neighboring points
    # Note: j is 0-based in Python, so we need to adjust the MATLAB formula
    # MATLAB: mod(j-4, M) + 1 for j-3 (1-based)
    # Python: (j - 3) % M for j-3 (0-based)
    idx_m3 = (j - 3) % M  # Periodic: j-3
    idx_m2 = (j - 2) % M  # Periodic: j-2
    idx_m1 = (j - 1) % M  # Periodic: j-1
    idx_p1 = (j + 1) % M  # Periodic: j+1
    idx_p2 = (j + 2) % M  # Periodic: j+2
    idx_p3 = (j + 3) % M  # Periodic: j+3
    
    # LHS: d_j + alpha*(d_{j-1}+d_{j+1}) + beta*(d_{j-2}+d_{j+2})
    A_row[0, j] = 1
    A_row[0, idx_m1] = alpha
    A_row[0, idx_p1] = alpha
    A_row[0, idx_m2] = beta
    A_row[0, idx_p2] = beta
    
    # RHS: (1/h^gamma) * [a1*u_{j-3} + a2*u_{j-2} + a3*u_{j-1} + a4*u_j + a5*u_{j+1} + a6*u_{j+2} + a7*u_{j+3}]
    # Non-symmetric format for Weyl derivative
    scale = 1.0 / (h ** gamma)
    B_row[0, j] = a4 * scale
    B_row[0, idx_m3] = a1 * scale
    B_row[0, idx_m2] = a2 * scale
    B_row[0, idx_m1] = a3 * scale
    B_row[0, idx_p1] = a5 * scale
    B_row[0, idx_p2] = a6 * scale
    B_row[0, idx_p3] = a7 * scale
    
    return A_row, B_row


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    k_wave = 1
    M = 32
    test_pde_with_nonperiodic_schemes_h(gamma, k_wave, M)
    plt.show()
