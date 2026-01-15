import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
import os

# Add the current directory to path to import solve_redu_inter_h
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_redu_inter_h import solve_redu_inter_h


def test_pde_2d_periodic_schemes_h(gamma, M):
    """
    Test 2D PDE: D_x^gamma u + D_y^gamma u + u = f with periodic boundary conditions
    Source term: f(x,y) = sin(2pi x) sin(2pi y)
    
    Input:
        gamma: fractional derivative order (e.g., 0.5, 1.0, 1.5)
        M: number of grid points per direction (x_0, x_1, ..., x_{M-1})
           covering [0, 1) × [0, 1) with periodicity
           - Can be a scalar (single test) or a vector/array (multiple tests)
    """
    
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
         X, Y, u_exact_2D, u_num_2D, err_2D, use_S, S_err) = \
            solve_single_pde_2d_periodic(gamma, M, c_int)
        
        # Plot exact solution
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_exact_2D, cmap='viridis', shade=True, antialiased=True)
        fig1.colorbar(surf1, ax=ax1)
        ax1.set_xlabel('x', fontsize=14)
        ax1.set_ylabel('y', fontsize=14)
        ax1.set_zlabel('u_{exact}(x,y)', fontsize=14)
        ax1.set_title(f'Exact Solution: u(x,y) = sin(2π x) sin(2π y), γ={gamma:.2f}',
                     fontsize=14, fontweight='bold')
        ax1.tick_params(labelsize=12)
        
        # Plot numerical solution
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u_num_2D, cmap='viridis', shade=True, antialiased=True)
        ax2.set_zlim(-1, 1)
        fig2.colorbar(surf2, ax=ax2)
        ax2.set_xlabel('x', fontsize=14)
        ax2.set_ylabel('y', fontsize=14)
        ax2.set_zlabel('u_{num}(x,y)', fontsize=14)
        ax2.set_title(f'Numerical Solution, γ={gamma:.2f}, N={M}',
                     fontsize=14, fontweight='bold')
        ax2.tick_params(labelsize=12)
        
        # Plot error
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf3 = ax3.plot_surface(X, Y, err_2D, cmap='viridis', shade=True, antialiased=True)
        fig3.colorbar(surf3, ax=ax3)
        ax3.set_xlabel('x', fontsize=14)
        ax3.set_ylabel('y', fontsize=14)
        ax3.set_zlabel('Error', fontsize=14)
        ax3.set_title(f'Error: u_num - u_exact, γ={gamma:.2f}, N={M}',
                     fontsize=14, fontweight='bold')
        ax3.tick_params(labelsize=12)
        
        # Output results in formatted table
        print('\n=== Results ===')
        print(f'Testing 2D PDE (gamma={gamma:.2f}) on f(x,y)=sin(2πx)sin(2πy) with periodic boundary conditions')
        print(f'S operator 1D error (sin(2πx)): {S_err:.2e}')
        if use_S:
            print('Using S operator form: L = S_x + S_y + I, u = L \\ f')
        else:
            print('Using A/B form: (A2 + B2) * u = A2 * f (S operator error too large)')
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
        
        # Run tests for each M
        for i in range(n_tests):
            M = M_array[i]
            h = 1.0 / M
            h_array[i] = h
            
            # Get coefficients using h-based auto-selection for each M
            c_int, _, _, _ = solve_redu_inter_h(gamma, h, None, False)
            
            (L1_errors[i], L2_errors[i], Lmax_errors[i],
             relative_L1_errors[i], relative_L2_errors[i], relative_Lmax_errors[i],
             X_plot, Y_plot, u_exact_plot, u_num_plot, err_plot, _, _) = \
                solve_single_pde_2d_periodic(gamma, M, c_int)
            
            # Plot for last test
            if i == n_tests - 1:
                # Plot exact solution
                fig1 = plt.figure(figsize=(8, 6))
                ax1 = fig1.add_subplot(111, projection='3d')
                surf1 = ax1.plot_surface(X_plot, Y_plot, u_exact_plot, cmap='viridis',
                                        shade=True, antialiased=True)
                fig1.colorbar(surf1, ax=ax1)
                ax1.set_xlabel('x', fontsize=14)
                ax1.set_ylabel('y', fontsize=14)
                ax1.set_zlabel('u_{exact}(x,y)', fontsize=14)
                ax1.set_title(f'Exact Solution: u(x,y) = sin(2π x) sin(2π y), γ={gamma:.2f}',
                             fontsize=14, fontweight='bold')
                ax1.tick_params(labelsize=12)
                
                # Plot numerical solution
                fig2 = plt.figure(figsize=(8, 6))
                ax2 = fig2.add_subplot(111, projection='3d')
                surf2 = ax2.plot_surface(X_plot, Y_plot, u_num_plot, cmap='viridis',
                                        shade=True, antialiased=True)
                ax2.set_zlim(-1, 1)
                fig2.colorbar(surf2, ax=ax2)
                ax2.set_xlabel('x', fontsize=14)
                ax2.set_ylabel('y', fontsize=14)
                ax2.set_zlabel('u_{num}(x,y)', fontsize=14)
                ax2.set_title(f'Numerical Solution, γ={gamma:.2f}, M={M}',
                             fontsize=14, fontweight='bold')
                ax2.tick_params(labelsize=12)
                
                # Plot error
                fig3 = plt.figure(figsize=(8, 6))
                ax3 = fig3.add_subplot(111, projection='3d')
                surf3 = ax3.plot_surface(X_plot, Y_plot, err_plot, cmap='viridis',
                                        shade=True, antialiased=True)
                fig3.colorbar(surf3, ax=ax3)
                ax3.set_xlabel('x', fontsize=14)
                ax3.set_ylabel('y', fontsize=14)
                ax3.set_zlabel('Error', fontsize=14)
                ax3.set_title(f'Error: u_num - u_exact, γ={gamma:.2f}, M={M}',
                             fontsize=14, fontweight='bold')
                ax3.tick_params(labelsize=12)
        
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


def solve_single_pde_2d_periodic(gamma, M, c_int):
    """
    Solve single 2D PDE with periodic boundary conditions for Weyl derivative.
    
    Input:
        gamma: fractional derivative order
        M: number of grid points per direction
        c_int: coefficients [a1, a2, a3, a5, a6, a7, alpha, beta]
    
    Output:
        L1_err, L2_err, Lmax_err: absolute errors
        rel_L1, rel_L2, rel_Lmax: relative errors
        X, Y: meshgrid coordinates
        u_exact_2D, u_num_2D, err_2D: 2D solution arrays
        use_S_form: whether S operator form was used
        S_operator_error: S operator error
    """
    
    h = 1.0 / M
    x = np.arange(M) * h
    y = np.arange(M) * h
    
    # Build 1D matrices A_1D and B_1D
    A_1D = sp.lil_matrix((M, M))
    B_1D = sp.lil_matrix((M, M))
    
    for j in range(M):
        A_row, B_row = build_row_internal_periodic(M, h, gamma, c_int, j)
        A_1D[j, :] = A_row
        B_1D[j, :] = B_row
    
    # Convert to CSR format for efficient operations
    A_1D = A_1D.tocsr()
    B_1D = B_1D.tocsr()
    
    # Verify S operator accuracy for sin(2πx)
    # For Weyl derivative: D^gamma[sin(2πkx)] = (2πk)^gamma * sin(2πkx + πγ/2)
    k_wave = 1
    u_test = np.sin(2 * np.pi * k_wave * x)
    d_exact = (2 * np.pi * k_wave) ** gamma * np.sin(2 * np.pi * k_wave * x + np.pi * gamma / 2)
    
    # Compute S * u = A^{-1} * (B * u)
    B_times_u = B_1D @ u_test
    d_num = spsolve(A_1D, B_times_u)
    S_operator_error = np.linalg.norm(d_num - d_exact, ord=np.inf)
    
    # Use S form if error is small enough, otherwise use A/B form
    use_S_form = (S_operator_error < 1e-5)
    
    if use_S_form:
        # Use S operator form: L = S_x + S_y + I, u = L \ f
        # Compute S_1D = A_1D^{-1} * B_1D by solving for each column
        S_1D_dense = np.zeros((M, M))
        B_1D_dense = B_1D.toarray()
        for col in range(M):
            S_1D_dense[:, col] = spsolve(A_1D, B_1D_dense[:, col])
        S_1D = S_1D_dense
        I_M = sp.eye(M)
        I_2D = sp.eye(M * M)
        L_2D = sp.kron(S_1D, I_M) + sp.kron(I_M, S_1D) + I_2D
        
        X, Y = np.meshgrid(x, y)
        # For Weyl derivative: D_x^gamma u + D_y^gamma u + u = f
        # where u = sin(2πx)sin(2πy), so:
        # D_x^gamma u = (2π)^gamma * sin(2πx + πγ/2) * sin(2πy)
        # D_y^gamma u = (2π)^gamma * sin(2πx) * sin(2πy + πγ/2)
        # f = (1 + 2*(2π)^gamma*cos(πγ/2)) * sin(2πx)sin(2πy) + (2π)^gamma*sin(πγ/2) * sin(2π(x+y))
        coeff1 = 1 + 2 * (2 * np.pi * k_wave) ** gamma * np.cos(np.pi * gamma / 2)
        coeff2 = (2 * np.pi * k_wave) ** gamma * np.sin(np.pi * gamma / 2)
        f_2D = coeff1 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) + \
               coeff2 * np.sin(2 * np.pi * (X + Y))
        f_vec = f_2D.flatten()
        
        if M <= 64:
            cond_L = np.linalg.cond(L_2D.todense())
            if cond_L > 1e10:
                warnings.warn(f'System matrix L is ill-conditioned (cond={cond_L:e}); results may be unstable',
                             UserWarning)
        
        u_vec = spsolve(L_2D, f_vec)
    else:
        # Use A/B form: (A2 + B2) * u = A2 * f
        I_M = sp.eye(M)
        A2 = sp.kron(A_1D, I_M) + sp.kron(I_M, A_1D)
        B2 = sp.kron(B_1D, I_M) + sp.kron(I_M, B_1D)
        
        X, Y = np.meshgrid(x, y)
        # For Weyl derivative: D_x^gamma u + D_y^gamma u + u = f
        coeff1 = 1 + 2 * (2 * np.pi * k_wave) ** gamma * np.cos(np.pi * gamma / 2)
        coeff2 = (2 * np.pi * k_wave) ** gamma * np.sin(np.pi * gamma / 2)
        f_2D = coeff1 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) + \
               coeff2 * np.sin(2 * np.pi * (X + Y))
        f_vec = f_2D.flatten()
        
        rhs2 = A2 @ f_vec
        system_matrix = A2 + B2
        
        if M <= 64:
            cond_sys = np.linalg.cond(system_matrix.todense())
            if cond_sys > 1e10:
                warnings.warn(f'System matrix (A2 + B2) is ill-conditioned (cond={cond_sys:e}); results may be unstable',
                             UserWarning)
        
        u_vec = spsolve(system_matrix, rhs2)
    
    u_exact_2D = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u_exact_vec = u_exact_2D.flatten()
    
    u_num_2D = u_vec.reshape(M, M)
    err_2D = u_num_2D - u_exact_2D
    err_vec = err_2D.flatten()
    
    L1_err = h ** 2 * np.sum(np.abs(err_vec))
    L2_err = h * np.sqrt(np.sum(err_vec ** 2))
    Lmax_err = np.max(np.abs(err_vec))
    
    u_exact_L1_norm = h ** 2 * np.sum(np.abs(u_exact_vec))
    u_exact_L2_norm = h * np.sqrt(np.sum(u_exact_vec ** 2))
    u_exact_Lmax_norm = np.max(np.abs(u_exact_vec))
    
    eps_val = np.finfo(float).eps
    rel_L1 = (L1_err / u_exact_L1_norm) if u_exact_L1_norm > eps_val else L1_err
    rel_L2 = (L2_err / u_exact_L2_norm) if u_exact_L2_norm > eps_val else L2_err
    rel_Lmax = (Lmax_err / u_exact_Lmax_norm) if u_exact_Lmax_norm > eps_val else Lmax_err
    
    return (L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax,
            X, Y, u_exact_2D, u_num_2D, err_2D, use_S_form, S_operator_error)


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
    # a4 = -(a1 + a2 + a3 + a5 + a6 + a7)
    
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
    M = 32
    test_pde_2d_periodic_schemes_h(gamma, M)
    plt.show()
