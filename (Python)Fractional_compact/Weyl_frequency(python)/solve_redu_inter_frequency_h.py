import numpy as np
import matplotlib.pyplot as plt
import warnings


def solve_redu_inter_frequency_h(gamma, N=None, x_points_inter=None, plot_flag=False):
    """
    Solve reduced interpolation problem for Weyl derivative (frequency domain, internal scheme).
    
    Input:
        gamma: fractional derivative order
        N: number of grid points (if provided, h = 1/N and interpolation points will be auto-selected)
        x_points_inter: interpolation points in frequency space [0, pi] (optional)
        plot_flag: if True, plot comparison and error figures (optional, default: False)
    
    Output:
        x_opt: optimized parameters [a1, a2, a3, a5, a6, a7, alpha, beta]
        A: coefficient matrix
        b: right-hand side vector
        L2_error: L^2 error of the approximation (optional)
    """
    
    # Parameter validation and interpolation point selection
    if N is None and x_points_inter is None:
        raise ValueError('At least gamma and N (or x_points_inter) must be provided.')
    
    # Calculate h from N if provided
    if N is not None:
        h = 1.0 / N
    else:
        h = None
    
    # Determine mode: check if second argument is N (scalar) or x_points_inter (vector)
    if N is not None:
        N_array = np.asarray(N)
        if N_array.ndim == 0 and N > 0:  # scalar
            h = 1.0 / N
            # Method: Scale by h to cover frequency range
            w_max = 2 * np.pi * h  # Maximum frequency for one period in [0,1)
            w_max = min(w_max, np.pi)  # Cap at Nyquist frequency
            
            # Select points covering [w_max/8, w_max] with good distribution
            if w_max > np.pi / 32:
                # For reasonable h, use scaled points
                w_points = np.array([
                    max(w_max / 8, np.pi / 128),   # Low frequency (but not too small)
                    w_max / 4,                     # Mid-low frequency
                    w_max / 2,                     # Mid frequency
                    min(w_max, 3 * np.pi / 4),     # High frequency (capped at 3*pi/4 for stability)
                    min(w_max, np.pi)
                ])
            else:
                # For very small h, use fixed small points
                w_points = np.array([np.pi / 128, np.pi / 64, np.pi / 32, np.pi / 16])
            
            # Ensure points are in valid range and sorted
            w_points = np.clip(w_points, np.pi / 256, np.pi)  # Clamp to [pi/256, pi]
            w_points = np.sort(np.unique(w_points))
            
            # Ensure we have at least 4 points (if some are too close, add more)
            if len(w_points) < 4:
                # Add more points if needed
                additional_points = np.linspace(np.pi / 128, min(w_max * 2, np.pi / 4), 4)
                w_points = np.sort(np.unique(np.concatenate([w_points, additional_points])))
                w_points = w_points[:min(4, len(w_points))]  # Take first 4
            
        elif N_array.ndim > 0:  # vector (backward compatibility)
            # N is actually x_points_inter
            x_points_inter = N
            w_points = np.asarray(x_points_inter).flatten()  # Ensure 1D array
            w_points = np.sort(w_points)
            w_points = np.clip(w_points, 0, np.pi)
            h = None
            
            if len(w_points) == 0 or len(w_points) < 4:
                raise ValueError('x_points_inter must have at least 4 points.')
        else:
            raise ValueError('N must be a positive integer or a vector (x_points_inter).')
    else:
        w_points = None
    
    # If both N and x_points_inter are provided, x_points_inter takes precedence
    if x_points_inter is not None and len(x_points_inter) >= 4:
        w_points = np.asarray(x_points_inter).flatten()
        w_points = np.sort(w_points)
        w_points = np.clip(w_points, 0, np.pi)
    
    # Final validation
    if w_points is None or len(w_points) == 0 or len(w_points) < 4:
        raise ValueError('At least 4 interpolation points are required.')
    
    # Check if points are too small (may cause ill-conditioning)
    min_w = np.min(w_points)
    if min_w < np.pi / 128:
        warnings.warn(f'Smallest interpolation point ({min_w:.6f}) is very small. This may cause ill-conditioning.',
                     UserWarning)
    
    n_points = len(w_points)
    n_eq = 2 * n_points  # each point gives real + imag equation
    n_var = 8  # [a1, a2, a3, a5, a6, a7, alpha, beta]
    
    # Precompute trig constants
    s = np.sin(gamma * np.pi / 2)
    c = np.cos(gamma * np.pi / 2)
    
    # Target function (iw)^gamma
    def target_r(x):
        return c * x ** gamma
    
    def target_i(x):
        return s * x ** gamma
    
    # Define the basis functions
    # Real parts
    def phi_r1(x):
        return np.cos(3 * x) - 1  # a1
    
    def phi_r2(x):
        return np.cos(2 * x) - 1  # a2
    
    def phi_r3(x):
        return np.cos(x) - 1  # a3
    
    def phi_r4(x):
        return np.cos(x) - 1  # a5
    
    def phi_r5(x):
        return np.cos(2 * x) - 1  # a6
    
    def phi_r6(x):
        return np.cos(3 * x) - 1  # a7
    
    def phi_r7(x):
        return -2 * c * x ** gamma * np.cos(x)  # alpha
    
    def phi_r8(x):
        return -2 * c * x ** gamma * np.cos(2 * x)  # beta
    
    # Imaginary parts
    def phi_i1(x):
        return -np.sin(3 * x)  # a1
    
    def phi_i2(x):
        return -np.sin(2 * x)  # a2
    
    def phi_i3(x):
        return -np.sin(x)  # a3
    
    def phi_i4(x):
        return np.sin(x)  # a5
    
    def phi_i5(x):
        return np.sin(2 * x)  # a6
    
    def phi_i6(x):
        return np.sin(3 * x)  # a7
    
    def phi_i7(x):
        return -2 * s * x ** gamma * np.cos(x)  # alpha
    
    def phi_i8(x):
        return -2 * s * x ** gamma * np.cos(2 * x)  # beta
    
    phi_r = [phi_r1, phi_r2, phi_r3, phi_r4, phi_r5, phi_r6, phi_r7, phi_r8]
    phi_i = [phi_i1, phi_i2, phi_i3, phi_i4, phi_i5, phi_i6, phi_i7, phi_i8]
    
    # Initialize matrix and vector
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    # Fill in A and b: each point gives two equations (real and imaginary)
    for i in range(n_points):
        xi = w_points[i]
        # Real part
        A[2 * i, :] = [f(xi) for f in phi_r]
        b[2 * i] = target_r(xi)
        # Imag part
        A[2 * i + 1, :] = [f(xi) for f in phi_i]
        b[2 * i + 1] = target_i(xi)
    
    # Solve: use least-squares if overdetermined or underdetermined
    x_opt, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
    
    # Compute L^2 error on fine grid
    # Use w_max if available, otherwise use max(w_points) * 1.1
    if h is not None:
        w_max_for_error = max(min(np.pi, 2 * np.pi * h), np.max(w_points) * 1.1)
    else:
        w_max_for_error = np.max(w_points) * 1.1
    x_fine = np.linspace(0, w_max_for_error, 1000)
    exact_real_fine = target_r(x_fine)
    exact_imag_fine = target_i(x_fine)
    
    num_real_fine = np.zeros_like(x_fine)
    num_imag_fine = np.zeros_like(x_fine)
    for k in range(n_var):
        num_real_fine = num_real_fine + x_opt[k] * phi_r[k](x_fine)
        num_imag_fine = num_imag_fine + x_opt[k] * phi_i[k](x_fine)
    
    # Compute L^2 error: sqrt(integral[0,pi] |exact - numerical|^2 dx)
    # where |exact - numerical|^2 = (real_err)^2 + (imag_err)^2
    err_real_fine = exact_real_fine - num_real_fine
    err_imag_fine = exact_imag_fine - num_imag_fine
    err_mag_sq = err_real_fine ** 2 + err_imag_fine ** 2
    
    # Integrate using trapezoidal rule
    dx_fine = np.pi / (len(x_fine) - 1)
    L2_error = np.sqrt(dx_fine * (np.sum(err_mag_sq) - 0.5 * (err_mag_sq[0] + err_mag_sq[-1])))
    
    # Plot comparison and error figures (only if plot_flag is true)
    if plot_flag:
        # Determine x_max for continuous plotting range [0, max interpolation point]
        x_max = np.max(w_points)
        
        # Create continuous grid for plotting [0, x_max]
        x_plot = np.linspace(0, x_max, 1000)
        
        # Compute exact values on continuous grid
        exact_real = target_r(x_plot)
        exact_imag = target_i(x_plot)
        
        # Compute numerical approximation on continuous grid
        num_real = np.zeros_like(x_plot)
        num_imag = np.zeros_like(x_plot)
        for k in range(n_var):
            num_real = num_real + x_opt[k] * phi_r[k](x_plot)
            num_imag = num_imag + x_opt[k] * phi_i[k](x_plot)
        
        # Compute errors on continuous grid
        err_real = exact_real - num_real
        err_imag = exact_imag - num_imag
        
        # Compute absolute error magnitude: |D_h^{int} - (i*omega)^gamma|
        err_mag = np.sqrt(err_real ** 2 + err_imag ** 2)
        
        # Select 4 collocation points to highlight (take first 4)
        n_colloc = min(4, len(w_points))
        w_colloc = w_points[:n_colloc]
        
        # Compute values at collocation points for highlighting
        exact_real_colloc = target_r(w_colloc)
        exact_imag_colloc = target_i(w_colloc)
        num_real_colloc = np.zeros_like(w_colloc)
        num_imag_colloc = np.zeros_like(w_colloc)
        for k in range(n_var):
            num_real_colloc = num_real_colloc + x_opt[k] * phi_r[k](w_colloc)
            num_imag_colloc = num_imag_colloc + x_opt[k] * phi_i[k](w_colloc)
        err_real_colloc = exact_real_colloc - num_real_colloc
        err_imag_colloc = exact_imag_colloc - num_imag_colloc
        err_mag_colloc = np.sqrt(err_real_colloc ** 2 + err_imag_colloc ** 2)
        
        # Compute relative error: E_h(omega) = |D_h^{int}(omega) - (i*omega)^gamma| / (1 + |omega|^gamma)
        # Denominator: 1 + |omega|^gamma
        denom = 1 + np.abs(x_plot) ** gamma
        E_h = err_mag / denom
        
        # Compute values at ALL collocation points (w_points) for plots 5 and 6
        exact_real_all = target_r(w_points)
        exact_imag_all = target_i(w_points)
        num_real_all = np.zeros_like(w_points)
        num_imag_all = np.zeros_like(w_points)
        for k in range(n_var):
            num_real_all = num_real_all + x_opt[k] * phi_r[k](w_points)
            num_imag_all = num_imag_all + x_opt[k] * phi_i[k](w_points)
        err_real_all = exact_real_all - num_real_all
        err_imag_all = exact_imag_all - num_imag_all
        err_mag_all = np.sqrt(err_real_all ** 2 + err_imag_all ** 2)
        
        # Compute relative error at ALL collocation points
        denom_all = 1 + np.abs(w_points) ** gamma
        E_h_all = err_mag_all / denom_all
        
        # Ensure all values are valid for semilogy plot (avoid 0, NaN, Inf)
        # Add small epsilon to avoid log(0) issues in semilogy plot
        err_mag_positive = err_mag[err_mag > 0]
        if len(err_mag_positive) == 0:
            min_err = 1e-10  # Default minimum
        else:
            min_err = np.min(err_mag_positive)
            if min_err == 0:
                min_err = 1e-10
        eps_val = min(min_err * 1e-3, 1e-12)  # Use 0.1% of minimum or 1e-12, whichever is smaller
        err_mag_all = np.maximum(err_mag_all, eps_val)
        E_h_all = np.maximum(E_h_all, eps_val)
        
        # Debug: Check if all points have valid values
        print('Checking collocation points visibility:')
        print(f'  w_points: ', end='')
        print(' '.join([f'{w:.6f}' for w in w_points]))
        print(f'  err_mag_all: ', end='')
        print(' '.join([f'{e:.2e}' for e in err_mag_all]))
        print(f'  E_h_all: ', end='')
        print(' '.join([f'{e:.2e}' for e in E_h_all]))
        
        # Print L^2 error
        h_display = h if h is not None else 1.0 / 64
        print(f'Internal scheme: L^2 error = {L2_error:.6e} (gamma={gamma:.2f}, h={h_display:.6e})')
        print(f'Number of collocation points: {len(w_points)}')
        print(f'Error magnitudes at collocation points: ', end='')
        print(' '.join([f'{e:.2e}' for e in err_mag_all]))
        
        # Plot 1: Real part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_real, 'b-', linewidth=2, label='Exact Real Part')
        plt.plot(x_plot, num_real, 'r--', linewidth=2, label='Numerical Real Part')
        plt.plot(w_colloc, exact_real_colloc, 'ko', markersize=4, markerfacecolor='k',
                label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Real Part', fontsize=12)
        N_display = N if N is not None else 64
        plt.title(f'Real Part Comparison (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at collocation points (after all settings)
        y_lims = plt.ylim()
        for k in range(n_colloc):
            plt.plot([w_colloc[k], w_colloc[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        # Plot 2: Imaginary part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_imag, 'b-', linewidth=2, label='Exact Imaginary Part')
        plt.plot(x_plot, num_imag, 'r--', linewidth=2, label='Numerical Imaginary Part')
        plt.plot(w_colloc, exact_imag_colloc, 'ko', markersize=4, markerfacecolor='k',
                label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Imaginary Part', fontsize=12)
        plt.title(f'Imaginary Part Comparison (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at collocation points (after all settings)
        y_lims = plt.ylim()
        for k in range(n_colloc):
            plt.plot([w_colloc[k], w_colloc[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        # Plot 3: Real part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_real, 'r-', linewidth=2, label='Real Part Error')
        plt.plot(x_plot, np.zeros_like(x_plot), 'k--', linewidth=1, label='Zero line')
        plt.plot(w_colloc, err_real_colloc, 'ko', markersize=4, markerfacecolor='k',
                label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Real Part Error', fontsize=12)
        plt.title(f'Real Part Error (Exact - Numerical) (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at collocation points (after all settings)
        y_lims = plt.ylim()
        for k in range(n_colloc):
            plt.plot([w_colloc[k], w_colloc[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        # Plot 4: Imaginary part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_imag, 'r-', linewidth=2, label='Imaginary Part Error')
        plt.plot(x_plot, np.zeros_like(x_plot), 'k--', linewidth=1, label='Zero line')
        plt.plot(w_colloc, err_imag_colloc, 'ko', markersize=4, markerfacecolor='k',
                label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Imaginary Part Error', fontsize=12)
        plt.title(f'Imaginary Part Error (Exact - Numerical) (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at collocation points (after all settings)
        y_lims = plt.ylim()
        for k in range(n_colloc):
            plt.plot([w_colloc[k], w_colloc[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        # Plot 5: L^2 absolute error
        plt.figure(figsize=(6, 4))
        plt.semilogy(x_plot, err_mag, 'r-', linewidth=2, label='L^2 Absolute Error')
        # Mark ALL collocation points
        plt.semilogy(w_points, err_mag_all, 'ko', markersize=4, markerfacecolor='k',
                    label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('L^2 Absolute Error', fontsize=12)
        plt.title(f'L^2 Absolute Error (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at ALL collocation points (after all settings, including log scale)
        y_lims = plt.ylim()
        # Ensure y-axis includes all collocation points
        y_min = min([y_lims[0], np.min(err_mag_all)])
        y_max = max([y_lims[1], np.max(err_mag_all)])
        if y_min != y_lims[0] or y_max != y_lims[1]:
            plt.ylim([y_min, y_max])
            y_lims = plt.ylim()  # Update y_lims after adjustment
        for k in range(len(w_points)):
            plt.plot([w_points[k], w_points[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        # Plot 6: Relative error E_h(omega)
        plt.figure(figsize=(6, 4))
        plt.semilogy(x_plot, E_h, 'r-', linewidth=2, label='E_h(ω)')
        # Mark ALL collocation points
        plt.semilogy(w_points, E_h_all, 'ko', markersize=4, markerfacecolor='k',
                    label='Collocation points')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Relative Error E_h(ω)', fontsize=12)
        plt.title(f'Relative Error E_h(ω) (γ={gamma:.2f}, N={N_display})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        # Add gray dashed vertical lines at ALL collocation points (after all settings, including log scale)
        plt.draw()
        y_lims = plt.ylim()
        # Ensure y-axis includes all collocation points
        y_min = min([y_lims[0], np.min(E_h_all)])
        y_max = max([y_lims[1], np.max(E_h_all)])
        if y_min != y_lims[0] or y_max != y_lims[1]:
            plt.ylim([y_min, y_max])
            y_lims = plt.ylim()  # Update y_lims after adjustment
        for k in range(len(w_points)):
            plt.plot([w_points[k], w_points[k]], y_lims, '--', color=[0.5, 0.5, 0.5],
                    linewidth=1.5)
        
        plt.show()
    
    return x_opt, A, b, L2_error


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    N = 64
    x_opt, A, b, L2_error = solve_redu_inter_frequency_h(gamma, N, None, plot_flag=True)
    print(f"Optimized parameters: {x_opt}")
    print(f"L^2 error: {L2_error:.6e}")
