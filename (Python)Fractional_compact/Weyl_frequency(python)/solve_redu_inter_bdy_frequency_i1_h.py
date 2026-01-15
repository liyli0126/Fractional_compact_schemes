import numpy as np
import matplotlib.pyplot as plt
import warnings


def solve_redu_inter_bdy_frequency_i1_h(gamma, h=None, x_points_inter=None, plot_flag=False):
    """
    Solve reduced interpolation problem for Weyl derivative boundary scheme i1 (frequency domain).
    
    Input:
        gamma: fractional derivative order
        h: grid spacing (if provided, interpolation points will be auto-selected based on h)
        x_points_inter: interpolation points in frequency space [0, pi] (optional)
        plot_flag: if True, plot comparison and error figures (optional, default: False)
    
    Output:
        x_opt: optimized parameters [a2, a3, a4, a5, a6, a7, alpha, beta]
               (a1 eliminated, a1 = -(a2+a3+a4+a5+a6+a7))
        A: coefficient matrix
        b: right-hand side vector
        L2_error: L^2 error of the approximation (optional)
    """
    
    # Parameter validation and interpolation point selection
    if h is None and x_points_inter is None:
        raise ValueError('At least gamma and h (or x_points_inter) must be provided.')
    
    # Determine mode: check if second argument is h (scalar) or x_points_inter (vector)
    if h is not None:
        h_array = np.asarray(h)
        if h_array.ndim == 0 and 0 < h <= 1:  # scalar
            # Method: Scale by h to cover frequency range
            w_max = 2 * np.pi * h  # Maximum frequency for one period in [0,1)
            w_max = min(w_max, np.pi)  # Cap at Nyquist frequency
            
            # Select points covering [w_max/16, w_max] with good distribution
            # Boundary schemes need at least 4 points (8 parameters, 2 equations per point)
            if w_max > np.pi / 32:
                # For reasonable h, use scaled points
                x_points = np.array([
                    max(w_max / 16, np.pi / 128),  # Low frequency (but not too small)
                    w_max / 8,                     # Mid-low frequency
                    w_max / 4,                     # Mid frequency
                    min(w_max, np.pi / 4)          # High frequency (capped at pi/4 for stability)
                ])
            else:
                # For very small h, use fixed small points
                x_points = np.array([np.pi / 128, np.pi / 64, np.pi / 32, np.pi / 16])
            
            # Ensure points are in valid range and sorted
            x_points = np.clip(x_points, np.pi / 256, np.pi)  # Clamp to [pi/256, pi]
            x_points = np.sort(np.unique(x_points.flatten()))
            
            # Ensure we have at least 4 points (if some are too close, add more)
            if len(x_points) < 4:
                # Add more points if needed
                additional_points = np.linspace(np.pi / 128, min(w_max * 2, np.pi / 4), 4)
                x_points = np.sort(np.unique(np.concatenate([x_points, additional_points])))
                # Ensure at least 4 distinct points
                if len(x_points) < 4:
                    # Add evenly spaced points in valid range to reach 4
                    range_min = max(np.pi / 256, np.min(x_points))
                    range_max = min(np.pi, max(np.pi / 64, np.max(x_points)))
                    if range_max <= range_min:
                        range_max = range_min + np.pi / 32  # Ensure range is non-zero
                    additional_fill = np.linspace(range_min, range_max, 4)
                    x_points = np.sort(np.unique(np.concatenate([x_points, additional_fill])))
                # Ensure we keep at least 4 points
                if len(x_points) > 4:
                    x_points = x_points[:4]  # Keep first 4 if more than needed
            
        elif h_array.ndim > 0:  # vector (backward compatibility)
            # h is actually x_points_inter
            x_points_inter = h
            x_points = np.asarray(x_points_inter).flatten()  # Ensure 1D array
            x_points = np.sort(x_points)
            x_points = np.clip(x_points, 0, np.pi)
            
            if len(x_points) == 0 or len(x_points) < 4:
                raise ValueError('x_points_inter must have at least 4 points.')
        else:
            raise ValueError('h must be a positive scalar <= 1 (grid spacing) or a vector (x_points_inter).')
    else:
        x_points = None
    
    # If both h and x_points_inter are provided, x_points_inter takes precedence
    if x_points_inter is not None and len(x_points_inter) >= 4:
        x_points = np.asarray(x_points_inter).flatten()
        x_points = np.sort(x_points)
        x_points = np.clip(x_points, 0, np.pi)
    
    # Final validation
    if x_points is None or len(x_points) == 0 or len(x_points) < 4:
        raise ValueError('At least 4 interpolation points are required for boundary schemes.')
    
    # Check if points are too small (may cause ill-conditioning)
    min_x = np.min(x_points)
    if min_x < np.pi / 128:
        warnings.warn(f'Smallest interpolation point ({min_x:.6f}) is very small. This may cause ill-conditioning.',
                     UserWarning)
    
    n_points = len(x_points)
    n_eq = 2 * n_points  # each point gives real + imag equation
    n_var = 8  # [a2, a3, a4, a5, a6, a7, alpha, beta] (a1 eliminated, a1 = -(a2+a3+a4+a5+a6+a7))
    
    # Precompute trig constants
    s = np.sin(gamma * np.pi / 2)
    c = np.cos(gamma * np.pi / 2)
    
    # Target function (iw)^gamma
    def target_r(x):
        return c * x ** gamma
    
    def target_i(x):
        return s * x ** gamma
    
    # Define the basis functions (8 functions for boundary scheme i1)
    # a1 is eliminated: a1 = -(a2+a3+a4+a5+a6+a7)
    # Real parts
    def phi_r1(x):
        return np.cos(x) - 1  # a2
    
    def phi_r2(x):
        return np.cos(2 * x) - 1  # a3
    
    def phi_r3(x):
        return np.cos(3 * x) - 1  # a4
    
    def phi_r4(x):
        return np.cos(4 * x) - 1  # a5
    
    def phi_r5(x):
        return np.cos(5 * x) - 1  # a6
    
    def phi_r6(x):
        return np.cos(6 * x) - 1  # a7
    
    def phi_r7(x):
        return -c * x ** gamma * np.cos(x) + s * x ** gamma * np.sin(x)  # alpha
    
    def phi_r8(x):
        return -c * x ** gamma * np.cos(2 * x) + s * x ** gamma * np.sin(2 * x)  # beta
    
    # Imaginary parts
    def phi_i1(x):
        return np.sin(x)  # a2
    
    def phi_i2(x):
        return np.sin(2 * x)  # a3
    
    def phi_i3(x):
        return np.sin(3 * x)  # a4
    
    def phi_i4(x):
        return np.sin(4 * x)  # a5
    
    def phi_i5(x):
        return np.sin(5 * x)  # a6
    
    def phi_i6(x):
        return np.sin(6 * x)  # a7
    
    def phi_i7(x):
        return -s * x ** gamma * np.cos(x) - c * x ** gamma * np.sin(x)  # alpha
    
    def phi_i8(x):
        return -s * x ** gamma * np.cos(2 * x) - c * x ** gamma * np.sin(2 * x)  # beta
    
    phi_r = [phi_r1, phi_r2, phi_r3, phi_r4, phi_r5, phi_r6, phi_r7, phi_r8]
    phi_i = [phi_i1, phi_i2, phi_i3, phi_i4, phi_i5, phi_i6, phi_i7, phi_i8]
    
    # Initialize matrix and vector
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    # Fill in A and b: each point gives two equations (real and imaginary)
    for i in range(n_points):
        xi = x_points[i]
        # Real part
        A[2 * i, :] = [f(xi) for f in phi_r]
        b[2 * i] = target_r(xi)
        # Imag part
        A[2 * i + 1, :] = [f(xi) for f in phi_i]
        b[2 * i + 1] = target_i(xi)
    
    # Solve: use least-squares if overdetermined or underdetermined
    x_opt, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
    
    # Compute L^2 error on fine grid
    # x_max = max(min(pi, 2*pi*h), max(x_points) * 1.1) - Add 10% margin to include all points
    h_val = h if (h is not None and np.isscalar(h) and 0 < h <= 1) else 1.0 / 64
    x_max = max(min(np.pi, 2 * np.pi * h_val), np.max(x_points) * 1.1)
    x_fine = np.linspace(0, x_max, 1000)
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
        # Create fine grid for plotting
        x_plot = np.linspace(0, np.max(x_points), 200)
        
        # Compute exact values
        exact_real = target_r(x_plot)
        exact_imag = target_i(x_plot)
        
        # Compute numerical approximation
        num_real = np.zeros_like(x_plot)
        num_imag = np.zeros_like(x_plot)
        for k in range(n_var):
            num_real = num_real + x_opt[k] * phi_r[k](x_plot)
            num_imag = num_imag + x_opt[k] * phi_i[k](x_plot)
        
        # Compute errors
        err_real = exact_real - num_real
        err_imag = exact_imag - num_imag
        err_magnitude = np.sqrt(err_real ** 2 + err_imag ** 2)
        
        # Compute relative error: |exact - numerical| / (|w|^gamma + eps)
        eps_rel = 1e-10  # Small value to avoid division by zero
        rel_error = err_magnitude / (x_plot ** gamma + eps_rel)
        
        # Compute relative error at interpolation points
        exact_real_points = target_r(x_points)
        exact_imag_points = target_i(x_points)
        num_real_points = np.zeros_like(x_points)
        num_imag_points = np.zeros_like(x_points)
        for k in range(n_var):
            num_real_points = num_real_points + x_opt[k] * phi_r[k](x_points)
            num_imag_points = num_imag_points + x_opt[k] * phi_i[k](x_points)
        err_magnitude_points = np.sqrt((exact_real_points - num_real_points) ** 2 +
                                      (exact_imag_points - num_imag_points) ** 2)
        rel_error_points = err_magnitude_points / (x_points ** gamma + eps_rel)
        
        # Print L^2 error
        h_display = h if (h is not None and np.isscalar(h)) else 1.0 / 64
        print(f'Internal scheme: L^2 error = {L2_error:.6e} (gamma={gamma:.2f}, h={h_display:.6e})')
        
        # Plot 1: Real part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_real, 'b-', linewidth=2, label='Exact Real Part')
        plt.plot(x_plot, num_real, 'r--', linewidth=2, label='Numerical Real Part')
        plt.plot(x_points, target_r(x_points), 'ko', markersize=5, markerfacecolor='k',
                label='Collocation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Real Part', fontsize=12)
        h_display_for_title = h_display
        if h_display_for_title == 0:
            h_display_for_title = 1.0 / 64
        plt.title(f'Real Part Comparison (γ={gamma:.2f}, N={int(1/h_display_for_title)})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        # Add gray dashed vertical lines at interpolation points (after all settings)
        y_lim = plt.ylim()
        for i in range(len(x_points)):
            plt.plot([x_points[i], x_points[i]], y_lim, '--', color=[0.7, 0.7, 0.7],
                    linewidth=1)
        
        # Plot 2: Imaginary part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_imag, 'b-', linewidth=2, label='Exact Imaginary Part')
        plt.plot(x_plot, num_imag, 'r--', linewidth=2, label='Numerical Imaginary Part')
        plt.plot(x_points, target_i(x_points), 'ko', markersize=5, markerfacecolor='k',
                label='Collocation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Imaginary Part', fontsize=12)
        plt.title(f'Imaginary Part Comparison (γ={gamma:.2f}, N={int(1/h_display_for_title)})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        # Add gray dashed vertical lines at interpolation points (after all settings)
        y_lim = plt.ylim()
        for i in range(len(x_points)):
            plt.plot([x_points[i], x_points[i]], y_lim, '--', color=[0.7, 0.7, 0.7],
                    linewidth=1)
        
        # Plot 3: Real part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_real, 'r-', linewidth=2)
        plt.plot(x_points, np.zeros_like(x_points), 'ko', markersize=5, markerfacecolor='k',
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Real Part Error', fontsize=12)
        plt.title(f'Real Part Error (Exact - Numerical) (γ={gamma:.2f})', fontsize=12)
        plt.grid(True)
        plt.legend(['Error', 'Interpolation Points'], loc='best', fontsize=10)
        plt.tight_layout()
        # Add gray dashed vertical lines at interpolation points (after all settings)
        y_lim = plt.ylim()
        for i in range(len(x_points)):
            plt.plot([x_points[i], x_points[i]], y_lim, '--', color=[0.7, 0.7, 0.7],
                    linewidth=1)
        
        # Plot 4: Imaginary part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_imag, 'r-', linewidth=2)
        plt.plot(x_points, np.zeros_like(x_points), 'ko', markersize=5, markerfacecolor='k',
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Imaginary Part Error', fontsize=12)
        plt.title(f'Imaginary Part Error (Exact - Numerical) (γ={gamma:.2f})', fontsize=12)
        plt.grid(True)
        plt.legend(['Error', 'Interpolation Points'], loc='best', fontsize=10)
        plt.tight_layout()
        # Add gray dashed vertical lines at interpolation points (after all settings)
        y_lim = plt.ylim()
        for i in range(len(x_points)):
            plt.plot([x_points[i], x_points[i]], y_lim, '--', color=[0.7, 0.7, 0.7],
                    linewidth=1)
        
        # Plot 5: Relative error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, rel_error, 'r-', linewidth=2)
        plt.plot(x_points, rel_error_points, 'ko', markersize=5, markerfacecolor='k',
                label='Interpolation Points')
        plt.yscale('log')  # Use log scale for better visualization
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Relative Error', fontsize=12)
        plt.title(f'Relative Error (γ={gamma:.2f}, N={int(1/h_display_for_title)}, j=i_1)', fontsize=12)
        plt.grid(True)
        plt.legend(['Relative Error', 'Interpolation Points'], loc='best', fontsize=10)
        plt.tight_layout()
        # Add gray dashed vertical lines at interpolation points (after all settings, including log scale)
        y_lim = plt.ylim()
        for i in range(len(x_points)):
            plt.plot([x_points[i], x_points[i]], y_lim, '--', color=[0.7, 0.7, 0.7],
                    linewidth=1)
        
        plt.show()
    
    return x_opt, A, b, L2_error


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    h = 1.0 / 64
    x_opt, A, b, L2_error = solve_redu_inter_bdy_frequency_i1_h(gamma, h, None, plot_flag=True)
    print(f"Optimized parameters: {x_opt}")
    print(f"L^2 error: {L2_error:.6e}")
