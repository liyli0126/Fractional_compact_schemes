import numpy as np
import matplotlib.pyplot as plt
import warnings


def solve_redu_inter_h(gamma, h=None, x_points_inter=None, plot_flag=False):
    """
    Solve reduced interpolation problem for Weyl derivative (7-point scheme).

    Input:
        gamma: fractional derivative order
        h: grid spacing (if provided, interpolation points will be auto-selected based on h)
        x_points_inter: interpolation points in frequency space [0, pi] (optional)
        plot_flag: if True, plot comparison and error figures (optional, default: False)
    
    Output:
        x_opt: optimized parameters [a1, a2, a3, a5, a6, a7, alpha, beta]
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
            
            # Select points covering [w_max/8, w_max] with good distribution
            if w_max > np.pi / 32:
                # For reasonable h, use scaled points
                x_points = np.array([
                    max(w_max / 8, np.pi / 128),  # Low frequency (but not too small)
                    w_max / 4,                    # Mid-low frequency
                    w_max / 2,                    # Mid frequency
                    min(w_max, 3 * np.pi / 4),    # High frequency (capped at 3*pi/4 for stability)
                    min(w_max, np.pi)
                ])
            else:
                # For very small h, use fixed small points
                x_points = np.array([np.pi / 128, np.pi / 64, np.pi / 32, np.pi / 16])
            
            # Ensure points are in valid range and sorted
            x_points = np.clip(x_points, np.pi / 256, np.pi)  # Clamp to [pi/256, pi]
            x_points = np.sort(np.unique(x_points))
            
            # Ensure we have at least 4 points (if some are too close, add more)
            if len(x_points) < 4:
                # Add more points if needed
                additional_points = np.linspace(np.pi / 128, min(w_max * 2, np.pi / 4), 4)
                x_points = np.sort(np.unique(np.concatenate([x_points, additional_points])))
                x_points = x_points[:min(4, len(x_points))]  # Take first 4
            
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
        raise ValueError('At least 4 interpolation points are required.')
    
    # Check if points are too small (may cause ill-conditioning)
    min_x = np.min(x_points)
    if min_x < np.pi / 128:
        warnings.warn(f'Smallest interpolation point ({min_x:.6f}) is very small. This may cause ill-conditioning.', 
                     UserWarning)
    
    n_points = len(x_points)
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
    
    # Define the basis functions (8 functions for non-symmetric 7-point scheme)
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
    x_fine = np.linspace(0, np.pi, 1000)
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
    
    # Plot comparison and error figures (only if plot_flag is True)
    if plot_flag:
        # Create fine grid for plotting
        x_plot = np.linspace(0, np.pi, 200)
        
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
        
        # Print L^2 error
        print(f'Internal scheme: L^2 error = {L2_error:.6e} (gamma={gamma:.2f}, h={h:.6e})')
        
        # Plot 1: Real part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_real, 'b-', linewidth=2, label='Exact Real Part')
        plt.plot(x_plot, num_real, 'r--', linewidth=2, label='Numerical Real Part')
        plt.plot(x_points, target_r(x_points), 'ko', markersize=8, markerfacecolor='k', 
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Real Part', fontsize=12)
        plt.title(f'Real Part Comparison (γ={gamma:.2f})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        
        # Plot 2: Imaginary part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_imag, 'b-', linewidth=2, label='Exact Imaginary Part')
        plt.plot(x_plot, num_imag, 'r--', linewidth=2, label='Numerical Imaginary Part')
        plt.plot(x_points, target_i(x_points), 'ko', markersize=8, markerfacecolor='k', 
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Imaginary Part', fontsize=12)
        plt.title(f'Imaginary Part Comparison (γ={gamma:.2f})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        
        # Plot 3: Real part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_real, 'r-', linewidth=2, label='Error')
        plt.plot(x_points, np.zeros_like(x_points), 'ko', markersize=8, markerfacecolor='k', 
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Real Part Error', fontsize=12)
        plt.title(f'Real Part Error (Exact - Numerical) (γ={gamma:.2f})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        # Plot 4: Imaginary part error
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, err_imag, 'r-', linewidth=2, label='Error')
        plt.plot(x_points, np.zeros_like(x_points), 'ko', markersize=8, markerfacecolor='k', 
                label='Interpolation Points')
        plt.xlabel('x (frequency)', fontsize=12)
        plt.ylabel('Imaginary Part Error', fontsize=12)
        plt.title(f'Imaginary Part Error (Exact - Numerical) (γ={gamma:.2f})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        plt.show()
    
    return x_opt, A, b, L2_error


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    h = 1.0 / 64
    x_opt, A, b, L2_error = solve_redu_inter_h(gamma, h, None, plot_flag=True)
    print(f"Optimized parameters: {x_opt}")
    print(f"L^2 error: {L2_error:.6e}")
