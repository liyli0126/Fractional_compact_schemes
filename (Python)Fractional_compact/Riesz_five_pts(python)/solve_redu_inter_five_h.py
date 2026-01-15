import numpy as np
import matplotlib.pyplot as plt
import warnings


def solve_redu_inter_five_h(gamma, h=None, x_points_inter=None, do_plot=False):
    """
    Solve reduced interpolation problem for 5-point frequency scheme.
    
    Input:
        gamma: fractional derivative order
        h: grid spacing (if provided, interpolation points will be auto-selected based on h)
        x_points_inter: interpolation points in frequency space [0, pi]
        do_plot: whether to plot the results (default: False)
    
    Output:
        x_opt: optimized parameters [a1, a2, a3, alpha, beta]
        A: coefficient matrix
        b: right-hand side vector
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
                    max(w_max / 16, np.pi / 128),  # Low frequency (but not too small)
                    w_max / 8,                     # low frequency
                    w_max / 4,                     # Mid-low frequency
                    w_max / 2,                     # Mid frequency
                    min(w_max, 3 * np.pi / 4)      # High frequency (capped at 3*pi/4 for stability)
                ])
            else:
                # For very small h, use fixed small points
                x_points = np.array([np.pi / 128, np.pi / 64, np.pi / 32, np.pi / 16, np.pi / 8])
            
            # Ensure points are in valid range and sorted
            x_points = np.clip(x_points, np.pi / 256, np.pi)  # Clamp to [pi/256, pi]
            x_points = np.sort(np.unique(x_points))
            
            # Ensure we have at least 5 points (if some are too close, add more)
            if len(x_points) < 5:
                # Add more points if needed
                additional_points = np.linspace(np.pi / 128, min(w_max * 2, np.pi / 4), 5)
                x_points = np.sort(np.unique(np.concatenate([x_points, additional_points])))
                x_points = x_points[:min(5, len(x_points))]  # Take first 5
            
        elif h_array.ndim > 0:  # vector (backward compatibility)
            # h is actually x_points_inter
            x_points_inter = h
            x_points = np.asarray(x_points_inter).flatten()  # Ensure 1D array
            x_points = np.sort(x_points)
            x_points = np.clip(x_points, 0, np.pi)
            
            if len(x_points) == 0 or len(x_points) < 5:
                raise ValueError('x_points_inter must have at least 5 points.')
        else:
            raise ValueError('h must be a positive scalar <= 1 (grid spacing) or a vector (x_points_inter).')
    else:
        x_points = None
    
    # If both h and x_points_inter are provided, x_points_inter takes precedence
    if x_points_inter is not None and len(x_points_inter) >= 5:
        x_points = np.asarray(x_points_inter).flatten()
        x_points = np.sort(x_points)
        x_points = np.clip(x_points, 0, np.pi)
    
    # Final validation
    if x_points is None or len(x_points) == 0 or len(x_points) < 5:
        raise ValueError('At least 5 interpolation points are required.')
    
    # Check if points are too small (may cause ill-conditioning)
    min_x = np.min(x_points)
    if min_x < np.pi / 128:
        warnings.warn(f'Smallest interpolation point ({min_x:.6f}) is very small. This may cause ill-conditioning.', 
                     UserWarning)
    
    n_points = len(x_points)
    n_eq = n_points
    n_var = 5  # [a1, a2, a3, alpha, beta] (symmetric: a5=a3, a6=a2, a7=a1)
    
    # Target function: Riesz derivative -|w|^gamma
    def target_r(x):
        return -np.abs(x) ** gamma
    
    # Define the basis functions (5 functions for symmetric 7-point scheme)
    def phi_r1(x):
        return 2 * (np.cos(3 * x) - 1)  # a1 (symmetric to a7)
    
    def phi_r2(x):
        return 2 * (np.cos(2 * x) - 1)  # a2 (symmetric to a6)
    
    def phi_r3(x):
        return 2 * (np.cos(x) - 1)  # a3 (symmetric to a5)
    
    def phi_r4(x):
        return 2 * np.abs(x) ** gamma * np.cos(x)  # alpha
    
    def phi_r5(x):
        return 2 * np.abs(x) ** gamma * np.cos(2 * x)  # beta
    
    phi_r = [phi_r1, phi_r2, phi_r3, phi_r4, phi_r5]
    
    # Initialize matrix and vector
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    # Fill in A and b: each point gives one equation (real part only for Riesz)
    for i in range(n_points):
        xi = x_points[i]
        # Real part only (Riesz is pure real)
        A[i, :] = [f(xi) for f in phi_r]
        b[i] = target_r(xi)
    
    # Solve: use least-squares if overdetermined or underdetermined
    x_opt_5, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Expand to 5 parameters: [a1, a2, a3, alpha, beta]
    x_opt = np.array([x_opt_5[0], x_opt_5[1], x_opt_5[2], x_opt_5[3], x_opt_5[4]])
    
    if do_plot:
        # Determine frequency range for plotting
        # Use maximum frequency point instead of pi
        x_max = np.max(x_points)  # Use maximum frequency selection point
        x_plot = np.linspace(0, x_max, 1000)  # Dense frequency points for plotting
        
        # Compute exact solution (real part only for Riesz)
        exact_real = target_r(x_plot)
        
        # Compute numerical solution using the optimized parameters
        # Numerical solution: sum of basis functions weighted by x_opt
        num_real = np.zeros_like(x_plot)
        for k in range(5):
            num_real = num_real + x_opt[k] * phi_r[k](x_plot)
        
        # Values at collocation points
        exact_real_colloc = target_r(x_points)
        num_real_colloc = np.zeros_like(x_points)
        for k in range(5):
            num_real_colloc = num_real_colloc + x_opt[k] * phi_r[k](x_points)
        
        # Compute errors
        err_mag = np.abs(exact_real - num_real)
        eps_val = np.finfo(float).eps
        E_h = err_mag / (np.abs(exact_real) + eps_val)  # Relative error
        
        # Errors at collocation points
        err_mag_all = np.abs(exact_real_colloc - num_real_colloc)
        E_h_all = err_mag_all / (np.abs(exact_real_colloc) + eps_val)
        
        # Plot 1: Real part comparison
        plt.figure(figsize=(6, 4))
        plt.plot(x_plot, exact_real, 'b-', linewidth=2, label='Exact Real Part')
        plt.plot(x_plot, num_real, 'r--', linewidth=2, label='Numerical Real Part')
        plt.plot(x_points, exact_real_colloc, 'ko', markersize=4, markerfacecolor='k', label='Collocation points')
        plt.xlim([0, x_max])
        # Calculate y-axis range to include all data
        y_data_min = min([np.min(exact_real), np.min(num_real), np.min(exact_real_colloc)])
        y_data_max = max([np.max(exact_real), np.max(num_real), np.max(exact_real_colloc)])
        y_margin = (y_data_max - y_data_min) * 0.05  # 5% margin
        y_min = y_data_min - y_margin
        y_max = y_data_max + y_margin
        plt.ylim([y_min, y_max])
        # Get y-axis limits for vertical lines
        y_lims = plt.ylim()
        # Highlight collocation points with vertical lines covering entire y-axis
        for k in range(len(x_points)):
            plt.plot([x_points[k], x_points[k]], y_lims, '--', color=[0.5, 0.5, 0.5], linewidth=1.5)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Real-Valued', fontsize=12)
        N_str = f'N={int(1/h)}' if h is not None and h > 0 else 'N=unknown'
        plt.title(f'Exact vs Numerical Symbol Comparison (γ={gamma:.2f}, {N_str})', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        
        # Plot 2: L^2 absolute error
        plt.figure(figsize=(6, 4))
        plt.semilogy(x_plot, err_mag, 'r-', linewidth=2, label='L^2 Absolute Error')
        # Calculate y-axis range to include all collocation points
        y_data_min = min([np.min(err_mag_all), np.min(err_mag)])
        y_data_max = max([np.max(err_mag_all), np.max(err_mag)])
        # Add margin for semilogy (multiply/divide by a factor)
        y_min = max(eps_val, y_data_min * 0.5)  # Ensure positive and add margin below
        y_max = y_data_max * 2  # Add margin above
        plt.ylim([y_min, y_max])
        # Get y-axis limits for vertical lines
        y_lims = plt.ylim()
        # Create a dummy line for legend (dashed line style for interpolation point position)
        plt.plot([np.nan, np.nan], [np.nan, np.nan], '--', color=[0.5, 0.5, 0.5], 
                linewidth=1.5, label='Interpolation point position')
        # Draw vertical lines covering entire y-axis range for ALL collocation points
        for k in range(len(x_points)):
            plt.plot([x_points[k], x_points[k]], y_lims, '--', color=[0.5, 0.5, 0.5], linewidth=1.5)
        # Mark ALL collocation points individually to ensure visibility
        if len(x_points) > 0:
            for k in range(len(x_points)):
                plt.semilogy(x_points[k], err_mag_all[k], 'ko', markersize=5, markerfacecolor='k')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('L^2 Absolute Error', fontsize=12)
        N_str = f'N={int(1/h)}' if h is not None and h > 0 else 'N=unknown'
        plt.title(f'L^2 Absolute Error (γ={gamma:.2f}, {N_str})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        
        # Plot 3: Relative error E_h(omega)
        plt.figure(figsize=(6, 4))
        plt.semilogy(x_plot, E_h, 'r-', linewidth=2, label='E_h(w)')
        # Calculate y-axis range to include all collocation points
        y_data_min = min([np.min(E_h_all), np.min(E_h)])
        y_data_max = max([np.max(E_h_all), np.max(E_h)])
        # Add margin for semilogy (multiply/divide by a factor)
        y_min = max(eps_val, y_data_min * 0.5)  # Ensure positive and add margin below
        y_max = y_data_max * 2  # Add margin above
        plt.ylim([y_min, y_max])
        # Get y-axis limits for vertical lines
        y_lims = plt.ylim()
        # Create a dummy line for legend (dashed line style for interpolation point position)
        plt.plot([np.nan, np.nan], [np.nan, np.nan], '--', color=[0.5, 0.5, 0.5], 
                linewidth=1.5, label='Interpolation position')
        # Draw vertical lines covering entire y-axis range for ALL collocation points
        for k in range(len(x_points)):
            plt.plot([x_points[k], x_points[k]], y_lims, '--', color=[0.5, 0.5, 0.5], linewidth=1.5)
        # Mark ALL collocation points individually to ensure visibility
        if len(x_points) > 0:
            for k in range(len(x_points)):
                plt.semilogy(x_points[k], E_h_all[k], 'ko', markersize=5, markerfacecolor='k')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Relative Error E_h(ω)', fontsize=12)
        N_str = f'N={int(1/h)}' if h is not None and h > 0 else 'N=unknown'
        plt.title(f'Relative Error E_h(ω) (γ={gamma:.2f}, {N_str})', fontsize=12)
        plt.grid(True)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, x_max])
        plt.tight_layout()
        
        plt.show()
    
    return x_opt, A, b


if __name__ == '__main__':
    # Example usage
    gamma = 1.5
    h = 1/64
    x_opt, A, b = solve_redu_inter_five_h(gamma, h, do_plot=True)
    print(f"Optimized parameters: {x_opt}")