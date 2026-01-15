function test_pde_2d_periodic_schemes_h(gamma, M)
% Test 2D PDE: D_x^gamma u + D_y^gamma u + u = f with periodic boundary conditions
% Source term: f(x,y) = sin(2pi x) sin(2pi y)
%
% Input:
%   gamma: fractional derivative order (e.g., 0.5, 1.0, 1.5)
%   M: number of grid points per direction (x_0, x_1, ..., x_{M-1})
%      covering [0, 1) × [0, 1) with periodicity
%      - Can be a scalar (single test) or a vector (multiple tests)

    % Determine if single test or multiple tests
    if isscalar(M)
        M_array = round(M);  % Ensure integer
        single_test = true;
    else
        M_array = round(M(:));  % Ensure integers and column vector
        single_test = false;
        % Ensure all M values are positive integers
        if any(M_array <= 0) || any(M_array - round(M_array) > 1e-10)
            error('All values in M_array must be positive integers');
        end
    end

    if single_test
        % Single test: formatted output
        M = M_array;
        h = 1 / M;
       
        % Get coefficients using h-based auto-selection
        [c_int, ~, ~] = solve_redu_inter_five_h(gamma, h,[],false);
     
        [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, X, Y, u_exact_2D, u_num_2D, err_2D, use_S, S_err] = ...
            solve_single_pde_2d_periodic(gamma, M, c_int);
        
        % Plot exact solution
        figure('Position', [100, 100, 800, 600]);
        surf(X, Y, u_exact_2D);
        shading interp;
        clim([-1, 1]);
        colorbar;
        xlabel('x', 'FontSize', 14);
        ylabel('y', 'FontSize', 14);
        zlabel('u_{exact}(x,y)', 'FontSize', 14);
        title(sprintf('Exact Solution: u(x,y) = sin(2\\pi x) sin(2\\pi y), \\gamma=%.2f', gamma), ...
              'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12);
        
        % Plot numerical solution
        figure('Position', [383, 100, 800, 600]);
        surf(X, Y, u_num_2D);
        shading interp;
        clim([-1, 1]);
        zlim([-1, 1]);
        colorbar;
        xlabel('x', 'FontSize', 14);
        ylabel('y', 'FontSize', 14);
        zlabel('u_{num}(x,y)', 'FontSize', 14);
        title(sprintf('Numerical Solution, \\gamma=%.2f, M=%d', gamma, M), ...
              'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12);
        
        % Plot error
        figure('Position', [666, 100, 800, 600]);
        surf(X, Y, err_2D);
        shading interp;
        colorbar;
        xlabel('x', 'FontSize', 14);
        ylabel('y', 'FontSize', 14);
        zlabel('Error', 'FontSize', 14);
        title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f, M=%d', gamma, M), ...
              'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12);
        
        % Output results in formatted table
        fprintf('\n=== Results ===\n');
        fprintf('Testing 2D PDE (gamma=%.2f) on f(x,y)=sin(2πx)sin(2πy) with periodic boundary conditions\n', gamma);
        fprintf('S operator 1D error (sin(2πx)): %.2e\n', S_err);
        if use_S
            fprintf('Using S operator form: L = S_x + S_y + I, u = L \\ f\n');
        else
            fprintf('Using A/B form: (A2 + B2) * u = A2 * f (S operator error too large)\n');
        end
        fprintf('===========================================================================================================\n');
        fprintf('                                    RESULTS TABLE\n');
        fprintf('===========================================================================================================\n');
        fprintf('%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'M', 'h', 'L1 Error', 'L2 Error', 'L_max Error', 'Rel L1', 'Rel L2', 'Rel L_max');
        fprintf('-----------------------------------------------------------------------------------------------------------\n');
        fprintf('%-6d %-12.6e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                M, h, L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax);
        fprintf('===========================================================================================================\n');
        
    else
        % Multiple tests: run for each M and create table
        n_tests = length(M_array);
        h_array = zeros(n_tests, 1);
        L1_errors = zeros(n_tests, 1);
        L2_errors = zeros(n_tests, 1);
        Lmax_errors = zeros(n_tests, 1);
        relative_L1_errors = zeros(n_tests, 1);
        relative_L2_errors = zeros(n_tests, 1);
        relative_Lmax_errors = zeros(n_tests, 1);
        
        
        % Run tests for each M
        for i = 1:n_tests
            M = M_array(i);
            h = 1 / M;
            h_array(i) = h;
            
            % Get coefficients using h-based auto-selection for each M
            [c_int, ~, ~] = solve_redu_inter_five_h(gamma, h,[],false);
            
            [L1_errors(i), L2_errors(i), Lmax_errors(i), ...
             relative_L1_errors(i), relative_L2_errors(i), relative_Lmax_errors(i), ...
             X_plot, Y_plot, u_exact_plot, u_num_plot, err_plot, ~, ~] = ...
                solve_single_pde_2d_periodic(gamma, M, c_int);
            
            % Plot for last test
            if i == n_tests
                % Plot exact solution
                figure('Position', [100, 100, 800, 600]);
                surf(X_plot, Y_plot, u_exact_plot);
                shading interp;
                caxis([-1, 1]);
                colorbar;
                xlabel('x', 'FontSize', 14);
                ylabel('y', 'FontSize', 14);
                zlabel('u_{exact}(x,y)', 'FontSize', 14);
                title(sprintf('Exact Solution: u(x,y) = sin(2\\pi x) sin(2\\pi y), \\gamma=%.2f', gamma), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                set(gca, 'FontSize', 12);
                
                % Plot numerical solution
                figure('Position', [383, 100, 800, 600]);
                surf(X_plot, Y_plot, u_num_plot);
                shading interp;
                caxis([-1, 1]);
                colorbar;
                xlabel('x', 'FontSize', 14);
                ylabel('y', 'FontSize', 14);
                zlabel('u_{num}(x,y)', 'FontSize', 14);
                title(sprintf('Numerical Solution, \\gamma=%.2f, M=%d', gamma, M), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                set(gca, 'FontSize', 12);
                
                % Plot error
                figure('Position', [666, 100, 800, 600]);
                surf(X_plot, Y_plot, err_plot);
                shading interp;
                colorbar;
                xlabel('x', 'FontSize', 14);
                ylabel('y', 'FontSize', 14);
                zlabel('Error', 'FontSize', 14);
                title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f, M=%d', gamma, M), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                set(gca, 'FontSize', 12);
            end
        end
        
        % Output results table
        fprintf('===========================================================================================================\n');
        fprintf('                                    RESULTS TABLE\n');
        fprintf('===========================================================================================================\n');
        fprintf('%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'M', 'h', 'L1 Error', 'L2 Error', 'L_max Error', 'Rel L1', 'Rel L2', 'Rel L_max');
        fprintf('-----------------------------------------------------------------------------------------------------------\n');
        for i = 1:n_tests
            fprintf('%-6d %-12.6e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                    M_array(i), h_array(i), ...
                    L1_errors(i), L2_errors(i), Lmax_errors(i), ...
                    relative_L1_errors(i), relative_L2_errors(i), relative_Lmax_errors(i));
        end
        fprintf('===========================================================================================================\n');
    end
end

function [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, X, Y, u_exact_2D, u_num_2D, err_2D, use_S_form, S_operator_error] = ...
    solve_single_pde_2d_periodic(gamma, M, c_int)
    
    h = 1 / M;
    x = (0:M-1)' * h;
    y = (0:M-1)' * h;
    
    A_1D = spalloc(M, M, 5*M);
    B_1D = spalloc(M, M, 7*M);
    
    for j = 1:M
        [A_row, B_row] = build_row_internal_periodic(M, h, gamma, c_int, j);
        A_1D(j, :) = A_row;
        B_1D(j, :) = B_row;
    end
    
    % Verify S operator accuracy for sin(2πx)
    % For Riesz derivative: D^gamma[sin(2πkx)] = -|2πk|^gamma * sin(2πkx)
    k_wave = 1;
    u_test = sin(2*pi*k_wave*x);
    d_exact = -abs(2*pi*k_wave)^gamma * sin(2*pi*k_wave*x);
    
    % Compute S * u = A^{-1} * (B * u)
    B_times_u = B_1D * u_test;
    d_num = A_1D \ B_times_u;
    S_operator_error = norm(d_num - d_exact, inf);
    
    % Use S form if error is small enough, otherwise use A/B form
    use_S_form = (S_operator_error < 1e-5);
    
    if use_S_form
        % Use S operator form: L = S_x + S_y + I, u = L \ f
        S_1D = A_1D \ B_1D;
        I_M = speye(M);
        I_2D = speye(M*M);
        L_2D = kron(S_1D, I_M) + kron(I_M, S_1D) + I_2D;
        
        [X, Y] = meshgrid(x, y);
        % For Riesz derivative: D_x^gamma u + D_y^gamma u + u = f
        % where u = sin(2πx)sin(2πy), so:
        % D_x^gamma u = -|2π|^gamma * sin(2πx)sin(2πy)
        % D_y^gamma u = -|2π|^gamma * sin(2πx)sin(2πy)
        % f = (1 - 2*|2π|^gamma) * sin(2πx)sin(2πy)
        coeff = 1 - 2 * abs(2*pi*k_wave)^gamma;
        f_2D = coeff * sin(2*pi*X) .* sin(2*pi*Y);
        f_vec = f_2D(:);
        
        if M <= 64
            cond_L = cond(full(L_2D));
            if cond_L > 1e10
                warning('System matrix L is ill-conditioned (cond=%e); results may be unstable', cond_L);
            end
        end
        
        u_vec = L_2D \ f_vec;
    else
        % Use A/B form: (A2 + B2) * u = A2 * f
        I_M = speye(M);
        A2 = kron(A_1D, I_M) + kron(I_M, A_1D);
        B2 = kron(B_1D, I_M) + kron(I_M, B_1D);
        
        [X, Y] = meshgrid(x, y);
        % For Riesz derivative: D_x^gamma u + D_y^gamma u + u = f
        % where u = sin(2πx)sin(2πy), so:
        % D_x^gamma u = -|2π|^gamma * sin(2πx)sin(2πy)
        % D_y^gamma u = -|2π|^gamma * sin(2πx)sin(2πy)
        % f = (1 - 2*|2π|^gamma) * sin(2πx)sin(2πy)
        coeff = 1 - 2 * abs(2*pi*k_wave)^gamma;
        f_2D = coeff * sin(2*pi*X) .* sin(2*pi*Y);
        f_vec = f_2D(:);
        
        rhs2 = A2 * f_vec;
        system_matrix = A2 + B2;
        
        if M <= 64
            cond_sys = cond(full(system_matrix));
            if cond_sys > 1e10
                warning('System matrix (A2 + B2) is ill-conditioned (cond=%e); results may be unstable', cond_sys);
            end
        end
        
        u_vec = system_matrix \ rhs2;
    end
    u_exact_2D = sin(2*pi*X) .* sin(2*pi*Y);
    u_exact_vec = u_exact_2D(:);
    
    u_num_2D = reshape(u_vec, M, M);
    err_2D = u_num_2D - u_exact_2D;
    err_vec = err_2D(:);
    
    L1_err = h^2 * sum(abs(err_vec));
    L2_err = h * sqrt(sum(err_vec.^2));
    Lmax_err = max(abs(err_vec));
    
    u_exact_L1_norm = h^2 * sum(abs(u_exact_vec));
    u_exact_L2_norm = h * sqrt(sum(u_exact_vec.^2));
    u_exact_Lmax_norm = max(abs(u_exact_vec));
    
    rel_L1 = (u_exact_L1_norm > eps) * (L1_err / u_exact_L1_norm) + (u_exact_L1_norm <= eps) * L1_err;
    rel_L2 = (u_exact_L2_norm > eps) * (L2_err / u_exact_L2_norm) + (u_exact_L2_norm <= eps) * L2_err;
    rel_Lmax = (u_exact_Lmax_norm > eps) * (Lmax_err / u_exact_Lmax_norm) + (u_exact_Lmax_norm <= eps) * Lmax_err;

end

% -------------------------------------------------------------------------
% Helper: build row for internal points with periodic boundary conditions
% Riesz derivative format: symmetric 5-coefficient scheme
% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_internal_periodic(M, h, gamma, coeffs, j)
    % coeffs = [a1, a2, a3, alpha, beta] from solve_redu_inter_h
    % Riesz derivative: symmetric format
    % RHS: (1/h^gamma) * [a1*u_{j-3} + a2*u_{j-2} + a3*u_{j-1} + a4*u_j + a3*u_{j+1} + a2*u_{j+2} + a1*u_{j+3}]
    % where a4 = -2*(a1 + a2 + a3)
    
    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    alpha = coeffs(4); beta = coeffs(5);
    a4 = -2*(a1 + a2 + a3);

    A_row = sparse(1, M);
    B_row = sparse(1, M);

    % Use periodic indexing to find neighboring points
    idx_m3 = mod(j-4, M) + 1;  % Periodic: j-3
    idx_m2 = mod(j-3, M) + 1;  % Periodic: j-2
    idx_m1 = mod(j-2, M) + 1;  % Periodic: j-1
    idx_p1 = mod(j, M) + 1;    % Periodic: j+1
    idx_p2 = mod(j+1, M) + 1;  % Periodic: j+2
    idx_p3 = mod(j+2, M) + 1;  % Periodic: j+3

    % LHS: d_j + alpha*(d_{j-1}+d_{j+1}) + beta*(d_{j-2}+d_{j+2})
    A_row(j) = 1;
    A_row(idx_m1) = alpha;
    A_row(idx_p1) = alpha;
    A_row(idx_m2) = beta;
    A_row(idx_p2) = beta;

    % RHS: (1/h^gamma) * [a1*u_{j-3} + a2*u_{j-2} + a3*u_{j-1} + a4*u_j + a3*u_{j+1} + a2*u_{j+2} + a1*u_{j+3}]
    % Symmetric format for Riesz derivative
    scale = 1 / h^gamma;
    B_row(j) = a4 * scale;
    B_row(idx_m3) = a1 * scale;
    B_row(idx_m2) = a2 * scale;
    B_row(idx_m1) = a3 * scale;
    B_row(idx_p1) = a3 * scale;  % Symmetric: same as a3
    B_row(idx_p2) = a2 * scale;  % Symmetric: same as a2
    B_row(idx_p3) = a1 * scale;  % Symmetric: same as a1

end
