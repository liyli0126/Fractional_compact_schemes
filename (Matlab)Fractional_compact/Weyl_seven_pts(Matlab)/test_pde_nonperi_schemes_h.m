function test_pde_nonperi_schemes_h(gamma, N)
% Solve PDE using compact schemes with Weyl-consistent boundary treatment
% Input:
%   gamma: fractional derivative order (0 < gamma <= 2)
%   N: number of grid points (scalar or array)
% Test function: u(x) = exp(lambda*x) * sin(2*pi*x) on [0,1], lambda=1
% PDE: D^gamma u + kappa*u = f with homogeneous Dirichlet BC
% BC: u(0) = 0 (for 0 < gamma <= 1)
%     u(0) = 0, u(1) = 0 (for 1 < gamma <= 2)

    if nargin < 1 || isempty(gamma)
        error('gamma must be provided');
    end
    if nargin < 2 || isempty(N)
        error('N must be provided');
    end

    if isscalar(N)
        N_array = round(N);
        single_test = true;
    else
        N_array = round(N(:));
        single_test = false;
        if any(N_array <= 0) || any(N_array - round(N_array) > 1e-10)
            error('All values in N_array must be positive integers');
        end
    end

    if single_test
        N = N_array;
        L = 1;
        h = L / N;
        num_points = N + 1;

        [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
        [c_i0, ~, ~] = solve_redu_inter_bdy_i1_h(gamma, h);
        [c_i1, ~, ~] = solve_redu_inter_bdy_i2_h(gamma, h);
        [c_i2, ~, ~] = solve_redu_inter_bdy_i3_h(gamma, h);
        [c_iN_1, ~, ~] = solve_redu_inter_bdy_iN_h(gamma, h);
        [c_iN_2, ~, ~] = solve_redu_inter_bdy_iN_1_h(gamma, h);
        [c_iN, ~, ~] = solve_redu_inter_bdy_iN1_h(gamma, h);

        [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, x, u_exact, u_num] = ...
            solve_single_pde_poly(gamma, N, c_int, c_i0, c_i1, c_i2, c_iN_1, c_iN_2, c_iN);

        figure('Position', [100, 100, 800, 600]);
        plot(x, u_exact, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Exact Solution');
        hold on;
        plot(x, u_num, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
             'MarkerIndices', 1:max(1,floor(length(x)/20)):length(x), ...
             'MarkerFaceColor', 'r', 'DisplayName', 'Numerical Solution');
        hold off;
        xlabel('x', 'FontSize', 14);
        ylabel('u(x)', 'FontSize', 14);
        xlim([0, 1]);
        title(sprintf('Exact vs Numerical Solution: \\gamma=%.2f', gamma), ...
              'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 12);
        grid on;
        set(gca, 'FontSize', 12);
        
        % Plot error
        err = u_num - u_exact;
        figure('Position', [450, 100, 800, 600]);
        plot(x, err, 'r-', 'LineWidth', 2.5);
        xlabel('x', 'FontSize', 14);
        ylabel('Error (u_{num} - u_{exact})', 'FontSize', 14);
        xlim([0, 1]);
        title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f', gamma), ...
              'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        set(gca, 'FontSize', 12);

        fprintf('\n=== Results ===\n');
        if gamma > 1
            fprintf('Testing PDE (gamma=%.2f) on u(x)=exp(x)*sin(2*pi*x) with BC: u(0)=0, u(1)=0\n', gamma);
        else
            fprintf('Testing PDE (gamma=%.2f) on u(x)=exp(x)*sin(2*pi*x) with BC: u(0)=0\n', gamma);
        end
        fprintf('===========================================================================================================\n');
        fprintf('                                    RESULTS TABLE\n');
        fprintf('===========================================================================================================\n');
        fprintf('%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'N', 'h', 'L1 Error', 'L2 Error', 'L_max Error', 'Rel L1', 'Rel L2', 'Rel L_max');
        fprintf('-----------------------------------------------------------------------------------------------------------\n');
        fprintf('%-6d %-12.6e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                N, h, L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax);
        fprintf('===========================================================================================================\n');

    else
        n_tests = length(N_array);
        h_array = zeros(n_tests, 1);
        L1_errors = zeros(n_tests, 1);
        L2_errors = zeros(n_tests, 1);
        Lmax_errors = zeros(n_tests, 1);
        relative_L1_errors = zeros(n_tests, 1);
        relative_L2_errors = zeros(n_tests, 1);
        relative_Lmax_errors = zeros(n_tests, 1);

        for i = 1:n_tests
            N = N_array(i);
            L = 1;
            h = L / N;
            h_array(i) = h;

            [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
            [c_i0, ~, ~] = solve_redu_inter_bdy_i1_h(gamma, h);
            [c_i1, ~, ~] = solve_redu_inter_bdy_i2_h(gamma, h);
            [c_i2, ~, ~] = solve_redu_inter_bdy_i3_h(gamma, h);
            [c_iN, ~, ~] = solve_redu_inter_bdy_iN1_h(gamma, h);
            [c_iN_1, ~, ~] = solve_redu_inter_bdy_iN_h(gamma, h);
            [c_iN_2, ~, ~] = solve_redu_inter_bdy_iN_1_h(gamma, h);
            
            [L1_errors(i), L2_errors(i), Lmax_errors(i), ...
             relative_L1_errors(i), relative_L2_errors(i), relative_Lmax_errors(i), ...
             x_plot, u_exact_plot, u_num_plot] = ...
                solve_single_pde_poly(gamma, N, c_int, c_i0, c_i1, c_i2, c_iN_1, c_iN_2, c_iN);

            if i == n_tests || n_tests == 1
                figure('Position', [100, 100, 800, 600]);
                plot(x_plot, u_exact_plot, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Exact Solution');
                hold on;
                plot(x_plot, u_num_plot, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
                     'MarkerIndices', 1:max(1,floor(length(x_plot)/20)):length(x_plot), ...
                     'MarkerFaceColor', 'r', 'DisplayName', 'Numerical Solution');
                hold off;
                xlabel('x', 'FontSize', 14);
                ylabel('u(x)', 'FontSize', 14);
                xlim([0, 1]);
                title(sprintf('Exact vs Numerical Solution: \\gamma=%.2f, N=%d', gamma, N), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                legend('Location', 'best', 'FontSize', 12);
                grid on;
                set(gca, 'FontSize', 12);
                
                % Plot error
                err_plot = u_num_plot - u_exact_plot;
                figure('Position', [450, 100, 800, 600]);
                plot(x_plot, err_plot, 'r-', 'LineWidth', 2.5);
                xlabel('x', 'FontSize', 14);
                ylabel('Error (u_{num} - u_{exact})', 'FontSize', 14);
                xlim([0, 1]);
                title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f, N=%d', gamma, N), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 12);
            end
        end

        fprintf('===========================================================================================================\n');
        fprintf('                                    RESULTS TABLE\n');
        fprintf('===========================================================================================================\n');
        fprintf('%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'N', 'h', 'L1 Error', 'L2 Error', 'L_max Error', 'Rel L1', 'Rel L2', 'Rel L_max');
        fprintf('-----------------------------------------------------------------------------------------------------------\n');
        for i = 1:n_tests
            fprintf('%-6d %-12.6e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                    N_array(i), h_array(i), ...
                    L1_errors(i), L2_errors(i), Lmax_errors(i), ...
                    relative_L1_errors(i), relative_L2_errors(i), relative_Lmax_errors(i));
        end
        fprintf('===========================================================================================================\n');
    end
end

% -------------------------------------------------------------------------
function [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, x, u_exact, u_num] = ...
    solve_single_pde_poly(gamma, N, c_int, c_i0, c_i1, c_i2, c_iN_1, c_iN_2, c_iN)

    L = 1;
    h = L / N;
    num_points = N + 1;
    x = (0:N)' * h;

    lambda = 1.0;
    kappa = 1.0;

    u_exact = exp(lambda * x) .* sin(2 * pi * x);

    % Compute Weyl fractional derivative using complex representation
    % For u(x) = e^(lambda*x) * sin(2*pi*x) = Im(e^(lambda + i*2*pi)*x)
    % Weyl derivative: D^gamma [Im(e^zx)] = Im(z^gamma * e^zx)
    % where z = lambda + i*2*pi
    z = lambda + 1i * 2 * pi;
    
    % Compute z^gamma using principal branch: z^gamma = exp(gamma * log(z))
    % Use atan2 for correct phase handling in (-pi, pi]
    r_z = abs(z);
    theta_z = atan2(imag(z), real(z));  % Use atan2 for correct quadrant
    
    % z^gamma = r_z^gamma * exp(i * gamma * theta_z)
    r_gamma = r_z^gamma;
    theta_gamma = gamma * theta_z;
    
    % Real and imaginary parts of z^gamma
    z_gamma_real = r_gamma * cos(theta_gamma);
    z_gamma_imag = r_gamma * sin(theta_gamma);
    
    % D^gamma u = Im(z^gamma * e^(zx)) = Im((a+ib) * e^(lambda*x + i*2*pi*x))
    %            = Im((a+ib) * e^(lambda*x) * (cos(2*pi*x) + i*sin(2*pi*x)))
    %            = e^(lambda*x) * [a*sin(2*pi*x) + b*cos(2*pi*x)]
    D_gamma_u = exp(lambda * x) .* (z_gamma_real * sin(2 * pi * x) + z_gamma_imag * cos(2 * pi * x));
    
    % Alternative formulation using amplitude and phase (equivalent but may have different numerical behavior)
    % phase_shift = angle(z_gamma);  % This equals theta_gamma
    % D_gamma_u_alt = exp(lambda * x) .* r_gamma .* sin(2 * pi * x + theta_gamma);
    
    f_exact = D_gamma_u + kappa * u_exact;

    A_mat = spalloc(num_points, num_points, 5*num_points);
    B_mat = spalloc(num_points, num_points, 7*num_points);

    for idx = 1:num_points
        if idx == 1
            [A_row, B_row] = build_row_bdy_i1(num_points, h, gamma, c_i0, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        elseif idx == 2
            [A_row, B_row] = build_row_bdy_i2(num_points, h, gamma, c_i1, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        elseif idx == 3
            [A_row, B_row] = build_row_bdy_i3(num_points, h, gamma, c_i2, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        elseif idx == num_points
            [A_row, B_row] = build_row_bdy_iN1(num_points, h, gamma, c_iN, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        elseif idx == num_points - 1
            [A_row, B_row] = build_row_bdy_iN(num_points, h, gamma, c_iN_1, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        elseif idx == num_points - 2
            [A_row, B_row] = build_row_bdy_iN_1(num_points, h, gamma, c_iN_2, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;

        else
            [A_row, B_row] = build_row_internal(num_points, h, gamma, c_int, idx);
            A_mat(idx, :) = A_row;
            B_mat(idx, :) = B_row;
        end
    end

    % Construct system: (B + h^gamma*kappa*A) * u = h^gamma * A * f
    B_mat_modified = B_mat + (h^gamma * kappa) * A_mat;
    rhs_scaled = (h^gamma) * (A_mat * f_exact);
    
    % Apply boundary conditions based on gamma
    % For 0 < gamma <= 1: u(0) = 0
    % For 1 < gamma <= 2: u(0) = 0, u(1) = 0
    % Note: u_exact(0) = 0 and u_exact(1) = 0, so boundary values are 0
    if gamma > 1
        % Two boundary conditions: u(0) = 0, u(1) = 0
        B_mat_modified(1, :) = 0;
        B_mat_modified(1, 1) = 1;
        rhs_scaled(1) = 0;
        
        B_mat_modified(end, :) = 0;
        B_mat_modified(end, end) = 1;
        rhs_scaled(end) = 0;
    else
        % One boundary condition: u(0) = 0
        B_mat_modified(1, :) = 0;
        B_mat_modified(1, 1) = 1;
        rhs_scaled(1) = 0;
    end
    
    % Check condition number for diagnostics
    B_mat_full = full(B_mat_modified);
    cond_B = cond(B_mat_full);
    
    % Diagnostic information for fractional order issues
    if abs(gamma - round(gamma)) > 0.01  % Non-integer gamma
        fprintf('Diagnostic: gamma=%.2f (fractional), h=%.6e, h^gamma=%.6e, cond=%e\n', ...
                gamma, h, h^gamma, cond_B);
    end
    
    if cond_B > 1e12
        warning('System matrix is ill-conditioned (cond=%e) for gamma=%.2f; results may be unstable', cond_B, gamma);
    end
    
    % Solve the system (boundary conditions already applied, system should be well-posed)
    u_num = B_mat_modified \ rhs_scaled;
    err = u_num - u_exact;
    
    % Check boundary error vs interior error
    boundary_err = max([abs(err(1)), abs(err(2)), abs(err(3)), abs(err(end)), abs(err(end-1)), abs(err(end-2))]);
    interior_err = max(abs(err(4:end-3)));
    if boundary_err > 10 * interior_err && abs(gamma - round(gamma)) > 0.01
        fprintf('Warning: Boundary error (%.4e) >> Interior error (%.4e) for gamma=%.2f\n', ...
                boundary_err, interior_err, gamma);
        fprintf('  This suggests boundary closure schemes may need refinement for fractional orders.\n');
    end

    L1_err = h * sum(abs(err));
    L2_err = sqrt(h * sum(err.^2));
    Lmax_err = max(abs(err));

    u_exact_L1_norm = h * sum(abs(u_exact));
    u_exact_L2_norm = sqrt(h * sum(u_exact.^2));
    u_exact_Lmax_norm = max(abs(u_exact));

    if u_exact_L1_norm > eps
        rel_L1 = L1_err / u_exact_L1_norm;
    else
        rel_L1 = L1_err;
    end

    if u_exact_L2_norm > eps
        rel_L2 = L2_err / u_exact_L2_norm;
    else
        rel_L2 = L2_err;
    end

    if u_exact_Lmax_norm > eps
        rel_Lmax = Lmax_err / u_exact_Lmax_norm;
    else
        rel_Lmax = Lmax_err;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_i1(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a2 a3 a4 a5 a6 a7 alpha beta]
    a2 = coeffs(1); a3 = coeffs(2); a4 = coeffs(3);
    a5 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a1 = -(a2 + a3 + a4 + a5 + a6 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    A_row(idx) = 1;
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end
    if idx + 2 <= num_points
        A_row(idx + 2) = beta;
    end

    B_row(idx) = a1;
    if idx + 1 <= num_points
        B_row(idx + 1) = a2;
    end
    if idx + 2 <= num_points
        B_row(idx + 2) = a3;
    end
    if idx + 3 <= num_points
        B_row(idx + 3) = a4;
    end
    if idx + 4 <= num_points
        B_row(idx + 4) = a5;
    end
    if idx + 5 <= num_points
        B_row(idx + 5) = a6;
    end
    if idx + 6 <= num_points
        B_row(idx + 6) = a7;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_i2(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a1 a3 a4 a5 a6 a7 alpha beta]
    a1 = coeffs(1); a3 = coeffs(2); a4 = coeffs(3);
    a5 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a2 = -(a1 + a3 + a4 + a5 + a6 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    A_row(idx) = 1;
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end
    if idx + 2 <= num_points
        A_row(idx + 2) = beta;
    end

    if idx - 1 >= 1
        B_row(idx - 1) = a1;
    end
    B_row(idx) = a2;
    if idx + 1 <= num_points
        B_row(idx + 1) = a3;
    end
    if idx + 2 <= num_points
        B_row(idx + 2) = a4;
    end
    if idx + 3 <= num_points
        B_row(idx + 3) = a5;
    end
    if idx + 4 <= num_points
        B_row(idx + 4) = a6;
    end
    if idx + 5 <= num_points
        B_row(idx + 5) = a7;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_i3(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a1 a2 a4 a5 a6 a7 alpha beta]
    a1 = coeffs(1); a2 = coeffs(2); a4 = coeffs(3);
    a5 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a3 = -(a1 + a2 + a4 + a5 + a6 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    A_row(idx) = 1;
    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end
    if idx - 2 >= 1
        A_row(idx - 2) = beta;
    end
    if idx + 2 <= num_points
        A_row(idx + 2) = beta;
    end

    if idx - 2 >= 1
        B_row(idx - 2) = a1;
    end
    if idx - 1 >= 1
        B_row(idx - 1) = a2;
    end
    B_row(idx) = a3;
    if idx + 1 <= num_points
        B_row(idx + 1) = a4;
    end
    if idx + 2 <= num_points
        B_row(idx + 2) = a5;
    end
    if idx + 3 <= num_points
        B_row(idx + 3) = a6;
    end
    if idx + 4 <= num_points
        B_row(idx + 4) = a7;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_iN_1(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a1 a2 a3 a4 a6 a7 alpha beta] (a5 eliminated)
    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    a4 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a5 = -(a1 + a2 + a3 + a4 + a6 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    A_row(idx) = 1;
    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end
    if idx - 2 >= 1
        A_row(idx - 2) = beta;
    end
    if idx + 2 <= num_points
        A_row(idx + 2) = beta;
    end

    if idx - 4 >= 1
        B_row(idx - 4) = a1;
    end
    if idx - 3 >= 1
        B_row(idx - 3) = a2;
    end
    if idx - 2 >= 1
        B_row(idx - 2) = a3;
    end
    if idx - 1 >= 1
        B_row(idx - 1) = a4;
    end
    B_row(idx) = a5;
    if idx + 1 <= num_points
        B_row(idx + 1) = a6;
    end
    if idx + 2 <= num_points
        B_row(idx + 2) = a7;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_iN(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a1 a2 a3 a4 a5 a7 alpha beta] (a6 eliminated)
    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    a4 = coeffs(4); a5 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a6 = -(a1 + a2 + a3 + a4 + a5 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    if idx - 2 >= 1
        A_row(idx - 2) = beta;
    end
    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    A_row(idx) = 1;
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end

    if idx - 5 >= 1
        B_row(idx - 5) = a1;
    end
    if idx - 4 >= 1
        B_row(idx - 4) = a2;
    end
    if idx - 3 >= 1
        B_row(idx - 3) = a3;
    end
    if idx - 2 >= 1
        B_row(idx - 2) = a4;
    end
    if idx - 1 >= 1
        B_row(idx - 1) = a5;
    end
    B_row(idx) = a6;
    if idx + 1 <= num_points
        B_row(idx + 1) = a7;
    end
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_bdy_iN1(num_points, h, gamma, coeffs, idx)
    % 7-point format: coeffs = [a1 a2 a3 a4 a5 a6 alpha beta] (a7 eliminated)
    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    a4 = coeffs(4); a5 = coeffs(5); a6 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a7 = -(a1 + a2 + a3 + a4 + a5 + a6);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    A_row(idx) = 1;
    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    if idx - 2 >= 1
        A_row(idx - 2) = beta;
    end

    if idx - 6 >= 1
        B_row(idx - 6) = a1;
    end
    if idx - 5 >= 1
        B_row(idx - 5) = a2;
    end
    if idx - 4 >= 1
        B_row(idx - 4) = a3;
    end
    if idx - 3 >= 1
        B_row(idx - 3) = a4;
    end
    if idx - 2 >= 1
        B_row(idx - 2) = a5;
    end
    if idx - 1 >= 1
        B_row(idx - 1) = a6;
    end
    B_row(idx) = a7;
end

% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_internal(num_points, h, gamma, coeffs, idx)
    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    a5 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);
    a4 = -(a1 + a2 + a3 + a5 + a6 + a7);

    A_row = sparse(1, num_points);
    B_row = sparse(1, num_points);

    A_row(idx) = 1;
    if idx - 1 >= 1
        A_row(idx - 1) = alpha;
    end
    if idx + 1 <= num_points
        A_row(idx + 1) = alpha;
    end
    if idx - 2 >= 1
        A_row(idx - 2) = beta;
    end
    if idx + 2 <= num_points
        A_row(idx + 2) = beta;
    end

    B_row(idx) = a4;
    if idx - 3 >= 1
        B_row(idx - 3) = a1;
    end
    if idx - 2 >= 1
        B_row(idx - 2) = a2;
    end
    if idx - 1 >= 1
        B_row(idx - 1) = a3;
    end
    if idx + 1 <= num_points
        B_row(idx + 1) = a5;
    end
    if idx + 2 <= num_points
        B_row(idx + 2) = a6;
    end
    if idx + 3 <= num_points
        B_row(idx + 3) = a7;
    end
end
