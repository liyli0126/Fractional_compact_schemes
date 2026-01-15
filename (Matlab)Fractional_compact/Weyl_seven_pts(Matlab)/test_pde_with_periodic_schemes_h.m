function test_pde_with_periodic_schemes_h(gamma, k_wave, M)
% Input:
    %   gamma: fractional derivative order (e.g., 0.5, 1.0, 1.5)
    %   M: number of grid points (x_0, x_1, ..., x_{M-1}) covering [0, 1) with periodicity
    %      - Can be a scalar (single test) or a vector (multiple tests)
    %   k_wave: wave number for test function u(x) = sin(2π k_wave x) (default: 1)

    
    % Set default k_wave if not provided
    if nargin < 3 || isempty(k_wave)
        k_wave = 1;
    end

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
        [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
     
        
        [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, x, u_exact, u_num] = ...
            solve_single_pde_periodic(gamma, M, c_int, k_wave);
        
        % Plot exact vs numerical solution
        figure('Position', [50, 100, 800, 600]);
        plot(x, u_exact, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Exact Solution');
        hold on;
        plot(x, u_num, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
             'MarkerIndices', 1:max(1,floor(length(x)/20)):length(x), ...
             'MarkerFaceColor', 'r', 'DisplayName', 'Numerical Solution');
        hold off;
        xlabel('x', 'FontSize', 14);
        ylabel('u(x)', 'FontSize', 14);
        xlim([0, 1]);
        title(sprintf('Exact vs Numerical Solution: u(x)=sin(2\\pi*%.2f*x), \\gamma=%.2f', k_wave, gamma), ...
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
        title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f, M=%d', gamma, M), ...
              'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        set(gca, 'FontSize', 12);
        
        % Output results in formatted table
        fprintf('\n=== Results ===\n');
        fprintf('Testing PDE (gamma=%.2f) on u(x)=sin(2*pi*%.2f*x) with periodic boundary conditions\n', gamma, k_wave);
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
        
        fprintf('Testing PDE (gamma=%.2f) on u(x)=sin(2*pi*%.2f*x) with periodic boundary conditions\n', gamma, k_wave);
        fprintf('Grid points: ');
        fprintf('%d ', M_array);
        fprintf('\n\n');
        
        % Run tests for each M
        for i = 1:n_tests
            M = M_array(i);
            h = 1 / M;
            h_array(i) = h;
            
            % Get coefficients using h-based auto-selection for each M
            [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
            
            [L1_errors(i), L2_errors(i), Lmax_errors(i), ...
             relative_L1_errors(i), relative_L2_errors(i), relative_Lmax_errors(i), ...
             x_plot, u_exact_plot, u_num_plot] = ...
                solve_single_pde_periodic(gamma, M, c_int, k_wave);
            
            % Plot for last test
            if i == n_tests
                % Plot exact vs numerical solution
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
                title(sprintf('Exact vs Numerical Solution: u(x)=sin(2\\pi*%.2f*x), \\gamma=%.2f, N=%d', k_wave, gamma, M), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                legend('Location', 'best', 'FontSize', 12);
                grid on;
                set(gca, 'FontSize', 12);
                
                % Plot error
                err_plot = u_num_plot - u_exact_plot;
                figure('Position', [950, 100, 800, 600]);
                plot(x_plot, err_plot, 'r-', 'LineWidth', 2.5);
                xlabel('x', 'FontSize', 14);
                ylabel('Error (u_{num} - u_{exact})', 'FontSize', 14);
                xlim([0, 1]);
                title(sprintf('Error: u_{num} - u_{exact}, \\gamma=%.2f, N=%d', gamma, M), ...
                      'FontSize', 14, 'FontWeight', 'bold');
                grid on;
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

% -------------------------------------------------------------------------
% Helper function: solve single PDE for given M with periodic boundary conditions
% -------------------------------------------------------------------------
function [L1_err, L2_err, Lmax_err, rel_L1, rel_L2, rel_Lmax, x, u_exact, u_num] = ...
    solve_single_pde_periodic(gamma, M, c_int, k_wave)
    
    % Set default k_wave if not provided
    if nargin < 4 || isempty(k_wave)
        k_wave = 1;
    end
    
    h = 1 / M;
    % Grid points: x = (0:M-1)' * h to cover [0, 1) (periodic, exclude x=1)
    x = (0:M-1)' * h;

    % Build global LHS matrix A (for d) and RHS matrix B (for u) with
    % sparse matrix
    A_mat = spalloc(M, M, 5*M);  % compact: usually 3-5 diagonals
    B_mat = spalloc(M, M, 7*M);  % FD: up to 7-point stencil

    % Exact RHS of PDE: 
    u_exact =  sin(2*pi*k_wave*x);
    d_exact = (2*pi*k_wave)^gamma * sin(2*pi*k_wave*x + pi*gamma/2);
    f_exact = d_exact + u_exact;

    % Fill rows using periodic indexing
    for j = 1:M  % MATLAB index (1-based)
        % All points use internal scheme with periodic indexing
        [A_row, B_row] = build_row_internal_periodic(M, h, gamma, c_int, j);
        A_mat(j, :) = A_row;
        B_mat(j, :) = B_row;
    end

    % Solve (B + A) * u = A * f
    rhs = A_mat * f_exact;
    
    % Transfer to dense matrix, and the system (B_mat + A_mat) * u = rhs is well-conditioned
    B_mat_full = full(B_mat);
    A_mat_full = full(A_mat);
    B_mat_modified = B_mat_full +  A_mat_full;  
    
    % Check condition number and rank of modified B_mat
    cond_B = cond(B_mat_modified);
    
    % Solve (B_mat + A_mat) * u = rhs
    if cond_B > 1e10
        warning('(B_mat + lambda * A_mat) is ill-conditioned (cond=%e); results may be unstable', cond_B, M);
    end
    u_num = B_mat_modified \ rhs;

    err = u_num - u_exact;
    
    % Compute errors
    L1_err = h * sum(abs(err));
    L2_err = sqrt(h * sum(err.^2));
    Lmax_err = max(abs(err));
    
    % Compute relative errors
    u_exact_L1_norm = h * sum(abs(u_exact));
    u_exact_L2_norm = sqrt(h * sum(u_exact.^2));
    u_exact_Lmax_norm = max(abs(u_exact));
    
    % Calculate relative errors (avoid division by zero)
    if u_exact_L1_norm > eps
        rel_L1 = L1_err / u_exact_L1_norm;
    else
        rel_L1 = L1_err;  % If exact solution is zero, use absolute error
    end
    
    if u_exact_L2_norm > eps
        rel_L2 = L2_err / u_exact_L2_norm;
    else
        rel_L2 = L2_err;  % If exact solution is zero, use absolute error
    end
    
    if u_exact_Lmax_norm > eps
        rel_Lmax = Lmax_err / u_exact_Lmax_norm;
    else
        rel_Lmax = Lmax_err;  % If exact solution is zero, use absolute error
    end
    
end

% -------------------------------------------------------------------------
% Helper: build row for internal points with periodic boundary conditions
% -------------------------------------------------------------------------
function [A_row, B_row] = build_row_internal_periodic(M, h, gamma, coeffs, j)
    % j: MATLAB index (1-based)
    % coeffs = [a1,a2,a3,a5,a6,a7,alpha,beta] from solve_redu_inter_h
    %   LHS: d_j + alpha*(d_{j-1}+d_{j+1}) + beta*(d_{j-2}+d_{j+2})
    %   RHS: (1/h^gamma) * [ a1*(u_{j-3}-u_j) + a2*(u_{j-2}-u_j) + a3*(u_{j-1}-u_j)
    %                   + a5*(u_{j+1}-u_j) + a6*(u_{j+2}-u_j) + a7*(u_{j+3}-u_j) ]


    a1 = coeffs(1); a2 = coeffs(2); a3 = coeffs(3);
    a5 = coeffs(4); a6 = coeffs(5); a7 = coeffs(6);
    alpha = coeffs(7); beta = coeffs(8);

    a4 = -(a1 + a2 + a3 + a5 + a6 + a7);

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

    % RHS: (1/h^gamma) * [ a1*u_{j-3} + a2*u_{j-2} + a3*u_{j-1} + a5*u_{j+1} + a6*u_{j+2} + a7*u_{j+3}
    %               - (a1+a2+a3+a5+a6+a7)*u_j ]
    B_row(j) = a4 / h^gamma;
    B_row(idx_m3) = a1 / h^gamma;
    B_row(idx_m2) = a2 / h^gamma;
    B_row(idx_m1) = a3 / h^gamma;
    B_row(idx_p1) = a5 / h^gamma;
    B_row(idx_p2) = a6 / h^gamma;
    B_row(idx_p3) = a7 / h^gamma;

end
