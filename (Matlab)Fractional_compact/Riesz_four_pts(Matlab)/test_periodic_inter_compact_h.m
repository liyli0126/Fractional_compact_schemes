function test_periodic_inter_compact_h(gamma, k_wave, M_array)
% Test the compact scheme for the Riesz derivative
% on periodic domain [0, 1) using u(x) = sin(2*pi*k*x)
%
% Input:
%   gamma: fractional derivative order (0 < gamma <= 2)
%   k_wave: wave number
%   M_array: scalar or vector of grid points (number of points in [0,1))

    if nargin < 2 || isempty(k_wave)
        k_wave = 1;
    end
    if nargin < 3 || isempty(M_array)
        M_array = 64;
    end

    if isscalar(M_array)
        M_array = M_array(1);
        single_test = true;
    else
        single_test = false;
    end

    L = 1;  % domain length

    if single_test
        M = M_array;
        h = L / M;
        [c_int, ~, ~] = solve_redu_inter_four_h(gamma, h,[],false);
        
        [L1_error, L2_error, max_error, ~, ~, ~, x, exact_deriv, d_num, ~] = ...
            compute_single_test(gamma, k_wave, M, L, c_int);
        
        fprintf('Testing Weyl Derivative (gamma=%.2f) on u(x)=sin(2π*%.2f*x) with M=%d points\n', gamma, k_wave, M);
        fprintf('L1=%.4e, L2=%.4e, Lmax=%.4e\n', L1_error, L2_error, max_error);
        
        % Plot (extend to x=1 via periodicity)
        x_ext = [x; 1];
        exact_ext = [exact_deriv; exact_deriv(1)];
        d_num_ext = [d_num; d_num(1)];
        
        figure('Position', [100,100,800,400]);
        plot(x_ext, exact_ext, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Derivative' );
        hold on;
        plot(x_ext, d_num_ext, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
             'MarkerIndices', 1:max(1,floor(length(x_ext)/20)):length(x_ext), ...
             'MarkerFaceColor','r', 'DisplayName', 'Numerical Derivative');
        hold off;
        xlabel('x'); ylabel('D^\gamma u(x)');
        title(sprintf('Periodic Test: gamma=%.2f, k=%.2f', gamma, k_wave), 'Interpreter');
        legend('Location','best'); grid on; xlim([0,1]);
        
    else
        % Convergence test
        n = length(M_array);
        L1_err     = zeros(n,1);
        L2_err     = zeros(n,1);
        Linf_err   = zeros(n,1);
        relL1_err  = zeros(n,1);
        relL2_err  = zeros(n,1);
        relLinf_err = zeros(n,1);
        h_arr      = zeros(n,1);
        
        fprintf('Testing gamma=%.2f, k=%.2f\n', gamma, k_wave);
        for i = 1:n
            M = M_array(i);
            h = L / M;
            h_arr(i) = h;
            [c_int, ~, ~] = solve_redu_inter_four_h(gamma, h,[],false);
            [L1_err(i), L2_err(i), Linf_err(i), ...
             relL1_err(i), relL2_err(i), relLinf_err(i)] = ...
                compute_single_test(gamma, k_wave, M, L, c_int);
        end
        
        % Output results table (absolute and relative errors, no rate)
        fprintf('===========================================================================================================\n');
        fprintf('                                    RESULTS TABLE\n');
        fprintf('===========================================================================================================\n');
        fprintf('%-6s %-10s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'M', 'h', 'L1 Error', 'L2 Error', 'Max Error', 'Rel L1', 'Rel L2', 'Rel Max');
        fprintf('-----------------------------------------------------------------------------------------------------------\n');
        for i = 1:n
            fprintf('%-6d %-10.3e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                    M_array(i), h_arr(i), ...
                    L1_err(i), L2_err(i), Linf_err(i), ...
                    relL1_err(i), relL2_err(i), relLinf_err(i));
        end

        
        % Final comparison plot (largest M)
        M_plot = M_array(end);
        h = L / M_plot;
        [c_int, ~, ~] = solve_redu_inter_four_h(gamma, h,[],false);
        [~, ~, ~, ~, ~, ~, x, exact_deriv, d_num, ~] = ...
            compute_single_test(gamma, k_wave, M_plot, L, c_int);
        
        x_ext = [x; 1];
        exact_ext = [exact_deriv; exact_deriv(1)];
        d_num_ext = [d_num; d_num(1)];
        
        M_fine = 1000;
        x_fine = linspace(0,1,M_fine)';
        [~, exact_fine] = compute_periodic_test_function(gamma, k_wave, x_fine);
        exact_fine(end) = exact_fine(1); % enforce periodicity
        
        figure('Position', [300,300,800,500]);
        plot(x_fine, exact_fine, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Exact');
        hold on;
        plot(x_ext, d_num_ext, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
             'MarkerIndices', 1:max(1,floor(length(x_ext)/20)):length(x_ext), ...
             'MarkerFaceColor','r', 'DisplayName', 'Numerical');
        hold off;
        xlabel('x'); ylabel('D^\gamma u(x)');
        title(sprintf('Exact vs Numerical (M=%d, $\\gamma=%.2f$)', M_plot, gamma), 'Interpreter','latex');
        legend('Location','best'); grid on; xlim([0,1]);
    end
end

% -------------------------------------------------------------------------
function [u, exact_deriv] = compute_periodic_test_function(gamma, k_wave, x)
    u = sin(2*pi*k_wave*x);
    exact_deriv = -abs(2*pi*k_wave)^gamma * sin(2*pi*k_wave*x);
end

% -------------------------------------------------------------------------
function varargout = compute_single_test(gamma, k_wave, M, L, c_int)
    h = L / M;
    x = (0:M-1)' * h;  % periodic grid: [0, 1)
    [u, exact_deriv] = compute_periodic_test_function(gamma, k_wave, x);
    
    % Extract coefficients from c_int = [a1,a2,a3,alpha,beta]
    a1 = c_int(1); a2 = c_int(2); a3 = c_int(3);
    
    alpha = c_int(4); beta = c_int(5);
    a4 = -2*(a1 + a2 + a3); 
    
    % Build RHS 
    RHS = zeros(M,1);
    for j = 1:M
        j_m3 = mod(j-4, M) + 1;  % j-3
        j_m2 = mod(j-3, M) + 1;  % j-2
        j_m1 = mod(j-2, M) + 1;  % j-1
        j_p1 = mod(j  , M) + 1;  % j+1
        j_p2 = mod(j+1, M) + 1;  % j+2
        j_p3 = mod(j+2, M) + 1;  % j+3
        
        RHS(j) = (1/h^gamma) * ( ...
            a1*u(j_m3) + a2*u(j_m2) + a3*u(j_m1) + ...
            a4*u(j) + ...
            a3*u(j_p1) + a2*u(j_p2) + a1*u(j_p3) ...
        );
    end
    
    % Build LHS matrix A 
    A = spalloc(M, M, 5*M);
    for j = 1:M
        A(j,j) = 1;
        j_m1 = mod(j-2, M) + 1;
        j_p1 = mod(j  , M) + 1;
        j_m2 = mod(j-3, M) + 1;
        j_p2 = mod(j+1, M) + 1;
        A(j, j_m1) = alpha; A(j, j_p1) = alpha;
        A(j, j_m2) = beta;  A(j, j_p2) = beta;
    end
    
    d_num = A \ RHS;
    
    % Errors
    err = d_num - exact_deriv;
    abs_err = abs(err);
    L1 = h * sum(abs_err);
    L2 = sqrt(h * sum(err.^2));
    Linf = max(abs_err);
    
    exact_L1 = h * sum(abs(exact_deriv));
    exact_L2 = sqrt(h * sum(exact_deriv.^2));
    exact_Linf = max(abs(exact_deriv));
    relL1 = L1 / (exact_L1 + eps);
    relL2 = L2 / (exact_L2 + eps);
    relLinf = Linf / (exact_Linf + eps);
    
    if nargout <= 6
        varargout = {L1, L2, Linf, relL1, relL2, relLinf};
    else
        varargout = {L1, L2, Linf, relL1, relL2, relLinf, x, exact_deriv, d_num, abs_err};
    end
end