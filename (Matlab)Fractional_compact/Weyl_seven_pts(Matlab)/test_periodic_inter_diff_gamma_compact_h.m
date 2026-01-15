function test_periodic_inter_diff_gamma_compact_h(M, k_wave, gamma_array)
% Scan gamma for a fixed M to test the compact periodic scheme
% on [0, 1) using u(x) = sin(2*pi*k*x).
%
% Input:
%   M           : fixed number of grid points in [0,1)
%   k_wave      : wave number (default: 1)
%   gamma_array : vector of gamma values in (0,2]

    if nargin < 1 || isempty(M)
        M = 64;
    end
    if nargin < 2 || isempty(k_wave)
        k_wave = 1;
    end
    if nargin < 3 || isempty(gamma_array)
        gamma_array = 0.1:0.1:2.0;
    end

    M = round(M(1));
    if M <= 0
        error('M must be a positive integer.');
    end

    L = 1;                  % domain length
    h = L / M;

    % convert gamma_array to row vector, for plotting
    gamma_array = gamma_array(:).';
    n_gamma = length(gamma_array);

    % Allocate error arrays
    L1_err      = zeros(n_gamma, 1);
    L2_err      = zeros(n_gamma, 1);
    Lmax_err    = zeros(n_gamma, 1);
    relL1_err   = zeros(n_gamma, 1);
    relL2_err   = zeros(n_gamma, 1);
    relLmax_err = zeros(n_gamma, 1);

    for i = 1:n_gamma
        gamma = gamma_array(i);
        if gamma <= 0 || gamma > 2
            warning('gamma out of (0,2] range: %.2f', gamma);
            L1_err(i) = NaN;  L2_err(i) = NaN;  Lmax_err(i) = NaN;
            relL1_err(i) = NaN; relL2_err(i) = NaN; relLmax_err(i) = NaN;
            continue;
        end

        fprintf('  gamma = %.2f\n', gamma);
        [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
        [L1_err(i), L2_err(i), Lmax_err(i), ...
         relL1_err(i), relL2_err(i), relLmax_err(i)] = ...
            compute_single_test(gamma, k_wave, M, L, c_int);
    end

    % |Lmax| vs gamma 
    valid = ~isnan(Lmax_err);

    figure('Position', [200, 150, 800, 400]);
    plot(gamma_array(valid), Lmax_err(valid), 'ro-', ...
         'LineWidth', 1.5, 'MarkerFaceColor', 'r');
    xlabel('$\gamma$', 'Interpreter', 'latex');
    ylabel('$L_\infty$ Error (Absolute)', 'Interpreter', 'latex');
    title(sprintf('Absolute $L_\\infty$ Error vs $\\gamma$ (N = %d)', M), ...
      'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');  



    figure('Position', [200, 600, 800, 400]);
    plot(gamma_array(valid), relLmax_err(valid), 'bs-', ...
     'LineWidth', 1.5, 'MarkerFaceColor', 'b');

    xlabel('$\gamma$', 'Interpreter', 'latex');
    ylabel('$L_\infty$ Error (Relative)', 'Interpreter', 'latex');
    title(sprintf('Relative $L_\\infty$ Error vs $\\gamma$ (N = %d)', M), ...
      'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');  

    

    % gammaerror table
    fprintf('\n');
    fprintf('=====================================================================\n');
    fprintf('         ERROR vs GAMMA (M = %d, h = %.3e)\n', M, h);
    fprintf('=====================================================================\n');

    
    fprintf('%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
            'gamma', 'L1', 'L2', 'Lmax', 'Rel L1', 'Rel L2', 'Rel Lmax');
    fprintf('---------------------------------------------------------------------\n');
    for i = 1:n_gamma
        if ~isnan(Lmax_err(i))
            fprintf('%.2f     %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n', ...
                    gamma_array(i), L1_err(i), L2_err(i), Lmax_err(i), ...
                    relL1_err(i), relL2_err(i), relLmax_err(i));
        end
    end

    % exact vs numerical for extreme gamma values
    if n_gamma >= 1
        gamma_extremes = [gamma_array(1), gamma_array(end)];
        for k = 1:length(gamma_extremes)
            gamma = gamma_extremes(k);
            if gamma <= 0 || gamma > 2
                continue;
            end

            [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
            [~, ~, ~, ~, ~, ~, x, exact_deriv, d_num, ~] = ...
                compute_single_test(gamma, k_wave, M, L, c_int);

            % extend to x=1 via periodicity for plotting
            x_ext = [x; 1];
            exact_ext = [exact_deriv; exact_deriv(1)];
            d_num_ext = [d_num; d_num(1)];

            figure('Position', [300 + 100*(k-1), 100, 800, 400]);
            plot(x_ext, exact_ext, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact');
            hold on;
            plot(x_ext, d_num_ext, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
                 'MarkerIndices', 1:max(1,floor(length(x_ext)/20)):length(x_ext), ...
                 'MarkerFaceColor', 'r', 'DisplayName', 'Numerical');
            hold off;
            title(sprintf('\\gamma = %.1f (M = %d, k = %.2f)', gamma, M, k_wave));
            xlabel('x'); ylabel('D^\\gamma u(x)');
            legend('Location','best'); grid on;
            xlim([0,1]);
        end
    end

end

% -------------------------------------------------------------------------
function [u, exact_deriv] = compute_periodic_test_function(gamma, k_wave, x)
    u = sin(2*pi*k_wave*x);
    exact_deriv = (2*pi*k_wave)^gamma * sin(2*pi*k_wave*x + pi*gamma/2);
end

% -------------------------------------------------------------------------
function varargout = compute_single_test(gamma, k_wave, M, L, c_int)
    h = L / M;
    x = (0:M-1)' * h;  % periodic grid: [0, 1)
    [u, exact_deriv] = compute_periodic_test_function(gamma, k_wave, x);
    
    % Extract coefficients from c_int = [a1,a2,a3,a5,a6,a7,alpha,beta]
    a1 = c_int(1); a2 = c_int(2); a3 = c_int(3);
    a5 = c_int(4); a6 = c_int(5); a7 = c_int(6);
    alpha = c_int(7); beta = c_int(8);
    a4 = -(a1 + a2 + a3 + a5 + a6 + a7); 
    
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
            a5*u(j_p1) + a6*u(j_p2) + a7*u(j_p3) ...
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
    Lmax = max(abs_err);
    
    exact_L1 = h * sum(abs(exact_deriv));
    exact_L2 = sqrt(h * sum(exact_deriv.^2));
    exact_Lmax = max(abs(exact_deriv));
    relL1 = L1 / (exact_L1 + eps);
    relL2 = L2 / (exact_L2 + eps);
    relLmax = Lmax / (exact_Lmax + eps);
    
    if nargout <= 6
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax};
    else
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax, x, exact_deriv, d_num, abs_err};
    end
end
