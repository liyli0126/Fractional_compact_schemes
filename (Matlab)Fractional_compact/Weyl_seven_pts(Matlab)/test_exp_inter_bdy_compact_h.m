function test_exp_inter_bdy_compact_h(gamma, N_array)
% Test the compact scheme for the Weyl derivative
% on [-1, 1] using u(x) = e^{lambda x} - 1 with compact boundary.
%
% Input:
%   gamma   : fractional derivative order (0 < gamma <= 2)
%   N_array : scalar or vector of interval numbers (h = 1/N, num_points = N+1)

    if nargin < 2 || isempty(N_array)
        N_array = 64;
    end

    if isscalar(N_array)
        N_array = N_array(1);
        single_test = true;
    else
        single_test = false;
    end

    L = 21;  % domain length (from -10 to 1)

    if single_test
        N = N_array;
        h = L / N;
       
        [L1_error, L2_error, max_error, ~, ~, ~, x, exact_deriv, d_num, ~, point_errors] = ...
            compute_single_test(gamma,  N, L);
        
        fprintf('Testing Weyl Derivative (gamma=%.2f) on u(x)=e^{\\lambda x}-1 with N=%d intervals (%d points)\n', gamma,  N, N+1);
        fprintf('L1=%.4e, L2=%.4e, L∞=%.4e\n', L1_error, L2_error, max_error);
        
        % Print errors at specific boundary and interior points
        if ~isempty(point_errors) && ~isempty(point_errors.indices)
            fprintf('\n===========================================================================================================\n');
            fprintf('                    ERROR ANALYSIS AT BOUNDARY AND INTERIOR POINTS\n');
            fprintf('===========================================================================================================\n');
            fprintf('%-8s %-12s %-15s %-15s %-15s %-15s\n', 'Index', 'x', 'Exact', 'Numerical', 'Error', 'Rel Error');
            fprintf('-----------------------------------------------------------------------------------------------------------\n');
            for i = 1:length(point_errors.indices)
                idx = point_errors.indices(i);
                fprintf('%-8d %-12.6f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
                    idx, x(idx), exact_deriv(idx), d_num(idx), ...
                    point_errors.errors(i), point_errors.rel_errors(i));
            end
            fprintf('===========================================================================================================\n');
        end
        
        figure('Position', [100,100,800,400]);
        plot(x, exact_deriv, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Derivative' );
        hold on;
        plot(x, d_num, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
            'MarkerIndices',1:5:length(x), 'MarkerFaceColor','r', 'DisplayName', 'Numerical Derivative');
        hold off;
        xlabel('x'); ylabel('D^\gamma u(x)');
        title(sprintf('Boundary Test: gamma=%.2f', gamma));
        legend('Location','best'); grid on; xlim([-20,1]);
        
    else
        % Convergence test: multiple N, output error table (no convergence rate)
        n = length(N_array);
        L1_err     = zeros(n,1);
        L2_err     = zeros(n,1);
        Lmax_err   = zeros(n,1);
        relL1_err  = zeros(n,1);
        relL2_err  = zeros(n,1);
        relLmax_err = zeros(n,1);
        h_arr      = zeros(n,1);
        
        for i = 1:n
            N = N_array(i);
            h = L / N;
            h_arr(i) = h;
            [L1_err(i), L2_err(i), Lmax_err(i), ...
             relL1_err(i), relL2_err(i), relLmax_err(i)] = ...
                compute_single_test(gamma, N, L);
        end

        % Output results table (absolute and relative errors, no rate)
        fprintf('\n%6s %10s %12s %12s %12s %12s %12s %12s\n', ...
                'N', 'h', 'L1', 'L2', 'Lmax', 'Rel L1', 'Rel L2', 'Rel Lmax');
        for i = 1:n
            fprintf('%6d %10.3e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n', ...
                    N_array(i), h_arr(i), ...
                    L1_err(i), L2_err(i), Lmax_err(i), ...
                    relL1_err(i), relL2_err(i), relLmax_err(i));
        end


        % Final comparison plot (largest N)
        N_plot = N_array(end);
        h = L / N_plot;
        [~, ~, ~, ~, ~, ~, x, exact_deriv, d_num, ~, point_errors] = ...
            compute_single_test(gamma, N_plot, L);
        
        % Print errors at specific boundary and interior points for convergence test
        if ~isempty(point_errors) && ~isempty(point_errors.indices)
            fprintf('\n===========================================================================================================\n');
            fprintf('                    ERROR ANALYSIS AT BOUNDARY AND INTERIOR POINTS (N=%d)\n', N_plot);
            fprintf('===========================================================================================================\n');
            fprintf('%-8s %-12s %-15s %-15s %-15s %-15s\n', 'Index', 'x', 'Exact', 'Numerical', 'Error', 'Rel Error');
            fprintf('-----------------------------------------------------------------------------------------------------------\n');
            for i = 1:length(point_errors.indices)
                idx = point_errors.indices(i);
                fprintf('%-8d %-12.6f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
                    idx, x(idx), exact_deriv(idx), d_num(idx), ...
                    point_errors.errors(i), point_errors.rel_errors(i));
            end
            fprintf('===========================================================================================================\n');
        end
        
        N_fine = 1000;
        x_fine = linspace(-20, 1, N_fine+1)';
        [~, exact_fine] = compute_exp_test_function(gamma, x_fine);
        
        figure('Position', [100, 100, 800, 600]);
        plot(x_fine, exact_fine, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Exact Derivative');
        hold on;
        plot(x, d_num, 'r--o', 'LineWidth', 2.5, 'MarkerSize', 4, ...
             'MarkerIndices', 1:max(1,floor(length(x)/20)):length(x), ...
             'MarkerFaceColor', 'r', 'DisplayName', 'Numerical Derivative');
        hold off;
        xlabel('x', 'FontSize', 14);
        ylabel('Fractional Derivative Value', 'FontSize', 14);
        xlim([-20, 1]);  
        title(sprintf('Exact vs Numerical Derivative: \\gamma=%.2f', ...
                     gamma), 'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 12);
        grid on;
        set(gca, 'FontSize', 12);
    end

end

% -------------------------------------------------------------------------
function [u, exact_deriv] = compute_exp_test_function(gamma, x)
    lambda = 1;
    u = exp(lambda * x) ;
    exact_deriv =  lambda .^gamma * exp(lambda * x);
end


%function [u, exact_deriv] = compute_exp_test_function(gamma, x)
%    u = ones(size(x));
%    exact_deriv = zeros(size(x));
%end

% -------------------------------------------------------------------------
function varargout = compute_single_test(gamma, N, L)
    h = L / N;
    
    % Check if h <= 1 (required by solve_redu_inter_h)
    if h > 1
        error('Grid spacing h = %.6f exceeds 1. Please use N >= L (N >= %d) to ensure h <= 1.', h, L);
    end
    
    num_points = N + 1;  % Total grid points: x_0 to x_N
    x = -20+ (0:N)' * h;  % num_points points on [-10,1]
    [u, exact_deriv] = compute_exp_test_function(gamma, x);


        % Coefficients
        [c_int, ~, ~] = solve_redu_inter_h(gamma, h);
        [c_i0, ~, ~]  = solve_redu_inter_bdy_i1_h(gamma, h);
        [c_i1, ~, ~]  = solve_redu_inter_bdy_i2_h(gamma, h);
        [c_i2, ~, ~]  = solve_redu_inter_bdy_i3_h(gamma, h);
        [c_iN_1, ~, ~] = solve_redu_inter_bdy_iN_h(gamma, h);
        if N >= 6
            [c_iN_2, ~, ~] = solve_redu_inter_bdy_iN_1_h(gamma, h);
        else
            c_iN_2 = [];
        end
        [c_iN, ~, ~] = solve_redu_inter_bdy_iN1_h(gamma, h);

         % Internal 7-point coefficients
        a1i = c_int(1); a2i = c_int(2); a3i = c_int(3);
        a5i = c_int(4); a6i = c_int(5); a7i = c_int(6);
        a4i = -(a1i+a2i+a3i+a5i+a6i+a7i);
        [alphai, betai] = deal(c_int(7), c_int(8));
  

        % Left boundary coefficients (7-point) - 1-based indexing
        % idx=1 (x_0)
        a2_1 = c_i0(1); a3_1 = c_i0(2); a4_1 = c_i0(3); a5_1 = c_i0(4);
        a6_1 = c_i0(5); a7_1 = c_i0(6);
        [alpha1, beta1] = deal(c_i0(7), c_i0(8));
        a1_1 = -(a2_1+a3_1+a4_1+a5_1+a6_1+a7_1);
    
        % idx=2 (x_1)
        a1_2 = c_i1(1); a3_2 = c_i1(2); a4_2 = c_i1(3); a5_2 = c_i1(4);
        a6_2 = c_i1(5); a7_2 = c_i1(6);
        [alpha2, beta2] = deal(c_i1(7), c_i1(8));
        a2_2 = -(a1_2+a3_2+a4_2+a5_2+a6_2+a7_2);
    
        % idx=3 (x_2)
        a1_3 = c_i2(1); a2_3 = c_i2(2); a4_3 = c_i2(3); a5_3 = c_i2(4);
        a6_3 = c_i2(5); a7_3 = c_i2(6);
        [alpha3, beta3] = deal(c_i2(7), c_i2(8));
        a3_3 = -(a1_3+a2_3+a4_3+a5_3+a6_3+a7_3);
    
    
        % Right boundary coefficients (7-point) - 1-based indexing
        % idx=N+1 (x_N)
        a1_N1 = c_iN(1); a2_N1 = c_iN(2); a3_N1 = c_iN(3); a4_N1 = c_iN(4);
        a5_N1 = c_iN(5); a6_N1 = c_iN(6);
        [alphaN1, betaN1] = deal(c_iN(7), c_iN(8));
        a7_N1 = -(a1_N1+a2_N1+a3_N1+a4_N1+a5_N1+a6_N1);
        
        % idx=N (x_{N-1}) - 7-point format
        a1_N = c_iN_1(1); a2_N = c_iN_1(2); a3_N = c_iN_1(3);
        a4_N = c_iN_1(4); a5_N = c_iN_1(5); a7_N = c_iN_1(6);
        [alphaN, betaN] = deal(c_iN_1(7), c_iN_1(8));
        a6_N = -(a1_N+a2_N+a3_N+a4_N+a5_N+a7_N);
    
  
        % idx=N-1 (x_{N-2}) - 7-point format
        a1_N_1 = c_iN_2(1); a2_N_1 = c_iN_2(2); a3_N_1 = c_iN_2(3);
        a4_N_1 = c_iN_2(4); a6_N_1 = c_iN_2(5); a7_N_1 = c_iN_2(6);
        [alphaN_1, betaN_1] = deal(c_iN_2(7), c_iN_2(8));
        a5_N_1 = -(a1_N_1+a2_N_1+a3_N_1+a4_N_1+a6_N_1+a7_N_1);
    
        % RHS matrix B
        B = spalloc(num_points, num_points, 7*num_points);
        for j = 1:num_points
            switch j
                case 1
                    % idx=1 (x_0) - 7-point format
                    B(j,j) = a1_1;      % B(1,1) = a1_1
                    if num_points>=2, B(j,j+1) = a2_1; end
                    if num_points>=3, B(j,j+2) = a3_1; end
                    if num_points>=4, B(j,j+3) = a4_1; end
                    if num_points>=5, B(j,j+4) = a5_1; end
                    if num_points>=6, B(j,j+5) = a6_1; end
                    if num_points>=7, B(j,j+6) = a7_1; end
                case 2
                    % idx=2 (x_1)
    
                    if num_points>=2, B(j,j-1) = a1_2; end 
                    B(j,j) =  a2_2; % B(2,2) = a2_2
                    if num_points>=3, B(j,j+1) = a3_2; end
                    if num_points>=4, B(j,j+2) = a4_2; end
                    if num_points>=5, B(j,j+3) = a5_2; end
                    if num_points>=6, B(j,j+4) = a6_2; end
                    if num_points>=7, B(j,j+5) = a7_2; end
                case 3
                    % idx=3 (x_2)
                    if num_points>=2, B(j,j-2) = a1_3; end
                    if num_points>=2, B(j,j-1) = a2_3; end
                    B(j,j) = a3_3;      % B(3,3) = a3_3
                    if num_points>=4, B(j,j+1) = a4_3; end
                    if num_points>=5, B(j,j+2) = a5_3; end
                    if num_points>=6, B(j,j+3) = a6_3; end
                    if num_points>=7, B(j,j+4) = a7_3; end
                case num_points
                    % idx=N+1 (x_N) - 7-point format
                    if num_points>=7, B(j,j-6) = a1_N1; end
                    if num_points>=6, B(j,j-5) = a2_N1; end
                    if num_points>=5, B(j,j-4) = a3_N1; end
                    if num_points>=4, B(j,j-3) = a4_N1; end
                    if num_points>=3, B(j,j-2) = a5_N1; end
                    if num_points>=2, B(j,j-1) = a6_N1; end
                    B(j,j) = a7_N1;    
                case num_points - 1
                    % idx=N (x_{N-1}) - 7-point format
                    if num_points>=7, B(j,j-5) = a1_N; end
                    if num_points>=6, B(j,j-4) = a2_N; end
                    if num_points>=5, B(j,j-3) = a3_N; end
                    if num_points>=4, B(j,j-2) = a4_N; end
                    if num_points>=3, B(j,j-1) = a5_N; end
                    B(j,j) = a6_N;
                    if num_points>=2, B(j,j+1) = a7_N; end
                case num_points - 2
                    % idx=N-1 (x_{N-2}) - 7-point format
                    if num_points>=7, B(j,j-4) = a1_N_1; end
                    if num_points>=6, B(j,j-3) = a2_N_1; end
                    if num_points>=5, B(j,j-2) = a3_N_1; end
                    if num_points>=4, B(j,j-1) = a4_N_1; end
                    B(j,j) = a5_N_1;
                    if num_points>=3, B(j,j+1) = a6_N_1; end
                    if num_points>=2, B(j,j+2) = a7_N_1; end
     
                otherwise
                    % Internal points 
                    if j>=4, B(j,j-3) = a1i; end
                    if j>=3, B(j,j-2) = a2i; end
                    if j>=2, B(j,j-1) = a3i; end
                    B(j,j) = a4i;
                    if j<=num_points-1, B(j,j+1) = a5i; end
                    if j<=num_points-2, B(j,j+2) = a6i; end
                    if j<=num_points-3, B(j,j+3) = a7i; end
            end
        end
    
        RHS = (1/h^gamma) * (B * u);
    
        % LHS matrix A
        A = spalloc(num_points, num_points, 5*num_points);
        for j = 1:num_points
            switch j
                case 1
                    % idx=1 (x_0)
                    A(j,j) = 1;
                    if num_points>=2, A(j,j+1) = alpha1; end
                    if num_points>=3, A(j,j+2) = beta1; end
                case 2
                    % idx=2 (x_1)
                    if num_points>=2, A(j,j-1) = alpha2; end
                    A(j,j) = 1;
                    if num_points>=3, A(j,j+1) = alpha2; end
                    if num_points>=4, A(j,j+2) = beta2; end
                case 3
                    % idx=3 (x_2)
                    if num_points>=3, A(j,j-2) = beta3; end
                    if num_points>=2, A(j,j-1) = alpha3; end
                    A(j,j) = 1;
                    if num_points>=4, A(j,j+1) = alpha3; end
                    if num_points>=5, A(j,j+2) = beta3; end
                case num_points
                    % idx=N+1 (x_N)
                    if num_points>=3, A(j,j-2) = betaN1; end
                    if num_points>=2, A(j,j-1) = alphaN1; end
                    A(j,j) = 1;
                case num_points - 1
                    % idx=N (x_{N-1})
                    if num_points>=3, A(j,j-2) = betaN; end
                    if num_points>=2, A(j,j-1) = alphaN; end
                    A(j,j) = 1;
                    if j+1 <= num_points, A(j,j+1) = alphaN; end
                case num_points - 2
                     % idx=N-1 (x_{N-2}) - boundary format
                    if num_points>=4, A(j,j-2) = betaN_1; end
                    if num_points>=3, A(j,j-1) = alphaN_1; end
                    A(j,j) = 1;
                    if j+1 <= num_points, A(j,j+1) = alphaN_1; end
                    if j+2 <= num_points, A(j,j+2) = betaN_1; end
                otherwise
                    % Internal points (idx=4 to idx=N-3)
                    A(j,j-2) = betai; A(j,j-1) = alphai; A(j,j) = 1;
                    A(j,j+1) = alphai; A(j,j+2) = betai;
            end
        end


    d_num = A \ RHS;

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

    % Compute errors at specific boundary and interior points
    % Note: num_points = N + 1, where N is the number of intervals
    point_errors = struct();
    point_errors.indices = [];
    point_errors.errors = [];
    point_errors.rel_errors = [];
    point_errors.labels = {};
    
    N = num_points - 1;  % Number of intervals
    
    if N >= 6
        % Left boundary points: indices 1, 2, 3 (x_0, x_1, x_2)
        left_indices = [1, 2, 3];
        left_labels = {'x_0 (left)', 'x_1 (left)', 'x_2 (left)'};
        
        % Right boundary points: indices num_points-2, num_points-1, num_points
        % (x_{N-2}, x_{N-1}, x_N) - only if N >= 6, otherwise num_points-2 uses internal scheme
        right_indices = [num_points-2, num_points-1, num_points];
        right_labels = {'x_{N-2} (right)', 'x_{N-1} (right)', 'x_N (right)'};
        
        % Interior point: middle point
        interior_idx = floor((num_points + 1) / 2);
        interior_label = sprintf('x_{%d} (interior)', interior_idx - 1);
        
        % Combine all indices: 6 boundary points + 1 interior point = 7 points total
        all_indices = [left_indices, interior_idx, right_indices];
        all_labels = [left_labels, {interior_label}, right_labels];
        
        % Compute errors
        for i = 1:length(all_indices)
            idx = all_indices(i);
            if idx >= 1 && idx <= num_points
                point_errors.indices(end+1) = idx;
                point_errors.errors(end+1) = abs_err(idx);
                point_errors.rel_errors(end+1) = abs_err(idx) / (abs(exact_deriv(idx)) + eps);
                point_errors.labels{end+1} = all_labels{i};
            end
        end
    elseif N >= 3
        % If N < 6, we still have left boundary points 1, 2, 3 and right boundary points
        % But num_points-2 uses internal scheme, so we only include num_points-1 and num_points
        left_indices = [1, 2, 3];
        left_labels = {'x_0 (left)', 'x_1 (left)', 'x_2 (left)'};
        
        right_indices = [num_points-1, num_points];
        right_labels = {sprintf('x_{%d} (right)', N-1), sprintf('x_%d (right)', N)};
        
        % Interior point if possible
        if num_points >= 5
            interior_idx = floor((num_points + 1) / 2);
            interior_label = sprintf('x_{%d} (interior)', interior_idx - 1);
            all_indices = [left_indices, interior_idx, right_indices];
            all_labels = [left_labels, {interior_label}, right_labels];
        else
            all_indices = [left_indices, right_indices];
            all_labels = [left_labels, right_labels];
        end
        
        % Compute errors
        for i = 1:length(all_indices)
            idx = all_indices(i);
            if idx >= 1 && idx <= num_points
                point_errors.indices(end+1) = idx;
                point_errors.errors(end+1) = abs_err(idx);
                point_errors.rel_errors(end+1) = abs_err(idx) / (abs(exact_deriv(idx)) + eps);
                point_errors.labels{end+1} = all_labels{i};
            end
        end
    elseif num_points >= 2
        % Very few points: just use available boundary points
        available_indices = [1, num_points];
        available_labels = {'x_0 (left)', sprintf('x_%d (right)', N)};
        
        for i = 1:length(available_indices)
            idx = available_indices(i);
            point_errors.indices(end+1) = idx;
            point_errors.errors(end+1) = abs_err(idx);
            point_errors.rel_errors(end+1) = abs_err(idx) / (abs(exact_deriv(idx)) + eps);
            point_errors.labels{end+1} = available_labels{i};
        end
    end

    if nargout <= 6
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax};
    elseif nargout <= 10
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax, x, exact_deriv, d_num, abs_err};
    else
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax, x, exact_deriv, d_num, abs_err, point_errors};
    end
end