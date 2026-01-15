function test_exp_inter_bdy_diff_gamma_compact_h(N, gamma_array)
% Scan gamma for a fixed N to test the compact non-periodic scheme
% on [0, 1] using u(x) = exp(lambda x).
%
% Input:
%   N           : fixed number of intervals (h = 1/N, num_points = N+1)
%   gamma_array : vector of gamma values in (0,2]

if nargin < 1 || isempty(N)
    N =64;
end

if nargin < 2 || isempty(gamma_array)
    gamma_array = 0.1:0.1:2.0;
end

N = round(N(1));
if N <= 0
    error('N must be a positive integer.');
end

L = 11;              
h = L / N;

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
    [L1_err(i), L2_err(i), Lmax_err(i), ...
     relL1_err(i), relL2_err(i), relLmax_err(i)] = ...
        compute_single_test(gamma, N, L);
end

% |Lmax| vs gamma 
valid = ~isnan(Lmax_err);

    figure('Position', [200, 150, 800, 400]);
    plot(gamma_array(valid), Lmax_err(valid), 'ro-', ...
     'LineWidth', 1.5, 'MarkerFaceColor', 'r');

    xlabel('$\gamma$', 'Interpreter', 'latex');
    ylabel('$L_\infty$ Error (Absolute)', 'Interpreter', 'latex');
    title(sprintf('Absolute $L_\\infty$ Error vs $\\gamma$ (N = %d)', N), ...
      'Interpreter', 'latex');


    grid on;
    set(gca, 'FontSize', 12);


    % Relative Lmax vs gamma
    figure('Position', [200, 600, 800, 400]);
    plot(gamma_array(valid), relLmax_err(valid), 'bs-', ...
         'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    xlabel('$\gamma$', 'Interpreter', 'latex');
     ylabel('$L_\infty$ Error (Relative)', 'Interpreter', 'latex');
    title(sprintf('Relative $L_\\infty$ Error vs $\\gamma$ (N = %d)', N), ...
      'Interpreter', 'latex');

    grid on;
    set(gca, 'FontSize', 12);




% gamma–error
fprintf('=====================================================================\n');
fprintf('         ERROR vs GAMMA (N = %d, h = %.3e)\n', N, h);
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

%  exact vs numerical 
if n_gamma >= 1
    gamma_extremes = [gamma_array(1), gamma_array(end)];
    for k = 1:length(gamma_extremes)
        gamma = gamma_extremes(k);
        if gamma <= 0 || gamma > 2
            continue;
        end

        [~, ~, ~, ~, ~, ~, x, exact_deriv, d_num, ~] = ...
            compute_single_test(gamma, N, L);

        figure('Position', [300 + 100*(k-1), 100, 800, 400]);
        plot(x, exact_deriv, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact');
        hold on;
        plot(x, d_num, 'r--o', 'MarkerSize', 3, 'MarkerFaceColor', 'r', ...
             'DisplayName', 'Numerical');
        hold off;
        title(sprintf('\\gamma = %.1f (N = %d)', gamma, N));
        xlabel('x'); ylabel('D^\gamma u(x)');
        legend('Location','best'); grid on;
        xlim([-10,1]);
    end
end

end

% -------------------------------------------------------------------------
function [u, exact_deriv] = compute_exp_test_function(gamma, x)
    lambda = 1;
    u = exp(lambda * x) ;
    exact_deriv =  lambda .^gamma * exp(lambda * x);
end


% -------------------------------------------------------------------------
function varargout = compute_single_test(gamma, N, L)
    h = L / N;
    num_points = N + 1;  % Total grid points: x_0 to x_N
    x = -10+(0:N)' * h;  % num_points points: x(1)=x_0, x(N+1)=x_N
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

    if nargout <= 6
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax};
    else
        varargout = {L1, L2, Lmax, relL1, relL2, relLmax, x, exact_deriv, d_num, abs_err};
    end
end