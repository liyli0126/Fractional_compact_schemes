function [x_opt, A, b, L2_error] = solve_redu_inter_frequency_h(gamma, N, x_points_inter, plot_flag)
% Input:
%  gamma: fractional derivative order
%  h: grid spacing (if provided, interpolation points will be auto-selected based on h)
%  x_points_inter: interpolation points in frequency space [0, pi] 
%  plot_flag: if true, plot comparison and error figures 
% Output:
%  params = [a1 a2 a3 a5 a6 a7 alpha beta]
%  x_opt: optimized parameters
%  A: coefficient matrix
%  b: right-hand side vector
%  L2_error: L^2 error of the approximation 

    % Parameter validation and interpolation point selection
    if nargin < 2
        error('At least gamma and h (or x_points_inter) must be provided.');
    end
    
    % Parse optional plot_flag parameter
    if nargin < 4 || isempty(plot_flag)
        plot_flag = false;
    end
    
    h = 1/N;
    
    % Determine mode: check if second argument is h (scalar) or x_points_inter (vector)
    if isscalar(h) && h > 0 && h <= 1
        
        % Method: Scale by h to cover frequency range 
        w_max = 2*pi*h;  % Maximum frequency for one period in [0,1)
        w_max = min(w_max, pi);  % Cap at Nyquist frequency
           
        % Select points covering [w_max/8, w_max] with good distribution
        if w_max > pi/32
            % For reasonable h, use scaled points
            w_points = [
                max(w_max / 8, pi/128),   ... Low frequency (but not too small)
                w_max / 4,                ... Mid-low frequency
                w_max / 2,                ... Mid frequency
                min(w_max, 3*pi /4),          ...High frequency (capped at 4*pi/5 for stability)
                min(w_max, pi)
            ];
        else
            % For very small h, use fixed small points
            w_points = [pi/128, pi/64, pi/32, pi/16];
        end
        
        % Ensure points are in valid range and sorted
        w_points = max(pi/256, min(w_points, pi));  % Clamp to [pi/256, pi]
        w_points = sort(unique(w_points));
        
        % Ensure we have at least 4 points (if some are too close, add more)
        if length(w_points) < 4
            % Add more points if needed
            additional_points = linspace(pi/128, min(w_max*2, pi/4), 4);
            w_points = sort(unique([w_points, additional_points]));
            w_points = w_points(1:min(4, length(w_points)));  % Take first 4
        end
        
        
    elseif isvector(h) && ~isscalar(h)
        %  h is actually x_points_inter (backward compatibility)
        x_points_inter = h;
        w_points = x_points_inter(:);  % Ensure column vector
        w_points = sort(w_points);
        w_points = max(0, min(w_points, pi));
        
        if isempty(w_points) || length(w_points) < 4
            error('x_points_inter must have at least 4 points.');
        end
        
    else
        error('h must be a positive scalar <= 1 (grid spacing) or a vector (x_points_inter).');
    end
    
    % If both h and x_points_inter are provided, x_points_inter takes precedence
    if nargin >= 3 && ~isempty(x_points_inter) && length(x_points_inter) >= 4
        w_points = x_points_inter(:);
        w_points = sort(w_points);
        w_points = max(0, min(w_points, pi));
       
    end
    
    % Final validation
    if isempty(w_points) || length(w_points) < 4
        error('At least 4 interpolation points are required.');
    end
    
    % Check if points are too small (may cause ill-conditioning)
    min_w = min(w_points);
    if min_w < pi/128
        warning('Smallest interpolation point (%.6f) is very small. This may cause ill-conditioning.', min_w);
    end

    n_points = length(w_points);
    n_eq = 2 * n_points;   % each point gives real + imag equation
    n_var = 8;             % [a1,a2,a3,a5,a6,a7,alpha,beta, theta]
    
    % precompute trig constants
    s = sin(gamma * pi / 2);
    c = cos(gamma * pi / 2);

    % Target function (iw)^gamma
    target_r = @(x) c * x .^ gamma;
    target_i = @(x) s * x .^ gamma;

    % Define the basis function
    phi_r = cell(8, 1);
    phi_i = cell(8, 1);

    % Define a1
    phi_r{1} = @(x) cos(3*x) -1;
    phi_i{1} = @(x) -sin(3*x);

    % Define a2
    phi_r{2} = @(x) cos(2*x) - 1;
    phi_i{2} = @(x) -sin(2*x);

    % Define a3
    phi_r{3} = @(x) cos(x) - 1;
    phi_i{3} = @(x) -sin(x);

    % Define a5
    phi_r{4} = @(x) cos(x) - 1;
    phi_i{4} = @(x) sin(x);

    % Define a6
    phi_r{5} = @(x) cos(2*x) - 1;
    phi_i{5} = @(x) sin(2*x);

    % Define a7
    phi_r{6} = @(x) cos(3*x) - 1;
    phi_i{6} = @(x) sin(3*x);

    % Define alpha
    phi_r{7} = @(x) -2 * c * x.^gamma .* cos(x);
    phi_i{7} = @(x) -2 * s * x.^gamma .* cos(x);

    % Define beta
    phi_r{8} = @(x) -2 * c * x.^gamma .* cos(2*x);
    phi_i{8} = @(x) -2 * s * x.^gamma .* cos(2*x);

    
    % Initialize matrix and vector
    A = zeros(n_eq, n_var);
    b = zeros(n_eq, 1);
    
    % Fill in A and b: each point gives two equations (real and imaginary)
    for i = 1:n_points
        xi = w_points(i);
        % Real part
        A(2*i-1, :) = cellfun(@(f) f(xi), phi_r)';
        b(2*i-1) = target_r(xi);
        % Imag part
        A(2*i, :) = cellfun(@(f) f(xi), phi_i)';
        b(2*i) = target_i(xi);
    end

        % Solve: use least-squares if overdetermined or underdetermined
        x_opt = A \ b;
        
    % Compute L^2 error on fine grid
    x_fine = linspace(0, w_max, 1000)';
    exact_real_fine = target_r(x_fine);
    exact_imag_fine = target_i(x_fine);
    
    num_real_fine = zeros(size(x_fine));
    num_imag_fine = zeros(size(x_fine));
    for k = 1:n_var
        num_real_fine = num_real_fine + x_opt(k) * phi_r{k}(x_fine);
        num_imag_fine = num_imag_fine + x_opt(k) * phi_i{k}(x_fine);
    end
    
    % Compute L^2 error: sqrt(integral[0,pi] |exact - numerical|^2 dx)
    % where |exact - numerical|^2 = (real_err)^2 + (imag_err)^2
    err_real_fine = exact_real_fine - num_real_fine;
    err_imag_fine = exact_imag_fine - num_imag_fine;
    err_mag_sq = err_real_fine.^2 + err_imag_fine.^2;
    
    % Integrate using trapezoidal rule
    dx_fine = pi / (length(x_fine) - 1);
    L2_error = sqrt(dx_fine * (sum(err_mag_sq) - 0.5*(err_mag_sq(1) + err_mag_sq(end))));
    
    % Plot comparison and error figures (only if plot_flag is true)
    if plot_flag
        % Determine x_max for continuous plotting range [0, x_max]
        x_max = max(w_points);
        if x_max < pi
            x_max = pi;  % Extend to pi if needed
        end
        
        % Create continuous grid for plotting [0, x_max]
        x_plot = linspace(0, x_max, 1000)';
        
        % Compute exact values on continuous grid
        exact_real = target_r(x_plot);
        exact_imag = target_i(x_plot);
        
        % Compute numerical approximation on continuous grid
        num_real = zeros(size(x_plot));
        num_imag = zeros(size(x_plot));
        for k = 1:n_var
            num_real = num_real + x_opt(k) * phi_r{k}(x_plot);
            num_imag = num_imag + x_opt(k) * phi_i{k}(x_plot);
        end
        
        % Compute errors on continuous grid
        err_real = exact_real - num_real;
        err_imag = exact_imag - num_imag;
        
        % Compute absolute error magnitude: |D_h^{int} - (i*omega)^gamma|
        err_mag = sqrt(err_real.^2 + err_imag.^2);
        
        % Select 4 collocation points to highlight (take first 4)
        n_colloc = min(4, length(w_points));
        w_colloc = w_points(1:n_colloc);
        
        % Compute values at collocation points for highlighting
        exact_real_colloc = target_r(w_colloc);
        exact_imag_colloc = target_i(w_colloc);
        num_real_colloc = zeros(size(w_colloc));
        num_imag_colloc = zeros(size(w_colloc));
        for k = 1:n_var
            num_real_colloc = num_real_colloc + x_opt(k) * phi_r{k}(w_colloc);
            num_imag_colloc = num_imag_colloc + x_opt(k) * phi_i{k}(w_colloc);
        end
        err_real_colloc = exact_real_colloc - num_real_colloc;
        err_imag_colloc = exact_imag_colloc - num_imag_colloc;
        err_mag_colloc = sqrt(err_real_colloc.^2 + err_imag_colloc.^2);
        
        % Compute relative error: E_h(omega) = |D_h^{int}(omega) - (i*omega)^gamma| / (1 + |omega|^gamma)
        % Denominator: 1 + |omega|^gamma
        denom = 1 + abs(x_plot).^gamma;
        E_h = err_mag ./ denom;
        
        % Compute values at ALL collocation points (w_points) for plots 5 and 6
        exact_real_all = target_r(w_points);
        exact_imag_all = target_i(w_points);
        num_real_all = zeros(size(w_points));
        num_imag_all = zeros(size(w_points));
        for k = 1:n_var
            num_real_all = num_real_all + x_opt(k) * phi_r{k}(w_points);
            num_imag_all = num_imag_all + x_opt(k) * phi_i{k}(w_points);
        end
        err_real_all = exact_real_all - num_real_all;
        err_imag_all = exact_imag_all - num_imag_all;
        err_mag_all = sqrt(err_real_all.^2 + err_imag_all.^2);
        
        % Compute relative error at ALL collocation points
        denom_all = 1 + abs(w_points).^gamma;
        E_h_all = err_mag_all ./ denom_all;
        
        % Ensure all values are valid for semilogy plot (avoid 0, NaN, Inf)
        % Add small epsilon to avoid log(0) issues in semilogy plot
        % Use a larger epsilon relative to the data range to ensure visibility
        % Find the minimum non-zero value in the error data to set appropriate epsilon
        min_err = min(err_mag(err_mag > 0));
        if isempty(min_err) || min_err == 0
            min_err = 1e-10;  % Default minimum
        end
        eps_val = min(min_err * 1e-3, 1e-12);  % Use 0.1% of minimum or 1e-12, whichever is smaller
        err_mag_all = max(err_mag_all, eps_val);
        E_h_all = max(E_h_all, eps_val);
        
        % Debug: Check if all points have valid values
        fprintf('Checking collocation points visibility:\n');
        fprintf('  w_points: ');
        fprintf('%.6f ', w_points);
        fprintf('\n');
        fprintf('  err_mag_all: ');
        fprintf('%.2e ', err_mag_all);
        fprintf('\n');
        fprintf('  E_h_all: ');
        fprintf('%.2e ', E_h_all);
        fprintf('\n');
        
        % Print L^2 error in title or as text
        fprintf('Internal scheme: L^2 error = %.6e (gamma=%.2f, h=%.6e)\n', L2_error, gamma, h);
        fprintf('Number of collocation points: %d\n', length(w_points));
        fprintf('Error magnitudes at collocation points: ');
        fprintf('%.2e ', err_mag_all);
        fprintf('\n');
        
        % Plot 1: Real part comparison
        figure('Position', [30, 100, 600, 400]);
        plot(x_plot, exact_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Real Part');
        hold on;
        plot(x_plot, num_real, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical Real Part');
        % Highlight 4 collocation points with vertical lines and markers
        % Draw vertical lines without legend entry
        y_lims = ylim;
        for k = 1:n_colloc
            h_line = plot([w_colloc(k), w_colloc(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        plot(w_colloc, exact_real_colloc, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
       % plot(w_colloc, num_real_colloc, 'rs', 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'DisplayName', 'Numerical at collocation');
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('Real Part', 'FontSize', 12);
        title(sprintf('Real Part Comparison (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        xlim([0, x_max]);
        
        % Plot 2: Imaginary part comparison
        figure('Position', [150, 100, 600, 400]);
        plot(x_plot, exact_imag, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Imaginary Part');
        hold on;
        plot(x_plot, num_imag, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical Imaginary Part');
        % Highlight 4 collocation points with vertical lines and markers
        % Draw vertical lines without legend entry
        y_lims = ylim;
        for k = 1:n_colloc
            h_line = plot([w_colloc(k), w_colloc(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        plot(w_colloc, exact_imag_colloc, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
       % plot(w_colloc, num_imag_colloc, 'rs', 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'DisplayName', 'Numerical at collocation');
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('Imaginary Part', 'FontSize', 12);
        title(sprintf('Imaginary Part Comparison (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        xlim([0, x_max]);
        
        % Plot 3: Real part error
        figure('Position', [300, 100, 600, 400]);
        plot(x_plot, err_real, 'r-', 'LineWidth', 2, 'DisplayName', 'Real Part Error');
        hold on;
        plot(x_plot, zeros(size(x_plot)), 'k--', 'LineWidth', 1, 'DisplayName', 'Zero line');
        % Highlight 4 collocation points with vertical lines and markers
        % Draw vertical lines without legend entry
        y_lims = ylim;
        for k = 1:n_colloc
            h_line = plot([w_colloc(k), w_colloc(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        plot(w_colloc, err_real_colloc, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('Real Part Error', 'FontSize', 12);
        title(sprintf('Real Part Error (Exact - Numerical) (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        xlim([0, x_max]);
        
        % Plot 4: Imaginary part error
        figure('Position', [500, 100, 600, 400]);
        plot(x_plot, err_imag, 'r-', 'LineWidth', 2, 'DisplayName', 'Imaginary Part Error');
        hold on;
        plot(x_plot, zeros(size(x_plot)), 'k--', 'LineWidth', 1, 'DisplayName', 'Zero line');
        % Highlight 4 collocation points with vertical lines and markers
        % Draw vertical lines without legend entry
        y_lims = ylim;
        for k = 1:n_colloc
            h_line = plot([w_colloc(k), w_colloc(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        plot(w_colloc, err_imag_colloc, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('Imaginary Part Error', 'FontSize', 12);
        title(sprintf('Imaginary Part Error (Exact - Numerical) (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        xlim([0, x_max]);
        
        % Plot 5: L^2 absolute error
        figure('Position', [700, 100, 600, 400]);
        semilogy(x_plot, err_mag, 'r-', 'LineWidth', 2, 'DisplayName', 'L^2 Absolute Error');
        hold on;
        % Mark ALL collocation points (all points should be visible now with eps_val)
        semilogy(w_points, err_mag_all, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
        % Get y-axis limits after plotting all data (to ensure full coverage)
        y_lims = ylim;
        % Ensure y-axis includes all collocation points
        y_min = min([y_lims(1), min(err_mag_all)]);
        y_max = max([y_lims(2), max(err_mag_all)]);
        ylim([y_min, y_max]);
        y_lims = ylim;  % Update y_lims after adjustment
        % Draw vertical lines covering entire y-axis range for ALL collocation points
        % Use plot with y_lims to ensure lines span full y-axis in semilogy plot
        for k = 1:length(w_points)
            h_line = plot([w_points(k), w_points(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('L^2 Absolute Error', 'FontSize', 12);
        title(sprintf('L^2 Absolute Error (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        xlim([0, x_max]);
        
        % Plot 6: Relative error E_h(omega)
        figure('Position', [900, 100, 600, 400]);
        semilogy(x_plot, E_h, 'r-', 'LineWidth', 2, 'DisplayName', 'E_h(\omega)');
        hold on;
        % Mark ALL collocation points (all points should be visible now with eps_val)
        semilogy(w_points, E_h_all, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
        % Get y-axis limits after plotting all data (to ensure full coverage)
        % Force update of y-axis limits
        drawnow;
        y_lims = ylim;
        % Ensure y-axis includes all collocation points
        y_min = min([y_lims(1), min(E_h_all)]);
        y_max = max([y_lims(2), max(E_h_all)]);
        ylim([y_min, y_max]);
        y_lims = ylim;  % Update y_lims after adjustment
        % Draw vertical lines covering entire y-axis range for ALL collocation points
        % In semilogy plot, use plot with y_lims to ensure lines span full y-axis
        for k = 1:length(w_points)
            h_line = plot([w_points(k), w_points(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        % Ensure y-axis limits are maintained
        ylim(y_lims);
        hold off;
        xlabel('Frequency', 'FontSize', 12);
        ylabel('Relative Error E_h(\omega)', 'FontSize', 12);
        title(sprintf('Relative Error E_h(\\omega) (\\gamma=%.2f, N=%d)', gamma, N), 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        xlim([0, x_max]);
    end
end
