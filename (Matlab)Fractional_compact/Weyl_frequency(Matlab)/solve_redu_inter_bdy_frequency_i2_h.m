function [x_opt, A, b, L2_error] = solve_redu_inter_bdy_frequency_i2_h(gamma, h, x_points_inter, plot_flag)
% Input:
%  gamma: fractional derivative order
%  h: grid spacing (if provided, interpolation points will be auto-selected based on h)
%  x_points_inter: interpolation points in frequency space [0, pi] (optional)
%  plot_flag: if true, plot comparison and error figures (optional, default: false)
% Output:
%  params = [a1 a3 a4 a5 a6 a7 alpha beta] (a2 eliminated, a2 = -(a1+a3+a4+a5+a6+a7))
%  x_opt: optimized parameters
%  A: coefficient matrix
%  b: right-hand side vector
%  L2_error: L^2 error of the approximation (optional)

    % Parameter validation and interpolation point selection
    if nargin < 2
        error('At least gamma and h (or x_points_inter) must be provided.');
    end
    
    % Parse optional plot_flag parameter
    if nargin < 4 || isempty(plot_flag)
        plot_flag = false;
    end


    
    % Determine mode: check if second argument is h (scalar) or x_points_inter (vector)
    if isscalar(h) && h > 0 && h <= 1
        
        % Method: Scale by h to cover frequency range 
        w_max = 2*pi*h;  % Maximum frequency for one period in [0,1)
        w_max = min(w_max, pi);  % Cap at Nyquist frequency
          
        % Select points covering [w_max/8, w_max] with good distribution
        % Boundary schemes need at least 4 points (8 parameters, 2 equations per point)
        if w_max > pi/32
            % For reasonable h, use scaled points
            x_points = [
                max(w_max / 8, pi/128),   ... Low frequency (but not too small)
                w_max / 4,                ... Mid-low frequency
                w_max / 2,                ... Mid frequency
                min(w_max, pi/4)          % High frequency (capped at pi/4 for stability)
            ];
        else
            % For very small h, use fixed small points
            x_points = [pi/128, pi/64, pi/32, pi/16];
        end
        
        % Ensure points are in valid range and sorted
        x_points = max(pi/256, min(x_points, pi));  % Clamp to [pi/256, pi]
        x_points = sort(unique(x_points(:)));  % Ensure column vector, then sort
        
        % Ensure we have at least 4 points (if some are too close, add more)
        if length(x_points) < 4
            % Add more points if needed
            additional_points = linspace(pi/128, min(w_max*2, pi/4), 4);
            additional_points = additional_points(:);  % Ensure column vector
            x_points = sort(unique([x_points; additional_points]));  % Vertical concatenation
            % Ensure at least 4 distinct points
            if length(x_points) < 4
                % Add evenly spaced points in valid range to reach 4
                range_min = max(pi/256, min(x_points));
                range_max = min(pi, max(pi/64, max(x_points)));
                if range_max <= range_min
                    range_max = range_min + pi/32;  % Ensure range is non-zero
                end
                additional_fill = linspace(range_min, range_max, 4)';
                x_points = sort(unique([x_points; additional_fill]));
            end
            % Ensure we keep at least 4 points
            if length(x_points) > 4
                x_points = x_points(1:4);  % Keep first 4 if more than needed
            end
        end
        
    elseif isvector(h) && ~isscalar(h)
        %  h is actually x_points_inter (backward compatibility)
        x_points_inter = h;
        x_points = x_points_inter(:);  % Ensure column vector
        x_points = sort(x_points);
        x_points = max(0, min(x_points, pi));
        
        if isempty(x_points) || length(x_points) < 4
            error('x_points_inter must have at least 4 points.');
        end
        
    else
        error('h must be a positive scalar <= 1 (grid spacing) or a vector (x_points_inter).');
    end
    
    % If both h and x_points_inter are provided, x_points_inter takes precedence
    if nargin >= 3 && ~isempty(x_points_inter) && length(x_points_inter) >= 4
        x_points = x_points_inter(:);
        x_points = sort(x_points);
        x_points = max(0, min(x_points, pi));
        
    end
    
    % Final validation
    if isempty(x_points) || length(x_points) < 4
        error('At least 4 interpolation points are required for boundary schemes.');
    end
    
    % Check if points are too small (may cause ill-conditioning)
    min_x = min(x_points);
    if min_x < pi/128
        warning('Smallest interpolation point (%.6f) is very small. This may cause ill-conditioning.', min_x);
    end

    n_points = length(x_points);
    n_eq = 2 * n_points;   % each point gives real + imag equation
    n_var = 8;             % [a1 a3 a4 a5 a6 a7 alpha beta] (a2 eliminated, a2 = -(a1+a3+a4+a5+a6+a7))
    
    % precompute trig constants
    s = sin(gamma * pi / 2);
    c = cos(gamma * pi / 2);

    % Target function (iw)^gamma
    target_r = @(x) c * x .^ gamma;
    target_i = @(x) s * x .^ gamma;

    % Define the basis function (cos(kx) - 1 form, matching internal scheme)
    phi_r = cell(8, 1);
    phi_i = cell(8, 1);

    % Define a1: exp(-ix) - 1
    phi_r{1} = @(x) cos(x) - 1;
    phi_i{1} = @(x) -sin(x);

    % Define a2: 
   % phi_r{2} = @(x) 1;
   % phi_i{2} = @(x) 0;

    % Define a3: 
    phi_r{2} = @(x) cos(x) - 1;
    phi_i{2} = @(x) sin(x);

    % Define a4: 
    phi_r{3} = @(x) cos(2*x) - 1;
    phi_i{3} = @(x) sin(2*x);

    % Define a5: 
    phi_r{4} = @(x) cos(3*x) - 1;
    phi_i{4} = @(x) sin(3*x);

    % Define a6: 
    phi_r{5} = @(x) cos(4*x) - 1;
    phi_i{5} = @(x) sin(4*x);

    % Define a7: 
    phi_r{6} = @(x) cos(5*x) - 1;
    phi_i{6} = @(x) sin(5*x);

    % Define alpha
    phi_r{7} = @(x) - 2 * c * x.^gamma .* cos(x);
    phi_i{7} = @(x) - 2 * s * x.^gamma .* cos(x);

    % Define beta
    phi_r{8} = @(x) - c * x.^gamma .* cos(2*x) + s * x.^gamma .* sin(2*x);
    phi_i{8} = @(x) - s * x.^gamma .* cos(2*x) - c * x.^gamma .* sin(2*x);

    
    % Initialize matrix and vector
    A = zeros(n_eq, n_var);
    b = zeros(n_eq, 1);
    
    % Fill in A and b: each point gives two equations (real and imaginary)
    for i = 1:n_points
        xi = x_points(i);
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
    x_max = max(min(pi, 2*pi*h), max(x_points) * 1.1);  % Add 10% margin to include all points
    x_fine = linspace(0, x_max, 1000)';
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
        % Create fine grid for plotting
        x_plot = linspace(0, max(x_points), 200)';
        
        % Compute exact values
        exact_real = target_r(x_plot);
        exact_imag = target_i(x_plot);
        
        % Compute numerical approximation
        num_real = zeros(size(x_plot));
        num_imag = zeros(size(x_plot));
        for k = 1:n_var
            num_real = num_real + x_opt(k) * phi_r{k}(x_plot);
            num_imag = num_imag + x_opt(k) * phi_i{k}(x_plot);
        end
        
        % Compute errors
        err_real = exact_real - num_real;
        err_imag = exact_imag - num_imag;
        err_magnitude = sqrt(err_real.^2 + err_imag.^2);
        
        % Compute relative error: |exact - numerical| / (|w|^gamma + eps)
        eps_rel = 1e-10;  % Small value to avoid division by zero
        rel_error = err_magnitude ./ (x_plot.^gamma + eps_rel);
        
        % Compute relative error at interpolation points
        exact_real_points = target_r(x_points);
        exact_imag_points = target_i(x_points);
        num_real_points = zeros(size(x_points));
        num_imag_points = zeros(size(x_points));
        for k = 1:n_var
            num_real_points = num_real_points + x_opt(k) * phi_r{k}(x_points);
            num_imag_points = num_imag_points + x_opt(k) * phi_i{k}(x_points);
        end
        err_magnitude_points = sqrt((exact_real_points - num_real_points).^2 + (exact_imag_points - num_imag_points).^2);
        rel_error_points = err_magnitude_points ./ (x_points.^gamma + eps_rel);
        
        % Print L^2 error in title or as text
        fprintf('Internal scheme: L^2 error = %.6e (gamma=%.2f, h=%.6e)\n', L2_error, gamma, h);
        
        % Plot 1: Real part comparison
        figure('Position', [30, 100, 600, 400]);
        plot(x_plot, exact_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Real Part');
        hold on;
        plot(x_plot, num_real, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical Real Part');
        plot(x_points, target_r(x_points), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation Points');
        % Add gray dashed vertical lines at interpolation points
        y_lim = ylim;
        for i = 1:length(x_points)
            h_line = plot([x_points(i), x_points(i)], y_lim, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        hold off;
        xlabel('x (frequency)', 'FontSize', 12);
        ylabel('Real Part', 'FontSize', 12);
        title(sprintf('Real Part Comparison (\\gamma=%.2f, N=%d)', gamma, 1/h), 'FontSize', 12);
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        
        % Plot 2: Imaginary part comparison
        figure('Position', [200, 100, 600, 400]);
        plot(x_plot, exact_imag, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Imaginary Part');
        hold on;
        plot(x_plot, num_imag, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical Imaginary Part');
        plot(x_points, target_i(x_points), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation Points');
        % Add gray dashed vertical lines at interpolation points
        y_lim = ylim;
        for i = 1:length(x_points)
            h_line = plot([x_points(i), x_points(i)], y_lim, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
            h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
        hold off;
        xlabel('x (frequency)', 'FontSize', 12);
        ylabel('Imaginary Part', 'FontSize', 12);
        title(sprintf('Imaginary Part Comparison (\\gamma=%.2f, N=%d)', gamma, 1/h), 'FontSize', 12);
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        
        % Plot 3: Real part error
        figure('Position', [400, 100, 600, 400]);
        plot(x_plot, err_real, 'r-', 'LineWidth', 2);
        hold on;
        plot(x_points, zeros(size(x_points)), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'DisplayName', 'Interpolation Points');
        % Add gray dashed vertical lines at interpolation points
        y_lim = ylim;
        for i = 1:length(x_points)
            plot([x_points(i), x_points(i)], y_lim, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        end
        hold off;
        xlabel('x (frequency)', 'FontSize', 12);
        ylabel('Real Part Error', 'FontSize', 12);
        title(sprintf('Real Part Error (Exact - Numerical) (\\gamma=%.2f)', gamma), 'FontSize', 12);
        grid on;
        legend('Error', 'Interpolation Points', 'Location', 'best', 'FontSize', 10);
        
        % Plot 4: Imaginary part error
        figure('Position', [650, 100, 600, 400]);
        plot(x_plot, err_imag, 'r-', 'LineWidth', 2);
        hold on;
        plot(x_points, zeros(size(x_points)), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'DisplayName', 'Interpolation Points');
        % Add gray dashed vertical lines at interpolation points
        y_lim = ylim;
        for i = 1:length(x_points)
            plot([x_points(i), x_points(i)], y_lim, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        end
        hold off;
        xlabel('x (frequency)', 'FontSize', 12);
        ylabel('Imaginary Part Error', 'FontSize', 12);
        title(sprintf('Imaginary Part Error (Exact - Numerical) (\\gamma=%.2f)', gamma), 'FontSize', 12);
        grid on;
        legend('Error', 'Interpolation Points', 'Location', 'best', 'FontSize', 10);
        
        % Plot 5: Relative error
        figure('Position', [900, 100, 600, 400]);
        plot(x_plot, rel_error, 'r-', 'LineWidth', 2);
        hold on;
        plot(x_points, rel_error_points, 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'DisplayName', 'Interpolation Points');
        set(gca, 'YScale', 'log');  % Use log scale for better visualization
        % Add gray dashed vertical lines at interpolation points
        y_lim = ylim;
        for i = 1:length(x_points)
            plot([x_points(i), x_points(i)], y_lim, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        end
        hold off;
        xlabel('x (frequency)', 'FontSize', 12);
        ylabel('Relative Error', 'FontSize', 12);
        title(sprintf('Relative Error  (\\gamma=%.2f, N=%d, j=i_2)', gamma, 1/h), 'FontSize', 12);
        grid on;
        legend('Relative Error', 'Interpolation Points', 'Location', 'best', 'FontSize', 10);
    end
end
