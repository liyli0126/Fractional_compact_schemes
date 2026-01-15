function [x_opt, A, b] = solve_redu_inter_four_frequency_h(gamma, h, x_points_inter,do_plot)
% Input:
%  gamma: fractional derivative order
%  h: grid spacing (if provided, interpolation points will be auto-selected based on h)
%  x_points_inter: interpolation points in frequency space [0, pi] 
% Output:
%  params = [a1 a2 a3 a5 a6 a7 alpha beta]
%  x_opt: optimized parameters
%  A: coefficient matrix
%  b: right-hand side vector

    % Parameter validation and interpolation point selection
    if nargin < 2
        error('At least gamma and h (or x_points_inter) must be provided.');
    end
    
    % Determine mode: check if second argument is h (scalar) or x_points_inter (vector)
    if isscalar(h) && h > 0 && h <= 1
        
        % Method: Scale by h to cover frequency range 
        w_max = 2*pi*h;  % Maximum frequency for one period in [0,1)
        w_max = min(w_max, pi);  % Cap at Nyquist frequency
           
        % Select points covering [w_max/8, w_max] with good distribution
        if w_max > pi/32
            % For reasonable h, use scaled points
            x_points = [
                max(w_max / 8, pi/128),   ... Low frequency (but not too small)
                w_max / 4,                ... Mid-low frequency
                w_max / 2,                ... Mid frequency
                min(w_max, 3*pi /4)          % High frequency (capped at 4*pi/5 for stability)
            ];
        else
            % For very small h, use fixed small points
            x_points = [pi/128, pi/64, pi/32, pi/16];
        end
        
        % Ensure points are in valid range and sorted
        x_points = max(pi/256, min(x_points, pi));  % Clamp to [pi/256, pi]
        x_points = sort(unique(x_points));
        
        % Ensure we have at least 4 points (if some are too close, add more)
        if length(x_points) < 4
            % Add more points if needed
            additional_points = linspace(pi/128, min(w_max*2, pi/4), 4);
            x_points = sort(unique([x_points, additional_points]));
            x_points = x_points(1:min(4, length(x_points)));  % Take first 4
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
        error('At least 4 interpolation points are required.');
    end
    
    % Check if points are too small (may cause ill-conditioning)
    min_x = min(x_points);
    if min_x < pi/128
        warning('Smallest interpolation point (%.6f) is very small. This may cause ill-conditioning.', min_x);
    end

    n_points = length(x_points);
    n_eq = n_points;  
    n_var = 5;             % [a1, a2, a3, alpha, beta] (symmetric: a5=a3, a6=a2, a7=a1)
    
    % Target function: Riesz derivative -|w|^gamma
    target_r = @(x) abs(x) .^ gamma;

    % Define the basis function (5 functions for symmetric 7-point scheme)
    phi_r = cell(5, 1);

    % Define a1 (symmetric to a7)
    phi_r{1} = @(x) 2*(cos(3*x) - 1);

    % Define a2 (symmetric to a6)
    phi_r{2} = @(x) 2*(cos(2*x) - 1);

    % Define a3 (symmetric to a5)
    phi_r{3} = @(x) 2*(cos(x) - 1);

    % Define alpha
    phi_r{4} = @(x) 2 * abs(x).^gamma .* cos(x);

    % Define beta
    phi_r{5} = @(x) 2 * abs(x).^gamma .* cos(2*x);
    
    % Initialize matrix and vector
    A = zeros(n_eq, n_var);
    b = zeros(n_eq, 1);
    
    % Fill in A and b: each point gives one equation (real part only for Riesz)
    for i = 1:n_points
        xi = x_points(i);
        % Real part only (Riesz is pure real)
        A(i, :) = cellfun(@(f) f(xi), phi_r)';
        b(i) = target_r(xi);
    end

    % Solve: use least-squares if overdetermined or underdetermined
    x_opt_5 = A \ b;
    
    % Expand to 5 parameters: [a1, a2, a3,  alpha, beta]
    x_opt = [x_opt_5(1); x_opt_5(2); x_opt_5(3); ...  % a1, a2, a3
             x_opt_5(4); x_opt_5(5)];                  % alpha, beta

    if do_plot
    % Determine frequency range for plotting
    % Ensure x_max covers all collocation points
    x_max = max(min(pi, 2*pi*h), max(x_points) * 1.1);  % Add 10% margin to include all points
    x_plot = linspace(0, x_max, 1000)';  % Dense frequency points for plotting
    
    % Compute exact solution (real part only for Riesz)
    exact_real = target_r(x_plot);
    
    % Compute numerical solution using the optimized parameters
    % Numerical solution: sum of basis functions weighted by x_opt
    num_real = zeros(size(x_plot));
    for k = 1:5
        num_real = num_real + x_opt(k) * phi_r{k}(x_plot);
    end
    
    % Values at collocation points
    exact_real_colloc = target_r(x_points);
    num_real_colloc = zeros(size(x_points));
    for k = 1:5
        num_real_colloc = num_real_colloc + x_opt(k) * phi_r{k}(x_points);
    end
    
    % Compute errors
    err_mag = abs(exact_real - num_real);
    E_h = err_mag ./ (abs(exact_real) + eps);  % Relative error
    
    % Errors at collocation points
    err_mag_all = abs(exact_real_colloc - num_real_colloc);
    E_h_all = err_mag_all ./ (abs(exact_real_colloc) + eps);
    
    % Plot 1: Real part comparison
    figure('Position', [30, 100, 600, 400]);
    plot(x_plot, exact_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Exact Real Part');
    hold on;
    plot(x_plot, num_real, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical Real Part');
    % Highlight 5 collocation points with vertical lines and markers
    y_lims = ylim;
    for k = 1:length(x_points)
        h_line = plot([x_points(k), x_points(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
        h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    plot(x_points, exact_real_colloc, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'DisplayName', 'Collocation points');
    hold off;
    xlabel('Frequency', 'FontSize', 12);
    ylabel('Real-Valued', 'FontSize', 12);
    title(sprintf('Exact vs Numerical Symbol Comparison (\\gamma=%.2f, N=%d)', gamma, 1/h), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    xlim([0, x_max]);
    
    % Plot 2: L^2 absolute error
    figure('Position', [400, 100, 600, 400]);
    semilogy(x_plot, err_mag, 'r-', 'LineWidth', 2, 'DisplayName', 'L^2 Absolute Error');
    hold on;
    % Calculate y-axis range to include all collocation points
    y_data_min = min([min(err_mag_all), min(err_mag)]);
    y_data_max = max([max(err_mag_all), max(err_mag)]);
    % Add margin for semilogy (multiply/divide by a factor)
    y_min = max(eps, y_data_min * 0.5);  % Ensure positive and add margin below
    y_max = y_data_max * 2;  % Add margin above
    ylim([y_min, y_max]);
    % Get y-axis limits for vertical lines
    y_lims = ylim;
    % Create a dummy line for legend (dashed line style for interpolation point position)
    h_legend_line = plot([NaN, NaN], [NaN, NaN], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'DisplayName', 'Interpolation point position');
    % Draw vertical lines covering entire y-axis range for ALL collocation points
    % Use plot (linear coordinates) - MATLAB handles the coordinate system conversion
    for k = 1:length(x_points)
        h_line = plot([x_points(k), x_points(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
        h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    % Mark ALL collocation points individually to ensure visibility
    if length(x_points) > 0
        for k = 1:length(x_points)
            semilogy(x_points(k), err_mag_all(k), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'HandleVisibility', 'off');
        end
    end
    hold off;
    xlabel('Frequency', 'FontSize', 12);
    ylabel('L^2 Absolute Error', 'FontSize', 12);
    title(sprintf('L^2 Absolute Error (\\gamma=%.2f, N=%d)', gamma, 1/h), 'FontSize', 12);
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    xlim([0, x_max]);
    
    % Plot 3: Relative error E_h(omega)
    figure('Position', [800, 100, 600, 400]);
    semilogy(x_plot, E_h, 'r-', 'LineWidth', 2, 'DisplayName', 'E_h(\omega)');
    hold on;
    % Calculate y-axis range to include all collocation points
    y_data_min = min([min(E_h_all), min(E_h)]);
    y_data_max = max([max(E_h_all), max(E_h)]);
    % Add margin for semilogy (multiply/divide by a factor)
    y_min = max(eps, y_data_min * 0.5);  % Ensure positive and add margin below
    y_max = y_data_max * 2;  % Add margin above
    ylim([y_min, y_max]);
    % Get y-axis limits for vertical lines
    y_lims = ylim;
    % Create a dummy line for legend (dashed line style for interpolation point position)
    h_legend_line = plot([NaN, NaN], [NaN, NaN], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'DisplayName', 'Interpolation  position');
    % Draw vertical lines covering entire y-axis range for ALL collocation points
    % Use plot (linear coordinates) - MATLAB handles the coordinate system conversion
    for k = 1:length(x_points)
        h_line = plot([x_points(k), x_points(k)], y_lims, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
        h_line.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    % Mark ALL collocation points individually to ensure visibility
    if length(x_points) > 0
        for k = 1:length(x_points)
            %semilogy(x_points(k), E_h_all(k), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'HandleVisibility', 'off');
        end
    end
    hold off;
    xlabel('Frequency', 'FontSize', 12);
    ylabel('Relative Error E_h(\omega)', 'FontSize', 12);
    title(sprintf('Relative Error E_h(\\omega) (\\gamma=%.2f, N=%d)', gamma, 1/h), 'FontSize', 12);
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    xlim([0, x_max]);
    end  % end if do_plot

end