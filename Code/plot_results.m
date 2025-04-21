function [fig1, fig2] = plot_results(history, snr_history, alg_names)
    % plot_results - Generates plots to compare the performance of different algorithms.
    %
    % Inputs:
    %   history     - A cell array where each cell contains a vector of objective function values
    %                 recorded at each iteration for a specific algorithm.
    %   snr_history - A cell array where each cell contains a vector of Signal-to-Noise Ratio (SNR)
    %                 values recorded at each iteration for a specific algorithm.
    %   alg_names   - A cell array of strings containing the names of the corresponding algorithms.
    %
    % Outputs:
    %   fig1        - Handle to the figure containing the plot of the objective gap versus iteration.
    %   fig2        - Handle to the figure containing the plot of SNR versus iteration.

    numAlgs = length(history); % Get the number of algorithms being compared.
    colorMap = turbo(numAlgs); % Generate a colormap (turbo) with a distinct color for each algorithm.
                               % 'turbo' is a built-in MATLAB colormap designed to be perceptually uniform.

    % Find best objective value across all algorithms and iterations.
    % This is used to plot the "objective gap" (how far each algorithm is from the best achieved value).
    f_best = min(cellfun(@min, history));
    % 'cellfun(@min, history)' applies the 'min' function to each cell in the 'history' cell array,
    % returning a vector of the minimum objective value achieved by each algorithm.
    % The outer 'min' then finds the overall minimum of these values.

    % Create the first figure: Objective Gap vs Iteration
    fig1 = figure; % Create a new figure.
    hold on;     % Enable holding the current plot so that subsequent plots are added to the same axes.
    for i = 1:length(history) % Loop through each algorithm's history.
        plot(history{i}-f_best, 'color', colorMap(i,:), 'LineWidth', 1.4, 'DisplayName', alg_names{i}, 'LineWidth', 1.5);
        % Plot the objective gap for the i-th algorithm.
        % 'history{i}-f_best' calculates the difference between the objective value at each iteration
        % and the best objective value found across all algorithms.
        % 'color', colorMap(i,:) sets the plot color using the i-th color from the generated colormap.
        % 'LineWidth' sets the thickness of the plotted line.
        % 'DisplayName' sets the name that will appear in the legend.
    end
    set(gca, 'YScale', 'log'); % Set the y-axis scale to logarithmic. This is often useful for visualizing
                               % the convergence rate, especially when the objective function decreases rapidly initially.
    title('Objective Gap vs Iteration', 'Interpreter','latex'); % Set the title of the plot, using LaTeX for formatting.
    xlabel('Iteration', 'Interpreter','latex');                % Set the label for the x-axis.
    ylabel('$f(x) - f^*$', 'Interpreter','latex');           % Set the label for the y-axis, using LaTeX for the mathematical notation.
    legend(alg_names, 'Location', 'best'); % Display a legend using the algorithm names, placed at the 'best' location
                                          % automatically determined by MATLAB to avoid overlapping with the data.
    grid on; % Turn on the grid lines for better readability.

    % Create the second figure: SNR vs Iteration
    fig2 = figure; % Create a new figure.
    hold on;     % Enable holding the current plot.
    for i = 1:length(snr_history) % Loop through each algorithm's SNR history.
        plot(snr_history{i}, 'color', colorMap(i,:), 'LineWidth', 1.4, 'DisplayName', alg_names{i}, 'LineWidth', 1.5);
        % Plot the SNR values for the i-th algorithm at each iteration.
        % The color and line properties are set similarly to the objective gap plot.
    end
    title('SNR vs Iteration', 'Interpreter','latex'); % Set the title of the plot.
    xlabel('Iteration', 'Interpreter','latex');        % Set the label for the x-axis.
    ylabel('SNR (dB)', 'Interpreter','latex');       % Set the label for the y-axis, indicating Signal-to-Noise Ratio in decibels.
    legend(alg_names, 'Location', 'best'); % Display the legend.
    grid on; % Turn on the grid.
end