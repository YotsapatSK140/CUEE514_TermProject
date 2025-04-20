clear; clc; close all;

% Experiment 1: Image Deblurring
% This script performs image deblurring using several optimization algorithms.
% It generates a blurred and noisy version of a standard phantom image,
% and then attempts to recover the original image using different l1-l2 optimization techniques.
% The performance of these algorithms is then compared.

% Parameters
imageSize = 128;  % Define the size of the test image (128x128 pixels).  Larger sizes increase computational cost.
max_iter = 100;   % Set the maximum number of iterations for the optimization algorithms.  This is a crucial parameter.
x_true = phantom(imageSize); % Generate a standard phantom image of the specified size.  Good for testing deblurring.

% Generate blur kernel
h = fspecial('gaussian', [15 15], 2); % Create a 2D Gaussian filter kernel for blurring.
                                     % [15 15] is the size of the kernel (should be odd), 2 is the standard deviation (blur amount).
                                     %  Consider making kernel size a parameter.

% Define forward and adjoint operators
A = @(z) reshape(imfilter(reshape(z,imageSize,imageSize),h,'circular'),[],1);
% Define the forward operator A. It takes a vector z:
%   1. Reshapes it into a 2D image.
%   2. Applies the Gaussian blur using imfilter with circular boundary conditions (important for reducing artifacts).
%   3. Reshapes the blurred image back into a column vector.
At = A; % Symmetric operator.  For a symmetric blur kernel and circular convolution, the adjoint is the same as A.

% Generate noisy blurred image
y = A(x_true(:)) + 0.01 * randn(imageSize^2, 1);
% Generate the noisy observed data y:
%   1. Blur the true image:  A(x_true(:)).  x_true(:) converts the true image to a vector.
%   2. Additive white Gaussian noise: 0.01 * randn(imageSize^2, 1).
%      - 0.01 is the noise level (standard deviation).  Make this a parameter.
%      - randn generates a vector of Gaussian random noise with the same size as the image.

lambda = 0.001; % Set the regularization parameter lambda.
                % This controls the trade-off between data fidelity (fitting the noisy data)
                % and the sparsity-promoting term.  Crucial parameter to tune.

% Algorithms to run
algorithms = {@PCD, @PCD_CG, @PCD_SESOP, @SESOP, @SSF, @SSF_CG, @SSF_SESOP, @FISTA};
% Cell array containing function handles to different sparse recovery algorithms.
alg_names = {'PCD', 'PCD-CG', 'PCD-SESOP', 'SESOP', 'SSF', 'SSF-CG', 'SSF-SESOP', 'FISTA'};
% Cell array of strings containing the names of the algorithms (for plotting/display).  Good practice.

% Pre-allocate memory
results = cell(length(algorithms), 1); % Cell array to store reconstructed images.  Efficient.
history = cell(length(algorithms), 1); % Cell array to store objective function histories. Useful for convergence analysis.
snr_history = cell(length(algorithms), 1); % Cell array to store SNR histories.  Essential for quality assessment.

% Run the algorithms
for i = 1:length(algorithms) % Iterate over each algorithm.
    fprintf('Running %s...\n', alg_names{i}); % Display algorithm name.  Helpful for tracking progress.

    % z0: Initial guess.  Starting with a zero image is common.  Consider other initializations?
    z0 = zeros(imageSize^2, 1);

    % Run the selected algorithm:
    %   - A, At: Forward and adjoint operators.
    %   - y: Noisy data.
    %   - lambda: Regularization parameter.
    %   - z0: Initial guess.
    %   - max_iter: Maximum iterations.
    %   - x_true(:):  The true image (for SNR calculation *inside* the algorithm).  Good practice.
    [z, hist, snr] = algorithms{i}(A, At, y, lambda, z0, max_iter, x_true(:));

    results{i} = reshape(z, imageSize, imageSize); % Reshape the recovered vector z back into an image.
    history{i} = hist;                       % Store the objective function history.
    snr_history{i} = snr;                   % Store the SNR history.
end

% Plotting and Visualization
fig1 = figure; % Create a new figure.  Use separate figures for organization.
t = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact'); % Create a tiled layout.

% Display original and noisy images
nexttile; imshow(x_true, []); title('Original'); axis off;       % Original image (ground truth).
nexttile; imshow(reshape(y, imageSize, imageSize), []); title('Noisy'); axis off; % Noisy blurred image.

% Display reconstructed images
for i = 1:length(results)
    nexttile;
    imshow(results{i}, []);
    title(alg_names{i}, 'Interpreter', 'latex'); % Use LaTeX for title formatting (consistent).
    axis off;
end

% Plot performance metrics (Objective Gap and SNR)
[fig2, fig3] = plot_results(history, snr_history, alg_names); % Call the plotting function.  Good modularity.

%% Save Figures
% exportgraphics(fig1, 'exp1_den_recon.pdf', 'ContentType', 'vector');
% exportgraphics(fig2, 'exp1_den_metrics1.pdf', 'ContentType', 'vector');
% exportgraphics(fig3, 'exp1_den_metrics2.pdf', 'ContentType', 'vector');