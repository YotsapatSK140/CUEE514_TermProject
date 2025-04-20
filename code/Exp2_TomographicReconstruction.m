clear; clc; close all;

% Experiment 2: Tomographic Reconstruction
% This script performs tomographic reconstruction using several optimization algorithms.
% It generates a sinogram (Radon transform) of a standard phantom image, adds noise,
% and then reconstructs the image from the noisy sinogram using different l1-l2 optimization techniques.
% The performance of these algorithms is then compared.

% Parameters
imageSize = 128;  % Define the size of the image to be reconstructed (128x128 pixels).
max_iter = 100;   % Set the maximum number of iterations for the reconstruction algorithms.  Important parameter.
lambda = 0.001;  % Set the regularization parameter lambda.  Controls sparsity vs. data fidelity.  Crucial parameter.
theta = 0:2:178; % Define the angles (in degrees) at which the projections were taken.
                 % This ranges from 0 to 178 degrees with a step of 2 degrees.  Fewer angles = faster, but lower quality.

% True image
x_true = phantom(imageSize); % Generate a standard Shepp-Logan phantom image.  A common test image for tomography.

% Generate sinogram and add noise
[R, xp] = radon(x_true, theta);
% Use the 'radon' function to compute the Radon transform (sinogram) of the true image
% at the specified angles 'theta'.
%   - R: The sinogram (projection data).
%   - xp:  Radial coordinates of the projections (not used here, but returned by radon).
y = R + 0.01 * randn(size(R));
% Additive white Gaussian noise to the sinogram R to simulate noisy measurements.
%   - 0.01: Noise level (standard deviation).  Make this a parameter.
%   - randn(size(R)):  Generates noise with the same dimensions as the sinogram.

% Define A and At (the forward and adjoint operators for the Radon transform)
R_op = @(z) radon(reshape(z, imageSize, imageSize), theta);
% Define a function handle 'R_op' that takes an image vector 'z', reshapes it into an image,
% and computes its Radon transform at the angles 'theta'.  This is the forward projection.

RT_op = @(z) iradon(z, theta, 'linear', 'Ram-Lak', 1.0, imageSize);
% Define a function handle 'RT_op' that takes a sinogram 'z' and computes its inverse Radon transform
% using the 'iradon' function (which approximates the adjoint).
%   - 'linear': Linear interpolation (for the projections).  Other options exist.
%   - 'Ram-Lak': Ram-Lak filter (a standard filter in tomographic reconstruction).  Other filters exist.
%   - 1.0: Filter frequency scaling (1.0 is the default).
%   - imageSize: Size of the reconstructed image.

A = @(z) reshape(R_op(z), [], 1);
% Define the forward operator A.  It takes an image vector 'z':
%   1. Applies the Radon transform (R_op).
%   2. Reshapes the sinogram into a column vector.

At = @(z) reshape(RT_op(reshape(z, size(y))), [], 1);
% Define the adjoint operator At.  It takes a sinogram vector 'z':
%   1. Reshapes it into the sinogram size (using size(y) to get the correct dimensions).
%   2. Applies the inverse Radon transform (RT_op) which approximates the adjoint.
%   3. Reshapes the result into a column vector (the reconstructed image).

% Prepare data for algorithms
x_vec = x_true(:);       % Flatten the true image into a column vector.  Consistent with operator definitions.
y_vec = reshape(y, [], 1); % Flatten the noisy sinogram into a column vector.  Consistent with operator definitions.

% Algorithms to run
algorithms = {@PCD, @PCD_CG, @PCD_SESOP, @SESOP, @SSF, @SSF_CG, @SSF-SESOP, @FISTA};
% Cell array containing function handles to different sparse recovery algorithms.
alg_names = {'PCD', 'PCD-CG', 'PCD-SESOP', 'SESOP', 'SSF', 'SSF-CG', 'SSF-SESOP', 'FISTA'};
% Cell array of strings containing the names of the algorithms (for plotting).  Good practice.

% Pre-allocate memory
results = cell(length(algorithms), 1);       % Cell array to store reconstructed images.
history = cell(length(algorithms), 1);       % Cell array to store objective function histories.
snr_history = cell(length(algorithms), 1);   % Cell array to store SNR histories.

% Run the algorithms
for i = 1:length(algorithms) % Iterate over each algorithm.
    fprintf('Running %s...\n', alg_names{i}); % Display algorithm name for progress tracking.

    z0 = zeros(size(x_vec)); % Initialize the starting point for each algorithm.  Zero image is a common starting point.

    % Run the selected algorithm:
    %   - A, At: Forward and adjoint operators.
    %   - y_vec: Noisy sinogram vector.
    %   - lambda: Regularization parameter.
    %   - z0: Initial guess.
    %   - max_iter: Maximum iterations.
    %   - x_vec: True image vector (for SNR calculation *inside* the algorithm).
    [z, hist, snr] = algorithms{i}(A, At, y_vec, lambda, z0, max_iter, x_vec);

    results{i} = reshape(z, imageSize, imageSize); % Reshape the reconstructed vector 'z' back into a 2D image.
    history{i} = hist;                         % Store the objective function history.
    snr_history{i} = snr;                     % Store the SNR history.
end

%% Plotting and Visualization
fig1 = figure; % Create a new figure.
t = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact'); % 3x4 tiled layout for images.

% Display original image and backprojection
nexttile; imshow(x_true, []); title('Original'); axis off;  % Original phantom image.
nexttile; imshow(rescale(RT_op(y)), []); title('Backprojection'); axis off; % Simple backprojection (iradon without optimization).
                                                                         % Rescale to [0,1] for proper display.  Important baseline.

% Display reconstructed images from each algorithm
for i = 1:length(results)
    nexttile;
    imshow(results{i}, []);
    title(alg_names{i}, 'Interpreter', 'latex'); % Use LaTeX for consistent title formatting.
    axis off;
end

% Plot performance metrics (Objective Gap and SNR)
[fig2, fig3] = plot_results(history, snr_history, alg_names); % Call the plotting function.
%   - fig2:  Handle to the Objective Gap plot.
%   - fig3:  Handle to the SNR plot.

%% Save Figures
% exportgraphics(fig1, 'exp2_tomo_recon.pdf', 'ContentType', 'vector');   % Save figures as vector PDFs for high quality.
% exportgraphics(fig2, 'exp2_tomo_metrics1.pdf', 'ContentType', 'vector');
% exportgraphics(fig3, 'exp2_tomo_metrics2.pdf', 'ContentType', 'vector');