clear; clc; close all;

% Experiment 3: Compressed Sensing with Synthetic Sparse Image
% This script performs compressed sensing reconstruction of a synthetic sparse image.
% It generates a sparse image, creates a partial Fourier sensing matrix,
% acquires noisy measurements, and reconstructs the image using several algorithms.
% Finally, it plots the original and reconstructed images, as well as the
% objective function history and SNR history for each algorithm.

% Parameters
imageSize = 128;        % Size of the square image (128x128 pixels)
n = imageSize^2;        % Total number of pixels in the image
m = floor(0.4 * n);     % Number of measurements (40% of the total pixels) - underdetermined
lambda = 0.001;         % Regularization parameter (controls sparsity)
max_iter = 1000;        % Maximum number of iterations for the reconstruction algorithms
sparsity_level = 0.05;  % Sparsity level of the true image (5% of pixels are non-zero)

% Create a synthetic sparse image in 2D
x_img = zeros(imageSize);                 % Initialize a black image
num_nonzero = floor(sparsity_level * n);  % Calculate the number of non-zero pixels
rand_idx = randsample(n, num_nonzero);    % Randomly select indices for non-zero pixels.  randsample avoids duplicate indices.
x_img(rand_idx) = randn(num_nonzero, 1);  % Assign random values (from a normal distribution) to the selected pixels
x_true = x_img(:);                        % Convert the 2D image to a 1D vector (ground truth)

% Define partial Fourier sensing matrix (Compressed Sensing measurement matrix)
perm = randperm(n);                       % Generate a random permutation of pixel indices
sample_idx = sort(perm(1:m));             % Select the first 'm' indices from the permutation (sorted for efficiency) - these are the indices where we sample in the Fourier domain
F = @(z) fft2(reshape(z, imageSize, imageSize)) / sqrt(n); % Define the 2D FFT operator, normalized
Fi = @(z) real(ifft2(z) * sqrt(n));       % Define the 2D inverse FFT operator (real part), normalized.  Important to take the real part.

% Measurement operator (A)
% This function applies the Fourier transform and then samples it at the indices given by sample_idx
A = @(z) fft_sample(z, imageSize, sample_idx);
% Adjoint operator (At)
% This function takes the sampled Fourier data, pads it with zeros, and then applies the inverse Fourier transform
At = @(z) ifft_sample_adjoint(z, imageSize, sample_idx, n);

% Generate noisy measurements
y_clean = A(x_true);                 % Compute the clean (noiseless) measurements
noise_level = 0.01;                  % Set the noise level (standard deviation of the Gaussian noise)
y = y_clean + noise_level * randn(size(y_clean)); % Add Gaussian noise to the measurements.  randn generates normally distributed random numbers.

% Algorithms to be used for reconstruction
algorithms = {@PCD, @PCD_CG, @PCD_SESOP, @SESOP, @SSF, @SSF_CG, @SSF_SESOP, @FISTA}; % Cell array of function handles to different algorithms.
alg_names = {'PCD', 'PCD-CG', 'PCD-SESOP', 'SESOP', 'SSF', 'SSF-CG', 'SSF-SESOP', 'FISTA'}; % Cell array of corresponding algorithm names (for plotting and display)

% Preallocate memory to store results
results = cell(length(algorithms), 1);       % Cell array to store the reconstructed images
history = cell(length(algorithms), 1);       % Cell array to store the objective function history (for each algorithm)
snr_history = cell(length(algorithms), 1);   % Cell array to store the SNR history (for each algorithm)

% Run all algorithms
for i = 1:length(algorithms)
    fprintf('Running %s...\n', alg_names{i}); % Display the name of the algorithm being run
    z0 = zeros(n, 1);                         % Initialize the starting point for the algorithm (all zeros)
    % Run the selected algorithm:
    %  - A:  The forward operator
    %  - At: The adjoint operator
    %  - y:  The noisy measurements
    %  - lambda: The regularization parameter
    %  - z0: The initial guess
    %  - max_iter: The maximum number of iterations
    %  - x_true: The true image (used to calculate SNR)
    [z, hist, snr] = algorithms{i}(A, At, y, lambda, z0, max_iter, x_true);
    results{i} = reshape(z, imageSize, imageSize); % Reshape the reconstructed vector 'z' into a 2D image
    history{i} = hist;                         % Store the objective function history
    snr_history{i} = snr;                     % Store the SNR history
end

%% Plotting and Visualization
% Plot: Original and Reconstructed Sparse Image
fig1 = figure;
t = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact'); % Create a tiled layout for displaying images
nexttile; imshow(reshape(x_true, imageSize, imageSize)); title('Original'); axis off; % Display the original image
for i = 1:length(results)
    nexttile;
    imshow(results{i});             % Display the reconstructed image from each algorithm
    title(alg_names{i}, 'Interpreter', 'latex'); % Display the algorithm name as the title, using LaTeX for formatting
    axis off;                           % Turn off axis labels for cleaner visualization
end

% Plot metrics (Objective Gap and SNR)
[fig2, fig3] = plot_results(history, snr_history, alg_names); % Call the plot_results function (defined below) to generate the metric plots

% Helper functions for A and At

% fft_sample:  Applies the Fourier transform and samples the result
function y = fft_sample(x, imageSize, sample_idx)
    F_x = fft2(reshape(x, imageSize, imageSize)) / sqrt(imageSize^2); % Compute the 2D FFT of the input image x, normalized
    y = reshape(F_x, [], 1);             % Reshape the FFT result into a 1D vector
    y = y(sample_idx);                   % Sample the elements of the vector at the indices given by sample_idx
end

% ifft_sample_adjoint:  Zero-pads the sampled Fourier data and applies the inverse Fourier transform
function x = ifft_sample_adjoint(y, imageSize, sample_idx, n)
    y_full = zeros(n, 1);               % Create a vector of zeros of length 'n' (the original number of pixels)
    y_full(sample_idx) = y;             % Place the sampled Fourier data 'y' into the correct locations in the full vector
    Y = reshape(y_full, imageSize, imageSize); % Reshape the full Fourier vector back into a 2D image
    x = real(ifft2(Y) * sqrt(imageSize^2));   % Compute the 2D inverse FFT, normalize, and take the real part
    x = x(:);                           % Convert the result back into a 1D vector
end

%% Save figures
% exportgraphics(fig1, 'exp3_cs_recon.pdf', 'ContentType', 'vector');   % Save the image reconstruction figure as a PDF (vector format for high quality)
% exportgraphics(fig2, 'exp3_cs_metrics1.pdf', 'ContentType', 'vector'); % Save the objective gap plot as a PDF
% exportgraphics(fig3, 'exp3_cs_metrics2.pdf', 'ContentType', 'vector'); % Save the SNR plot as a PDF