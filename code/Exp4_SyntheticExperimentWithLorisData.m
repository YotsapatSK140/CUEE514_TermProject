clear; clc; close all;

% Experiment 4: Synthetic LORIS Data Reproduction (from Zibulevsky & Elad)
% This script reproduces the compressed sensing experiment with synthetic data
% as presented in Zibulevsky & Elad, using the LORIS dataset generation approach.
% The experiment is run for two different values of the regularization parameter lambda
% to observe its effect on the reconstruction performance.

%% lambda = 1e-3
% Parameters (First experiment: lambda = 1e-3)
n = 8192;         % Signal length: Length of the sparse signal to be recovered (number of unknowns).
m = 1848;         % Number of measurements: Number of linear measurements obtained (should be less than n for CS).
lambda = 1e-3;    % Regularization parameter: Controls the sparsity-promoting effect.  Larger lambda = sparser solution.
sigma = 1e-4;     % Noise level: Standard deviation of the additive Gaussian noise.  Affects reconstruction quality.
sparsity = 0.05;  % Sparsity level: Fraction of non-zero entries in the true signal (5% in this case).
max_iter = 600;   % Maximum number of iterations for the optimization algorithms.  Convergence parameter.

% Generate ill-conditioned matrix H (K4 matrix) - This is the sensing matrix
rng(0); % Seed the random number generator for reproducibility.  Important for getting the same results on multiple runs.

[U, ~] = qr(randn(m));       % Generate a random m x m matrix and obtain its orthogonal factor U from QR decomposition.
                            % U is an orthogonal matrix.
[V, ~] = qr(randn(n));       % Generate a random n x n matrix and obtain its orthogonal factor V from QR decomposition.
                            % V is an orthogonal matrix.
s = logspace(0, -6, m);      % Generate m logarithmically spaced singular values, decreasing from 1 to 1e-6.
                            % These singular values control the condition number of the matrix H.
S = diag(s);                % Create a diagonal matrix S with the singular values on the diagonal.
H = U * S * V(:,1:m)';       % Construct the ill-conditioned matrix H (m x n) by multiplying U, S, and the transpose of the first m columns of V.
                            %  - V(:,1:m)' selects the first m columns of V and transposes them.
                            %  - This construction creates a matrix H with controlled singular values, making the inverse
                            %    problem challenging (ill-posed) and representative of compressed sensing scenarios.

% Generate sparse ground truth z_true (the signal we want to recover)
z_true = zeros(n,1);                             % Initialize a zero vector of length n (the signal vector).
idx = randperm(n, round(sparsity * n));         % Randomly select indices for the nonzero entries based on the sparsity level.
                                                %  - round(sparsity * n):  Calculates the number of non-zero elements.
                                                %  - randperm(n, ...):    Selects that many unique random indices from 1 to n.
z_true(idx) = 10 * randn(length(idx), 1);       % Assign random values (from a normal distribution with mean 0 and std dev 10)
                                                %  to the selected nonzero indices.  The '10*' scales the random values.

% Generate noisy measurements (the data we actually observe)
y = H * z_true + sigma * randn(m,1); % Generate the measurements y by:
                                    %  - Multiplying H (the sensing matrix) with z_true (the sparse signal).
                                    %  - Adding Gaussian noise with standard deviation 'sigma'.

% Compute f(z_true) as baseline objective (for comparison)
f_star = 0.5 * norm(H*z_true - y)^2 + lambda * norm(z_true,1);
% Calculate the objective function value at the true signal z_true.  This is the value of the function
% we are trying to minimize.  It serves as a lower bound (or a target) for the optimization algorithms.
% The objective function is composed of two terms:
%   - 0.5 * norm(H*z_true - y)^2:  The data fidelity term (least-squares error).  Measures how well the solution fits the measurements.
%   - lambda * norm(z_true, 1): The regularization term (L1 norm of z_true).  Promotes sparsity in the solution.

% Define A and At (the forward and adjoint operators)
A = @(z) H * z;   % The forward operator A is simply matrix multiplication by H (H * z).
At = @(z) H' * z;  % The adjoint (transpose) operator At is matrix multiplication by the conjugate transpose of H (H' * z).
                    %  For real-valued H, H' is the same as the transpose of H.

% Algorithm list
algorithms = {@PCD, @PCD_CG, @PCD_SESOP, @SESOP, @SSF, @SSF-CG, @SSF-SESOP, @FISTA};
% Cell array of function handles to the different sparse recovery algorithms being tested.
alg_names = {'PCD', 'PCD-CG', 'PCD-SESOP', 'SESOP', 'SSF', 'SSF-CG', 'SSF-SESOP', 'FISTA'};
% Cell array of corresponding algorithm names for plotting and display.

% Pre-allocate memory to store results
results = cell(length(algorithms),1);     % Cell array to store the recovered signals (z) for each algorithm.
history = cell(length(algorithms),1);     % Cell array to store the history of objective function values for each algorithm.
                                        %  Will store  f(z_k) - f_star  at each iteration k.
snr_history = cell(length(algorithms),1); % Cell array to store the history of SNR values for each algorithm.

% Run algorithms (First experiment: lambda = 1e-3)
for i = 1:length(algorithms) % Loop through each algorithm in the 'algorithms' cell array.
    fprintf('Running %s...\n', alg_names{i}); % Display the name of the algorithm being executed.

    z0 = zeros(n, 1); % Initialize the starting point for each algorithm as a zero vector of length n.
                       %  A common starting point, but other initializations could be tried.

    try
        % Run the i-th algorithm:
        %   - A, At:  Forward and adjoint operators.
        %   - y:      Noisy measurements.
        %   - lambda: Regularization parameter.
        %   - z0:     Initial guess.
        %   - max_iter: Maximum number of iterations.
        %   - z_true: The true signal (for calculating SNR *inside* the algorithm).
        [z, hist, snr] = algorithms{i}(A, At, y, lambda, z0, max_iter, z_true);
    catch
        fprintf('Error running %s\n', alg_names{i}); % Print an error message if the algorithm fails.
        z = z0;          % If an error occurs, return the initial guess.
        hist = NaN(max_iter,1); % Return a vector of NaNs for the objective history.
        snr = NaN(max_iter,1);    % Return a vector of NaNs for the SNR history.
        %  This prevents the script from crashing and allows it to continue with the other algorithms.
    end

    results{i} = z;               % Store the recovered signal (solution) z.
    history{i} = hist - f_star;   % Store the history of objective function values, relative to the optimal value f_star.
                                    %  This shows how close the algorithm gets to the best possible solution.
    snr_history{i} = snr;         % Store the history of SNR values, measuring the reconstruction quality.
end

% Plot results for lambda = 1e-3
[fig1, fig2] = plot_results(history, snr_history, alg_names); % Call the plotting function to visualize the results.
                                                            %  - fig1: Handle to the Objective Gap plot.
                                                            %  - fig2: Handle to the SNR plot.

%% lambda = 1e-6
% Repeat the experiment with a different regularization parameter lambda = 1e-6.
% This allows us to see how the choice of lambda affects the reconstruction.

% Parameters (Second experiment: lambda = 1e-6) - Re-define parameters for the second run.
n = 8192;
m = 1848;
lambda = 1e-6;    % Regularization parameter (smaller than before:  more emphasis on data fidelity, less on sparsity)
sigma = 1e-4;     % Noise level (same as before)
sparsity = 0.05;
max_iter = 3000;  % Maximum number of iterations (increased for smaller lambda, as convergence may be slower).

% Generate ill-conditioned matrix H (K4 matrix) - same as before (re-calculated for consistency, though technically not needed)
rng(0);
[U, ~] = qr(randn(m));
[V, ~] = qr(randn(n));
s = logspace(0, -6, m);
S = diag(s);
H = U * S * V(:,1:m)';

% Generate sparse ground truth z_true - same as before
z_true = zeros(n,1);
idx = randperm(n, round(sparsity * n));
z_true(idx) = 10 * randn(length(idx), 1);

% Generate noisy measurements - same as before
y = H * z_true + sigma * randn(m,1);

% Compute f(z_true) as baseline objective - same as before
f_star = 0.5 * norm(H*z_true - y)^2 + lambda * norm(z_true,1);

% Define A and At - same as before
A = @(z) H * z;
At = @(z) H' * z;

% Algorithm list - same as before
algorithms = {@PCD, @PCD_CG, @PCD_SESOP, @SESOP, @SSF, @SSF-CG, @SSF-SESOP, @FISTA};
alg_names = {'PCD', 'PCD-CG', 'PCD-SESOP', 'SESOP', 'SSF', 'SSF-CG', 'SSF-SESOP', 'FISTA'};

% Re-initialize results arrays for the second experiment (different lambda)
results = cell(length(algorithms),1);
history = cell(length(algorithms),1);
snr_history = cell(length(algorithms),1);

% Run algorithms for lambda = 1e-6 (Second experiment)
for i = 1:length(algorithms)
    fprintf('Running %s (lambda=1e-6)...\n', alg_names{i}); % Indicate the lambda value in the output.
    z0 = zeros(n, 1);
    try
        [z, hist, snr] = algorithms{i}(A, At, y, lambda, z0, max_iter, z_true);
    catch
        fprintf('Error running %s (lambda=1e-6)\n', alg_names{i});
        z = z0; hist = NaN(max_iter,1); snr = NaN(max_iter,1);
    end
    results{i} = z;
    history{i} = hist - f_star;
    snr_history{i} = snr;
end

% Plot results for lambda = 1e-6
[fig3, fig4] = plot_results(history, snr_history, alg_names); % Use different figure handles (fig3, fig4) to keep the plots separate.

%% Save Figures
% exportgraphics(fig1, 'exp4_loris_metrics1_1.pdf', 'ContentType', 'vector');
% exportgraphics(fig2, 'exp4_loris_metrics1_2.pdf', 'ContentType', 'vector');
% exportgraphics(fig3, 'exp4_loris_metrics2_1.pdf', 'ContentType', 'vector');
% exportgraphics(fig4, 'exp4_loris_metrics2_2.pdf', 'ContentType', 'vector');