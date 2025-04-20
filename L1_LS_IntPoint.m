function [z, history] = L1_LS_IntPoint(A, At, y, lambda, z0, maxIter)
    % L1_LS_IntPoint - Solves the L1 regularized least squares problem using an interior-point method.
    %
    % Solves the optimization problem:
    %   minimize_z 0.5 * ||A*z - y||_2^2 + lambda * ||z||_1
    %
    % This function utilizes the 'l1_ls' solver, which implements an interior-point method,
    % to find the solution. It first converts the potentially function handle based
    % forward operator A into a full matrix for compatibility with 'l1_ls'.
    %
    % Inputs:
    %   A       - Function handle for the forward operator (e.g., blurring).
    %   At      - Function handle for the adjoint of the forward operator (not directly used here).
    %   y       - The observed data vector.
    %   lambda  - The regularization parameter (controls sparsity).
    %   z0      - Initial guess for the solution vector z (not directly used by l1_ls, but included for consistency).
    %   maxIter - Maximum number of iterations (not directly used by l1_ls, but included for consistency).
    %
    % Outputs:
    %   z       - The estimated solution vector.
    %   history - A vector containing the history of the primal objective function values
    %             as reported by the 'l1_ls' solver.

    % Estimate m and n (dimensions of the problem)
    m = length(y);      % Number of measurements.
    n = length(z0);     % Dimension of the unknown signal.
    imageSize = round(sqrt(n)); % Estimate the size of the underlying image if applicable.

    % Convert A from function handle to matrix only once using persistent variables for caching.
    persistent A_mat_cached At_mat_cached imageSize_cached
    if isempty(A_mat_cached) || isempty(At_mat_cached) || imageSize ~= imageSize_cached
        fprintf("Building full matrix A (%d x %d)...\n", m, n);
        A_mat = zeros(m, n);
        for i = 1:n
            e = zeros(n,1); e(i) = 1; % Create a unit basis vector.
            e_img = reshape(e, imageSize, imageSize); % Reshape it into an image (if applicable).
            A_mat(:,i) = reshape(A(e_img), [], 1); % Apply the forward operator A to the basis vector and store the result as a column of A_mat.
        end
        At_mat = A_mat'; % Compute the transpose of A_mat, which is the adjoint for a real matrix.
        A_mat_cached = A_mat; % Cache the computed A_mat for future calls.
        At_mat_cached = At_mat; % Cache the computed At_mat.
        imageSize_cached = imageSize; % Cache the image size.
    else
        A_mat = A_mat_cached; % Use the cached A_mat if it has been computed before and the image size hasn't changed.
        At_mat = At_mat_cached; % Use the cached At_mat.
    end

    % Parameters for l1_ls (interior-point solver)
    tar_gap = 1e-3;   % Target duality gap for the interior-point method.
    quiet = true;     % Suppress output from the l1_ls solver.
    eta = 1e-3;       % Parameter related to the barrier update in the interior-point method.
    pcgmaxi = 1000;   % Maximum number of iterations for the preconditioned conjugate gradient (PCG) solver used internally by l1_ls.

    % Use matrix mode for l1_ls
    [z, status, hist] = l1_ls(A_mat, y, lambda, tar_gap, quiet);
    % Call the 'l1_ls' function with the full matrix A_mat, the measurements y,
    % the regularization parameter lambda, and the specified parameters for the solver.
    % 'l1_ls' returns the solution z, the status of the solver, and a history structure 'hist'.

    % Extract objective history
    if size(hist,1) >= 2
        history = hist(2,:)';  % The 'hist' structure from 'l1_ls' contains various information.
                                % Assuming the second row contains the primal objective function values.
                                % Transpose it to get a column vector for consistency with other algorithms.
    else
        history = NaN(maxIter, 1); % If the history doesn't have at least two rows, return a vector of NaNs.
    end
end

function [A_mat, At_mat] = build_A_matrix(H, imageSize)
% build_A_matrix - Builds the matrix representation of the forward operator A and its adjoint At.
%
% This function is a utility to convert a function handle based operator (e.g., convolution)
% into a full matrix. It uses a parallel for loop to speed up the computation.
%
% Inputs:
%   H         - Function handle representing the forward operator (e.g., @(z) imfilter(z, h, 'circular')).
%   imageSize - The spatial dimension of the underlying image (assuming a square image).
%
% Outputs:
%   A_mat     - The full matrix representation of the forward operator A (n x n).
%   At_mat    - The full matrix representation of the adjoint operator At (n x n).

n = imageSize^2; % Calculate the total number of elements in the image.
A_mat = zeros(n, n); % Initialize the A matrix with zeros.
fprintf("Building full matrix A (%d x %d)...\n", n, n);
parfor i = 1:n  % Use parallel for-loop for faster matrix build
    e = zeros(n,1); e(i) = 1; % Create a unit basis vector.
    e_img = reshape(e, imageSize, imageSize); % Reshape it into an image.
    A_mat(:,i) = reshape(H(e_img), [], 1); % Apply the forward operator H to the basis vector and store the result as a column of A_mat.
end
At_mat = A_mat';  % Since H (e.g., Gaussian blur with circular boundary) is often a symmetric operator in this context, its adjoint is its transpose.
end