function [z, history, snr_history] = SESOP(A, At, y, lambda, z0, maxIter, x_true)
    % SESOP - Subspace Expansion by Shifted and Orthogonalized Power method for sparse recovery.
    %
    % Solves the optimization problem:
    %   minimize_z 0.5 * ||A*z - y||_2^2 + lambda * ||z||_1
    %
    % This algorithm uses a subspace acceleration technique (SESOP) to speed up convergence.
    % It builds a subspace of descent directions based on the gradient and performs
    % optimization within this subspace. A soft-thresholding step is applied to promote sparsity.
    %
    % Inputs:
    %   A       - Function handle for the forward operator (e.g., matrix multiplication).
    %   At      - Function handle for the adjoint (transpose) of the forward operator.
    %   y       - The observed data vector.
    %   lambda  - The regularization parameter (controls sparsity).
    %   z0      - Initial guess for the solution vector z.
    %   maxIter - Maximum number of iterations to run the algorithm.
    %   x_true  - The true underlying signal (optional, for SNR calculation).
    %
    % Outputs:
    %   z           - The estimated solution vector.
    %   history     - A vector containing the objective function value at each iteration.
    %   snr_history - A vector containing the Signal-to-Noise Ratio (SNR) at each iteration (if x_true is provided).

    m = 7; % SESOP subspace size - the maximum number of vectors to maintain in the subspace D.
    z = z0;        % Initialize the solution with the initial guess.
    D = [];        % Initialize the subspace matrix D, which will store descent directions.
    history = zeros(maxIter, 1);     % Initialize an array to store the objective function values.
    snr_history = zeros(maxIter, 1); % Initialize an array to store the SNR values.

    for k = 1:maxIter % Main iteration loop.
        % Calculate the gradient of the smooth part of the objective function: A'(A*z - y).
        grad = At(A(z) - y);

        % Add the negative gradient as a new descent direction to the subspace.
        d = -grad;
        if isempty(D) % If the subspace is empty, initialize it with the gradient.
            D = d;
        else % Otherwise, append the new direction, keeping at most 'm' recent directions.
            D = [d, D(:, 1:min(end, m-1))];
        end

        % Subspace optimization using the current subspace D.
        G = D' * D; % Gram matrix of the subspace vectors.
        g = D' * grad; % Projection of the gradient onto the subspace.

        % Perform a safety check on the condition number of G.
        % If G is ill-conditioned, the subspace optimization step might be unstable.
        if cond(G) < 1e12  % Check if the condition number is below a threshold.
            alpha = -pinv(G) * g; % Calculate the optimal coefficients for the subspace update using the pseudoinverse.
            z = z + D * alpha;   % Update the solution by moving along the linear combination of the subspace vectors.
        else
            z = z - grad;  % Fallback to a simple gradient descent step if the subspace is ill-conditioned.
        end

        % Apply the soft-thresholding operator to enforce sparsity.
        z = soft_threshold(z, lambda);

        % Track the progress of the algorithm.
        history(k) = 0.5 * norm(A(z) - y)^2 + lambda * norm(z, 1);

        % Calculate and store the SNR if the true signal is provided.
        if nargin > 6 && ~isempty(x_true)
            snr_history(k) = 20 * log10(norm(x_true) / norm(x_true - z));
        end

        % Periodically reset the subspace to prevent it from becoming too large or poorly conditioned.
        if mod(k, 20) == 0
            D = [];
        end
    end
end

function z = soft_threshold(x, t)
    % soft_threshold - Applies the soft-thresholding (shrinkage) operator.
    %
    % Input:
    %   x - Input vector.
    %   t - Threshold vector (same size as x).
    %
    % Output:
    %   z - Soft-thresholded vector.
    %
    % For each element x_i and threshold t_i:
    %   z_i = sign(x_i) * max(abs(x_i) - t_i, 0)

    z = sign(x) .* max(abs(x) - t, 0);
end

function L = max_singular_value(A, At, v, iterations)
    % max_singular_value - Estimates the largest singular value (spectral norm) of the linear operator A.
    %
    % Input:
    %   A          - Function handle for the forward operator.
    %   At         - Function handle for the adjoint of the forward operator.
    %   v          - Initial vector for the power iteration.
    %   iterations - Number of power iterations to perform.
    %
    % Output:
    %   L          - Estimate of the largest singular value.
    %
    % This function uses the power iteration method to approximate the spectral norm.
    for i = 1:iterations
        v = A(v);       % Apply the forward operator.
        v = v/norm(v);  % Normalize the vector.
        v = At(v);      % Apply the adjoint operator.
    end
    L = norm(v); % The norm of the resulting vector is an estimate of the largest singular value.
end

function W = estimate_diag_AtA(A, At, n, samples)
    % estimate_diag_AtA - Estimates the diagonal elements of the matrix A'*A stochastically.
    %
    % Input:
    %   A       - Function handle for the forward operator.
    %   At      - Function handle for the adjoint of the forward operator.
    %   n       - Dimension of the unknown signal.
    %   samples - Number of random vectors to use for estimation.
    %
    % Output:
    %   W       - A vector containing the estimate of the diagonal of A'*A.
    %
    % This function uses random +/-1 vectors to estimate the diagonal.
    W = zeros(n,1); % Initialize the diagonal estimate.
    for i = 1:samples
        v = sign(randn(n,1)); % Generate a random vector with elements +1 or -1.
        W = W + (At(A(v))).^2; % Compute A'(A(v)), square the elements, and accumulate.
    end
    W = 1./(W/samples + eps); % Average the results and take the inverse (plus a small epsilon for stability).
                               % The inverse diagonal is often used as a preconditioner in coordinate descent methods.
end

function alpha = linesearch(A, y, z, d, lambda)
    % linesearch - Performs a backtracking line search to find a suitable step size.
    %
    % Input:
    %   A      - Function handle for the forward operator.
    %   y      - The observed data vector.
    %   z      - The current solution vector.
    %   d      - The search direction (v - z in the PCD algorithm).
    %   lambda - The regularization parameter.
    %
    % Output:
    %   alpha  - The chosen step size.
    %
    % This function uses a backtracking approach with the Armijo condition
    % to ensure sufficient decrease in the objective function.

    alpha = 1;     % Start with a step size of 1.
    rho = 0.5;     % Reduction factor for alpha.
    c = 1e-4;      % Sufficient decrease parameter (small positive constant).
    f0 = 0.5*norm(A(z)-y)^2 + lambda*norm(z,1); % Objective function value at the current z.
    g0 = A(z)-y;   % Residual vector.

    while true % Keep reducing alpha until the Armijo condition is met.
        z_new = z + alpha*d; % Calculate the new solution with the current step size.
        f_new = 0.5*norm(A(z_new)-y)^2 + lambda*norm(z_new,1); % Objective function value at the new z.

        % Check the Armijo condition: f(z + alpha*d) <= f(z) + c * alpha * <nabla f(z), d>
        % Here, we approximate <nabla f(z), d> with (A(z)-y)' * A(d).
        if f_new <= f0 + c*alpha*(g0'*A(d))
            break; % Exit the loop if the condition is satisfied.
        end
        alpha = rho*alpha; % Reduce the step size.
    end
end