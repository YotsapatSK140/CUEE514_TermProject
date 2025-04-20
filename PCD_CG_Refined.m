function [z, history, snr_history] = PCD_CG_Refined(A, At, y, lambda, z0, maxIter, s, x_true)
    % PCD_CG_Refined - Proximal Coordinate Descent with Conjugate Gradient and refined shrinkage for sparse recovery.
    %
    % Solves the optimization problem:
    %   minimize_z 0.5 * ||A*z - y||_2^2 + lambda * sum(s * log(1 + abs(z) / s))
    %
    % This algorithm combines Conjugate Gradient (CG) to minimize the smooth part of the objective
    % with a non-convex, concave shrinkage function that can provide better sparsity than the L1 norm,
    % especially for recovering signals with varying magnitudes.
    %
    % Inputs:
    %   A       - Function handle for the forward operator (e.g., matrix multiplication).
    %   At      - Function handle for the adjoint (transpose) of the forward operator.
    %   y       - The observed data vector.
    %   lambda  - The regularization parameter (controls sparsity).
    %   z0      - Initial guess for the solution vector z.
    %   maxIter - Maximum number of iterations to run the algorithm.
    %   s       - Parameter controlling the shape of the concave penalty.
    %   x_true  - The true underlying signal (optional, for SNR calculation).
    %
    % Outputs:
    %   z           - The estimated solution vector.
    %   history     - A vector containing the objective function value at each iteration.
    %   snr_history - A vector containing the Signal-to-Noise Ratio (SNR) at each iteration (if x_true is provided).

    z = z0;        % Initialize the solution with the initial guess.
    r = At(y - A(z)); % Initial residual (negative gradient of the smooth part).
    p = r;           % Initial conjugate direction is the residual.
    history = zeros(maxIter, 1);     % Initialize an array to store the objective function values.
    snr_history = zeros(maxIter, 1); % Initialize an array to store the SNR values.

    for k = 1:maxIter % Main iteration loop.
        % Conjugate Gradient (CG) step to minimize the smooth part: 0.5 * ||A*z - y||^2.
        Ap = At(A(p)); % Apply A and then A' to the conjugate direction.
        alpha = (r' * r) / (p' * Ap); % Calculate the step size (alpha) for CG.
        z = z + alpha * p;       % Update the solution z along the conjugate direction.
        r_new = r - alpha * Ap;  % Update the residual.
        beta = (r_new' * r_new) / (r' * r); % Calculate the coefficient (beta) for the next conjugate direction.
        p = r_new + beta * p;    % Update the conjugate direction.
        r = r_new;               % Update the residual for the next iteration.

        % Concave shrinkage step: Apply a non-convex penalty to promote sparsity.
        % The penalty used here is a form of the Log-Sum Penalty (LSP).
        % The proximal operator for this penalty is not as simple as soft-thresholding,
        % and the update here is a form of iterative soft-thresholding adapted for this penalty.
        z = sign(z) .* max(abs(z) - lambda * s ./ (s + abs(z)), 0);

        % Calculate and store the objective function value at the current iteration.
        % The objective function now includes the concave penalty term.
        history(k) = 0.5 * norm(A(z) - y)^2 + lambda * sum(s .* log(1 + abs(z) / s));

        % Calculate and store the SNR if the true signal is provided.
        if nargin > 7 && ~isempty(x_true)
            snr_history(k) = 20 * log10(norm(x_true(:)) / norm(x_true(:) - z));
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