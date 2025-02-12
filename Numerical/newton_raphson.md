## Algorithmic Complexity & Convergence Analysis

- **Convergence Properties:**  
  - Near a simple root (where f is differentiable and f′(root) ≠ 0), the Newton–Raphson method exhibits *quadratic convergence*. This means that once the method is “close enough” to the actual root, the number of correct digits roughly doubles with each iteration.
  - If the starting guess is far from the true root or if the function behaves poorly (e.g., very flat regions or inflection points near the root), the convergence can be slower or may even diverge.

- **Operational Complexity:**  
  - **Per Iteration:** Each iteration requires a constant number of operations: one function evaluation, one derivative evaluation, one division, and one subtraction. Thus, each iteration is O(1) provided that the cost to evaluate f and f′ is constant.
  - **Overall Complexity:** The worst-case time complexity is O(maxIterations) since the algorithm performs a fixed number of operations per iteration and stops when the iteration limit is reached.

- **Worst-Case Scenarios:**  
  - **Non-Convergence:** If the initial guess is not ideal or if the function has a horizontal tangent (f′ near zero), the algorithm might fail to converge within the given iteration limit.
  - **Division by Near-Zero Derivative:** The method includes a check for small derivatives (below a predefined threshold) to avoid division by zero or extremely large updates that destabilize the iteration.

- **Conditions Affecting Performance:**  
  - The convergence is heavily dependent on the quality of the initial guess and the behavior of both the function and its derivative.  
  - Functions with multiple roots or where the derivative is zero/multiple roots may require modifications (or an alternative method) because the quadratic convergence property might be lost.

This code and its accompanying analysis provide a robust template that you can integrate into various competitive programming tasks where root-finding is necessary. Enjoy coding!
 

### Why O(log n)?

Although each iteration of the Newton–Raphson method performs only a constant number of operations (evaluating f, its derivative, and doing an update), the convergence itself is *quadratic* when close enough to a simple root. This quadratic convergence means that, at each iteration, the number of correct digits roughly doubles. In many scenarios—for instance, computing the square root of a number n—the following holds:

- Suppose your initial error is E₀ and you want to achieve an error of ε.
- Quadratic convergence implies roughly  
  E₁ ≈ C·(E₀)²,  
  E₂ ≈ C·(E₁)², and so on.
- In order to reduce the error from E₀ to within ε, you only need on the order of log(log(E₀/ε)) iterations.  
- If you relate the required precision to the magnitude of your target number (say n), the number of iterations needed to get, for example, O(log n) correct digits is itself O(log n).

Thus, when the function is well-behaved (satisfying the assumptions behind quadratic convergence) and a good starting guess is used, the iterations needed grow only logarithmically with respect to the number of correct digits (or, in some settings, the size of the number n).

> **Note:**  
> In the worst-case scenario—when the guess is far off or if the derivative is very small—the method might not converge as quickly. In such cases, the maximum iteration count (or even diverging behavior) could be encountered. In our implementation, we address these via the convergence tolerance and a maximum iteration parameter.

