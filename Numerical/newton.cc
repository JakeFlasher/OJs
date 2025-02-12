#include <iostream>
#include <cmath>
#include <limits>

// Structure to hold the result of the Newton-Raphson method.
template <typename T>
struct NewtonRaphsonResult {
    bool converged;   // Indicates whether the method converged.
    T root;           // The computed root (if converged, or the last estimate).
    int iterations;   // The number of iterations performed.
};

// Generic Newton-Raphson function template.
// The template accepts callable objects for f and its derivative f'.
// Parameters:
//   f              : The function for which the root is to be found.
//   df             : The derivative of the function f.
//   initial        : The starting guess for the root.
//   epsilon        : The tolerance for convergence, defaults to 1e-6.
//   maxIterations  : Maximum allowed iterations to avoid infinite loops.
template <typename Function, typename Derivative, typename T>
NewtonRaphsonResult<T> newtonRaphson(Function f, Derivative df, T initial,
                                      T epsilon = static_cast<T>(1e-6),
                                      int maxIterations = 1000) {
    NewtonRaphsonResult<T> result;
    result.root = initial;
    
    // Iterate up to the maximum allowed iterations.
    for (int i = 0; i < maxIterations; ++i) {
        T f_val = f(result.root);   // Evaluate function at current estimate.
        T df_val = df(result.root); // Evaluate derivative at current estimate.
        
        // Define a threshold to check for near-zero derivative
        // to avoid division by zero (or a very small number)
        const T derivativeThreshold = static_cast<T>(1e-12);
        if (std::abs(df_val) < derivativeThreshold) {
            std::cerr << "Warning: Derivative near zero at iteration " << i
                      << ". Current estimate: " << result.root << "\n";
            result.converged = false;
            result.iterations = i;
            return result;
        }
        
        T delta = f_val / df_val;  // Newton-Raphson update step.
        result.root -= delta;      // Update the estimate.
        
        // Check if the update is small enough to consider it converged.
        if (std::abs(delta) < epsilon) {
            result.converged = true;
            result.iterations = i + 1;
            return result;
        }
    }
    
    // If we reach here, we did not converge within maxIterations.
    result.converged = false;
    result.iterations = maxIterations;
    return result;
}

int main() {
    // ===============================================================
    // Example 1: Compute the square root of a number.
    // We solve f(x) = x^2 - target = 0, i.e., find x such that x^2 = target.
    double target = 25.0;
    
    // Define f(x) and its derivative f'(x) using lambda expressions.
    auto f_square = [target](double x) -> double {
        return x * x - target;
    };
    auto df_square = [](double x) -> double {
        return 2 * x;
    };
    
    // A sensible initial guess can be target/2.
    double initial_guess_square = target / 2;
    auto result_square = newtonRaphson<decltype(f_square), decltype(df_square), double>(
        f_square, df_square, initial_guess_square
    );
    
    if (result_square.converged)
        std::cout << "Square root of " << target << " is approximately "
                  << result_square.root << " (converged in "
                  << result_square.iterations << " iterations).\n";
    else
        std::cout << "Square root computation did not converge for target "
                  << target << ".\n";
    
    // ===============================================================
    // Example 2: Compute the cube root of a number.
    // We solve f(x) = x^3 - target = 0, i.e., find x such that x^3 = target.
    double target_cube = 27.0;
    
    // Define f(x) for cube root and its derivative using lambdas.
    auto f_cube = [target_cube](double x) -> double {
        return x * x * x - target_cube;
    };
    auto df_cube = [](double x) -> double {
        return 3 * x * x;
    };
    
    // A simple initial guess can be target_cube / 3.
    double initial_guess_cube = target_cube / 3;
    auto result_cube = newtonRaphson<decltype(f_cube), decltype(df_cube), double>(
        f_cube, df_cube, initial_guess_cube
    );
    
    if (result_cube.converged)
        std::cout << "Cube root of " << target_cube << " is approximately "
                  << result_cube.root << " (converged in "
                  << result_cube.iterations << " iterations).\n";
    else
        std::cout << "Cube root computation did not converge for target "
                  << target_cube << ".\n";
    
    // ===============================================================
    // Example 3: Solve a common cubic equation in competitive programming.
    // Equation: f(x) = x^3 - 2*x - 5 = 0, which has one real root.
    // Its derivative is f'(x) = 3*x^2 - 2.
    auto f_cubic = [](double x) -> double {
        return x * x * x - 2 * x - 5;
    };
    auto df_cubic = [](double x) -> double {
        return 3 * x * x - 2;
    };
    
    // An initial guess is chosen based on the expected location of the root.
    double initial_guess_cubic = 2.0;
    auto result_cubic = newtonRaphson<decltype(f_cubic), decltype(df_cubic), double>(
        f_cubic, df_cubic, initial_guess_cubic
    );
    
    if (result_cubic.converged)
        std::cout << "Root of cubic equation x^3 - 2*x - 5 = 0 is approximately "
                  << result_cubic.root << " (converged in "
                  << result_cubic.iterations << " iterations).\n";
    else
        std::cout << "Cubic equation did not converge.\n";
    
    return 0;
}
