from flask import Flask, render_template, request
import sympy as sp
import re
import numpy as np  # For numerical computations

app = Flask(__name__)

def preprocess_function_input(func_str):
    """
    Preprocess the function string to insert '*' where implicit multiplication is used.
    For example: '4x' -> '4*x', 'x(2+3)' -> 'x*(2+3)'.
    """
    # Insert '*' between a digit and a letter: 4x -> 4*x
    func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
    # Insert '*' between a letter and a digit: x2 -> x*2
    func_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', func_str)
    # Insert '*' between a letter and an opening parenthesis: x( -> x*(
    func_str = re.sub(r'([a-zA-Z])\(', r'\1*(', func_str)
    # Insert '*' between a closing parenthesis and a letter: )x -> )*x
    func_str = re.sub(r'\)([a-zA-Z])', r')*\1', func_str)
    return func_str

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve form data
        function_str = request.form.get("function")
        try:
            x0 = float(request.form.get("x0"))
            epsilon_str = request.form.get("epsilon")
            # If epsilon is not provided, set it to a default value (e.g., 0.001)
            epsilon = float(epsilon_str) if epsilon_str else 0.001
        except ValueError:
            return render_template("index.html", error="Please enter valid numerical values for x₀ and epsilon.")

        # Preprocess the function string to handle implicit multiplications
        function_str = preprocess_function_input(function_str)

        # Define the symbol and parse the function
        x = sp.symbols('x')
        try:
            f_expr = sp.sympify(function_str)
        except sp.SympifyError:
            return render_template("index.html", error="Invalid function expression. Please check your syntax.")

        # Create Python functions for f(x) and its derivative f'(x)
        f = sp.lambdify(x, f_expr, 'numpy')
        df_expr = sp.diff(f_expr, x)  # Derivative of the function
        df = sp.lambdify(x, df_expr, 'numpy')

        # Prepare for the Newton-Raphson iteration
        steps = []  # To store iteration steps
        current_x = x0  # Starting point
        error = float('inf')  # Initial error is set to infinity
        iteration = 0  # Start iteration counter
        max_iterations = 100  # Max iterations to avoid infinite loops

        # Newton-Raphson iteration loop
        while error > epsilon and iteration < max_iterations:
            current_f = f(current_x)  # Evaluate the function at current_x
            current_df = df(current_x)  # Evaluate the derivative at current_x

            # Check if derivative is zero to avoid division by zero
            if current_df == 0:
                return render_template("index.html", error="Derivative is zero. Try another initial guess or function.")

            # Apply the Newton-Raphson formula to calculate next_x
            next_x = current_x - current_f / current_df
            error = abs(next_x - current_x)  # Calculate the error

            # Save the current step details
            steps.append({
                "iteration": iteration,
                "x_n": current_x,
                "f_x": current_f,
                "df_x": current_df,
                "next_x": next_x,
                "error": error,
                "formula": f"xₙ₊₁ = {current_x:.6f} - ({current_f:.6f})/({current_df:.6f}) = {next_x:.6f}"
            })

            # Prepare for the next iteration
            current_x = next_x
            iteration += 1  # Increment iteration count

        # --- Generate Graph Data ---
        # Collect all iteration x-values, starting with the initial guess
        iteration_points = [x0] + [step['next_x'] for step in steps]
        # Compute corresponding y values for the iteration points
        iteration_y = [float(f(x_val)) for x_val in iteration_points]

        # Determine the plotting range based on the iteration points
        min_x = min(iteration_points)
        max_x = max(iteration_points)
        margin = (max_x - min_x) * 0.5 if max_x != min_x else 1
        plot_min = min_x - margin
        plot_max = max_x + margin

        # Generate an array of x values for plotting the function (200 points)
        x_vals = np.linspace(plot_min, plot_max, 200)
        y_vals = f(x_vals)

        # Convert function to LaTeX format for rendering
        function_latex = sp.latex(f_expr)

        # Return the results to the template
        return render_template("result.html", 
                               steps=steps, 
                               solution=current_x, 
                               iterations=iteration,
                               x_vals=list(x_vals), 
                               y_vals=list(y_vals),
                               iteration_points=iteration_points, 
                               iteration_y=iteration_y,
                               function_str=function_latex)

    # If it's a GET request, render the input form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
