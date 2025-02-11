from flask import Flask, render_template, request
import sympy as sp
import numpy as np
from decimal import Decimal, InvalidOperation

# Import additional parsing tools
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

app = Flask(__name__)

# Allowed mathematical functions and constants
allowed_locals = {
    'ln': sp.log,
    'log': sp.log,
    'log10': lambda arg: sp.log(arg, 10),
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'sec': lambda x: 1/sp.cos(x),
    'cosec': lambda x: 1/sp.sin(x),
    'cot': lambda x: 1/sp.tan(x),
    'sqrt': sp.sqrt,
    'exp': sp.exp,
    'abs': sp.Abs,
    'pi': sp.pi,
    'E': sp.E
}

transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def newton_raphson(f, df, x0, epsilon=0.001, max_iterations=100):
    """Performs the Newton-Raphson method to find a root of f(x)."""
    steps = []
    current_x = float(x0)  # Convert Decimal to float for compatibility

    for iteration in range(max_iterations):
        current_f = float(f(current_x))  # Ensure float conversion
        current_df = float(df(current_x))

        if abs(current_df) < 1e-8:  # Check for division by near-zero
            return None, f"Derivative too small at x = {current_x:.6f}. The method may not converge."

        next_x = current_x - (current_f / current_df)
        error_val = abs(next_x - current_x)

        steps.append({
            "iteration": iteration,
            "x_n": current_x,
            "f_x": current_f,
            "df_x": current_df,
            "next_x": next_x,
            "error": error_val,
            "formula": f"xₙ₊₁ = {current_x:.6f} - ({current_f:.6f})/({current_df:.6f}) = {next_x:.6f}"
        })

        if abs(current_f) < epsilon:  # Stop if function value is close to zero
            return steps, None

        current_x = next_x

    return None, "Newton-Raphson method did not converge after 100 iterations. Try another initial guess."


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        function_str = request.form.get("function", "").strip()

        function_str = function_str.replace("−", "-")  # Fix Unicode minus sign
        function_str = function_str.replace("X", "x")  # Convert 'X' to 'x' for consistency
        function_str = function_str.lower()  # Convert the entire input to lowercase

        if not function_str:
            return render_template("index.html", error="Function input cannot be empty.")

        try:
            x = sp.symbols('x')
            f_expr = parse_expr(function_str, local_dict=allowed_locals, transformations=transformations)
        except (SyntaxError, ValueError, TypeError) as e:
            return render_template("index.html", error=f"Invalid function expression: Check syntax or use supported functions like sin(x), cos(x), etc. Details: {str(e)}")

        f = sp.lambdify(x, f_expr, 'numpy')
        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, 'numpy')

        try:
            x0 = float(request.form.get("x0"))  # Convert input directly to float
            epsilon = float(request.form.get("epsilon")) if request.form.get("epsilon") else 0.001

            if epsilon < 1e-10:
                return render_template("index.html", error="Epsilon is too small. Use a value ≥ 1e-10.")

        except ValueError:
            return render_template("index.html", error="Invalid numerical input for x₀ or epsilon.")

        steps, error = newton_raphson(f, df, x0, epsilon)

        if error:
            return render_template("index.html", error=error)

        solution = steps[-1]["next_x"]
        iteration_points = [x0] + [step["next_x"] for step in steps]
        iteration_y = [f(val) for val in iteration_points]

        min_x = min(iteration_points)
        max_x = max(iteration_points)
        margin = (max_x - min_x) * 0.5 if max_x != min_x else 1
        plot_min = min_x - margin
        plot_max = max_x + margin

        x_vals = np.linspace(plot_min, plot_max, 200)
        y_vals = np.array([f(x) if np.isfinite(f(x)) else np.nan for x in x_vals])  # Prevent overflow

        function_latex = sp.latex(f_expr)

        return render_template(
            "result.html",
            steps=steps,
            solution=solution,
            iterations=len(steps),
            x_vals=list(x_vals),
            y_vals=list(y_vals),
            iteration_points=iteration_points,
            iteration_y=iteration_y,
            function_str=function_latex
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
