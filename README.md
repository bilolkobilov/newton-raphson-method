# Newton-Raphson Method Web App

This is a web-based calculator for solving equations using the **Newton-Raphson method**. Users can input a function \( f(x) \), an initial guess \( x_0 \), and an epsilon value to find the root of the function step by step.

## Features
- Supports mathematical functions like `sin`, `cos`, `tan`, `ln`, `log`, `exp`, `sqrt`, and more.
- Accepts both lowercase and uppercase `X` in function input.
- Displays step-by-step calculations with the Newton-Raphson formula.
- Plots the function graph with iteration points.
- Includes JavaScript animations for better visualization.

## Technologies Used
- **Backend:** Python (Flask), SymPy for symbolic math.
- **Frontend:** HTML, CSS, JavaScript.
- **Graph Plotting:** Matplotlib, NumPy.
- **Deployment:** Vercel (or any other hosting platform).

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bilolkobilov/newton-raphson-method.git
   cd newton-raphson-method
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```
   The app will be accessible at `http://127.0.0.1:5000/`.

## Usage
1. Enter a function (e.g., `x**2 - 4`, `sin(x) - x/2`).
2. Input an initial guess \( x_0 \).
3. Set an epsilon value (default: `0.001`).
4. Click the **Calculate** button to view the root-finding process.

## License
This project is licensed under the **MIT License**.

## Contributing
Pull requests are welcome! If you have suggestions, open an issue or submit a PR.

## Author
[Bilol Kobilov](https://github.com/bilolkobilov)

