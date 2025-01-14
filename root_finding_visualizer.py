import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, lambdify, diff, symbols


class RootFindingAlgorithms:
    @staticmethod
    def bisection(func, a, b, tol, max_iter):
        iterations = []
        if func(a) * func(b) >= 0:
            return None, None

        for _ in range(max_iter):
            c = (a + b) / 2
            iterations.append(c)

            if abs(func(c)) < tol or abs(b - a) < tol:
                break

            if func(a) * func(c) < 0:
                b = c
            else:
                a = c

        return c, iterations

    @staticmethod
    def newton_raphson(func, dfunc, x0, tol, max_iter):
        iterations = [x0]

        for _ in range(max_iter):
            fx = func(x0)
            dfx = dfunc(x0)

            if abs(dfx) < 1e-10:
                break

            x1 = x0 - fx / dfx
            iterations.append(x1)

            if abs(x1 - x0) < tol:
                break

            x0 = x1

        return x1, iterations

    @staticmethod
    def secant(func, x0, x1, tol, max_iter):
        iterations = [x0, x1]

        for _ in range(max_iter):
            f0, f1 = func(x0), func(x1)

            if abs(f1 - f0) < 1e-10:
                break

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            iterations.append(x2)

            if abs(x2 - x1) < tol:
                break

            x0, x1 = x1, x2

        return x2, iterations


class RootFindingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Root Finding Algorithms Visualizer")
        self.setup_ui()

    def setup_ui(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(frame, text="Function f(x):").grid(row=0, column=0, sticky=tk.W)
        self.function_entry = tk.Entry(frame, width=30)
        self.function_entry.grid(row=0, column=1, pady=5)

        tk.Label(frame, text="Initial guess / Interval [a, b]:").grid(row=1, column=0, sticky=tk.W)
        self.initial_entry = tk.Entry(frame, width=30)
        self.initial_entry.grid(row=1, column=1, pady=5)

        tk.Label(frame, text="Tolerance:").grid(row=2, column=0, sticky=tk.W)
        self.tolerance_entry = tk.Entry(frame, width=30)
        self.tolerance_entry.insert(0, "1e-6")
        self.tolerance_entry.grid(row=2, column=1, pady=5)

        tk.Label(frame, text="Max Iterations:").grid(row=3, column=0, sticky=tk.W)
        self.iterations_entry = tk.Entry(frame, width=30)
        self.iterations_entry.insert(0, "50")
        self.iterations_entry.grid(row=3, column=1, pady=5)

        tk.Label(frame, text="Algorithm:").grid(row=4, column=0, sticky=tk.W)
        self.algorithm_combo = ttk.Combobox(frame, values=["Bisection", "Newton-Raphson", "Secant"])
        self.algorithm_combo.current(0)
        self.algorithm_combo.grid(row=4, column=1, pady=5)

        tk.Button(frame, text="Visualize", command=self.visualize).grid(row=5, columnspan=2, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def visualize(self):
        func_text = self.function_entry.get()
        initial_text = self.initial_entry.get()
        tol = float(self.tolerance_entry.get())
        max_iter = int(self.iterations_entry.get())
        algorithm = self.algorithm_combo.get()

        try:
            variables = symbols('x')
            func_expr = sympify(func_text)
            func = lambdify(variables, func_expr, modules=["numpy"])
            dfunc = lambdify(variables, diff(func_expr, variables), modules=["numpy"])

            if algorithm == "Bisection":
                a, b = map(float, initial_text.split(','))
                root, iterations = RootFindingAlgorithms.bisection(func, a, b, tol, max_iter)
            elif algorithm == "Newton-Raphson":
                x0 = float(initial_text)
                root, iterations = RootFindingAlgorithms.newton_raphson(func, dfunc, x0, tol, max_iter)
            elif algorithm == "Secant":
                x0, x1 = map(float, initial_text.split(','))
                root, iterations = RootFindingAlgorithms.secant(func, x0, x1, tol, max_iter)

            self.plot(func, iterations)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot(self, func, iterations):
        self.ax.clear()

        x_vals = np.linspace(min(iterations) - 1, max(iterations) + 1, 500)
        y_vals = func(x_vals)
        self.ax.plot(x_vals, y_vals, label="f(x)", color="blue")

        for i, x in enumerate(iterations):
            self.ax.scatter(x, func(x), color="red")
            self.ax.text(x, func(x), f"x{i}", fontsize=8)

        self.ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        self.ax.legend()
        self.ax.grid()

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RootFindingVisualizer(root)
    root.mainloop()
