#!/usr/bin/env python3
"""
Terminal-based calculator (calc.py)

Features:
- Safe expression evaluation using AST (no arbitrary code execution)
- Basic arithmetic, power, modulo, floor division
- math module functions/constants (sin, cos, pi, e, etc.)
- Variables assignment (e.g. x = 2 * pi)
- Formulas catalogue
- Plotting of functions (matplotlib)
- 'ans' token: refers to previous calculation result (error if none)
- Settings menu
- Result history access
- Commands: quit/exit, help, plot, formulas, formula, vars, history, clear, settings/set
"""

import ast
import math
import sys
from dataclasses import dataclass
import builtins
import re
from rich import print
import builtins as _builtins

# Catalogue of formulas accessible from the REPL as math.formulas() and math.formula(name)
# e.g. formulas:   formulas()              -> lists available formulas
#       single:    formula("circle_area")  -> shows details / expression

FORMULAS = {
    "pythagoras": {
        "title": "Pythagorean theorem",
        "formula": "c = sqrt(a² + b²)",
        "expr": "sqrt(a**2 + b**2)",
        "vars": "a, b (legs); c (hypotenuse)",
        "example": "a=3, b=4 -> c = sqrt(3**2+4**2)",
    },
    "law_sines": {
        "title": "Law of sines",
        "formula": "a/sin(A) = b/sin(B) = c/sin(C)",
        "expr": "a/sin(A) , b/sin(B) , c/sin(C)",
        "vars": "a,b,c (sides); A,B,C (opposite angles of each side)",
        "example": "a=5,A=30,B=45 -> b = (5 * sin(45)) / sin(30)",
    },
    "law_cosines": {
        "title": "Law of cosines",
        "formula": "c² = a² + b² - 2ab cos(C)",
        "expr": "sqrt(a**2 + b**2 - 2*a*b*cos(C))",
        "vars": "a,b (sides); C (angle opposite side c)",
        "example": "a=4,b=5,C=60 -> c = sqrt(4**2 + 5**2 - 2*4*5*cos(60))",
    },
    "herons_area": {
        "title": "Area of triangle (Heron's formula)",
        "formula": "S = sqrt(p(p-a)(p-b)(p-c)) where p=(a+b+c)/2",
        "expr": "sqrt(p*(p - a)*(p - b)*(p - c)) where p = (a + b + c) / 2",
        "vars": "a, b, c (sides)",
        "example": "a=3,b=4,c=5 -> S = sqrt(p*(p-3)*(p-4)*(p-5)) where p=(3+4+5)/2",
    },
    "distance_2d": {
        "title": "Distance between two points in 2D",
        "formula": "d = sqrt((x2 - x1)² + (y2 - y1)²)",
        "expr": "sqrt((x2 - x1)**2 + (y2 - y1)**2)",
        "vars": "x1, y1, x2, y2 (coordinates of the two points)",
        "example": "x1=1,y1=2,x2=4,y2=6 -> d = sqrt((4-1)**2 + (6-2)**2)",
    },
    "distance_3d": {
        "title": "Distance between two points in 3D",
        "formula": "d = sqrt((x2 - x1)² + (y2 - y1)² + (z2 - z1)²)",
        "expr": "sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)",
        "vars": "x1, y1, z1, x2, y2, z2 (coordinates of the two points)",
        "example": "x1=1,y1=2,z1=3,x2=4,y2=6,z2=8 -> d = sqrt((4-1)**2 + (6-2)**2 + (8-3)**2)",
    },
    "quadratic": {
        "title": "Quadratic formula (roots of ax²+bx+c=0 where a≠0)",
        "formula": "x = (-b ± sqrt(b²-4ac)) / (2a)",
        "expr": "(-b + sqrt(b**2 - 4*a*c)) / (2*a), (-b - sqrt(b**2 - 4*a*c)) / (2*a)",
        "vars": "a, b, c",
        "example": "a=1,b=-3,c=2 -> roots 1 and 2",
    },
    "triangle_area": {
        "title": "Area of a triangle",
        "formula": "S = 1/2 * a * b* sin(C)",
        "expr": "a * b * sin(C) / 2",
        "vars": "a, b (sides), C (opposite angle of side c)",
        "example": "a=4,b=3, C = 60 -> S = 4 * 3 * sin(60) / 2",
    },
    "compound_interest": {
        "title": "Compound interest",
        "formula": "A = P (1 + r/n)^(n t)",
        "expr": "P * (1 + r/n) ** (n * t)",
        "vars": "P (principal), r (rate), n (compounds/year), t (years)",
        "example": "P=1000,r=0.05,n=12,t=10",
    },
}


def _format_formula_entry(key, info):
    lines = [
        f"\n  [green]Formula[/green] : {info.get('formula','')}",
        f"  [green]Expr[/green]    : {info.get('expr','')}",
        f"  [green]Vars[/green]    : {info.get('vars','')}",
    ]
    if info.get("example"):
        lines.append(f"  [green]Example[/green] : {info.get('example')}\n")
    return "\n".join(lines)


def _list_formulas():
    # Return a multi-line string listing available formulas (also prints for convenience)
    keys = sorted(FORMULAS.keys())
    lines = [f"[green]{k}[/green]: {FORMULAS[k].get('title','')}" for k in keys]
    s = "\n".join(lines)
    print(f"\n{s}\n")


def _show_formula(name):
    print(name)
    if not isinstance(name, str):
        raise TypeError("formula name must be a string")
    key = name.strip().lower()
    if key in FORMULAS:
        s = _format_formula_entry(key, FORMULAS[key])
        print(s)
    else:
        raise NameError(f"Unknown formula: {name}")


# Export helpers into the math module so they become available inside the calculator's safe env
setattr(math, "formulas", _list_formulas)
setattr(math, "formula", _show_formula)

import matplotlib.pyplot as _plt

# store the real stdin hook so later wrapper (_input_hook) can call it
_real_input = _builtins.input


def _handle_plot_command(line: str):
    # Supported syntaxes:
    #   plot <expr> from <start> to <end> [samples N]
    #   plot <expr> <start> <end> [N]
    m = re.match(
        r"^\s*(?:plot|graph)\s+(.+?)\s+from\s+([+-]?\d+(?:\.\d*)?)\s+to\s+([+-]?\d+(?:\.\d*)?)(?:\s+samples\s+(\d+))?\s*$",
        line,
        re.I,
    )
    if not m:
        m = re.match(
            r"^\s*(?:plot|graph)\s+(.+?)\s+([+-]?\d+(?:\.\d*)?)\s+([+-]?\d+(?:\.\d*)?)(?:\s+(\d+))?\s*$",
            line,
            re.I,
        )
    if not m:
        print(
            "[yellow]Plot syntax: plot (expr) from (start) to (end) samples N  (e.g. plot sin(x) from 0 to 360 samples 400)[/yellow]"
        )
        return

    expr = m.group(1).strip()
    if expr.lower().startswith("y="):
        expr = expr[2:].strip()
    try:
        start = float(m.group(2))
        end = float(m.group(3))
    except Exception:
        print("[red]Invalid numeric range for plot[/red]")
        return
    samples = int(m.group(4)) if m.group(4) else 400
    if samples < 10 or samples > 10000:
        print("[red]Samples must be between 10 and 10000[/red]")
        return

    xs = [start + (end - start) * i / (samples - 1) for i in range(samples)]
    ys = []
    env = dict(ALLOWED_NAMES)

    for x in xs:
        env["x"] = x
        try:
            v = safe_eval(expr, env)
            ys.append(float(v))
        except Exception:
            ys.append(float("nan"))

    try:
        _plt.figure()
        _plt.plot(xs, ys, "-")
        _plt.xlabel("x")
        _plt.ylabel(expr)
        _plt.title(f"plot: {expr}")
        _plt.axvline(0, color="black")
        _plt.axhline(0, color="black")
        _plt.grid(True)
        _plt.show()
    except Exception as e:
        print("[red]Plot failed:[/red]", e)


def _plot_wrapper(prompt=""):
    try:
        line = _real_input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise
    if line.strip().lower().startswith(("plot ", "graph ")):
        _handle_plot_command(line)
        return ""  # cause repl to skip this iteration
    return line


# Install wrapper so subsequent assignment to _original_input will capture it.
_builtins.input = _plot_wrapper


@dataclass
class Settings:
    sig_figs: int = 6  # significant figures for displayed numeric results
    angle_mode: str = "deg"  # "rad" or "deg"


settings = Settings()

# Wrap selected math trig functions so they obey angle_mode (deg <-> rad conversion).
_original_math_funcs = {}


def _wrap_trig_in(func):
    def wrapped(x, func=func):
        if settings.angle_mode == "deg":
            return func(math.radians(x))
        return func(x)

    return wrapped


def _wrap_trig_out(func):
    def wrapped(x, func=func):
        res = func(x)
        if settings.angle_mode == "deg":
            return math.degrees(res)
        return res

    return wrapped


def _wrap_atan2(func):
    def wrapped(y, x, func=func):
        if settings.angle_mode == "deg":
            y = math.radians(y)
            x = math.radians(x)
            res = func(y, x)
            return math.degrees(res)
        return func(y, x)

    return wrapped


for name in ("sin", "cos", "tan"):
    if hasattr(math, name):
        _original_math_funcs[name] = getattr(math, name)
        setattr(math, name, _wrap_trig_in(_original_math_funcs[name]))

for name in ("asin", "acos", "atan"):
    if hasattr(math, name):
        _original_math_funcs[name] = getattr(math, name)
        setattr(math, name, _wrap_trig_out(_original_math_funcs[name]))

if hasattr(math, "atan2"):
    _original_math_funcs["atan2"] = getattr(math, "atan2")
    setattr(math, "atan2", _wrap_atan2(_original_math_funcs["atan2"]))


# Simple settings interactive page triggered by typing "settings" at prompt.
_original_input = builtins.input


def _settings_menu():
    while True:
        print(
            f"\nCurrent settings: sig_figs={settings.sig_figs}, angle_mode={settings.angle_mode}"
        )
        print(
            "[green]Options[/green]: [ 1 ] set sig figs  [ 2 ] set angle mode  [ 3 ] reset defaults  [ q ] quit settings"
        )
        try:
            choice = _original_input("settings> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            continue
        if choice == "1":
            val = _original_input("Enter significant figures (1-15): ").strip()
            try:
                n = int(val)
                if n < 1 or n > 15:
                    print("Value out of range")
                else:
                    settings.sig_figs = n
                    print("Updated")
            except Exception:
                print("Invalid integer")
        elif choice == "2":
            val = _original_input("Angle mode (rad/deg): ").strip().lower()
            if val in ("rad", "deg"):
                settings.angle_mode = val
                print("Updated")
            else:
                print("Invalid mode")
        elif choice == "3":
            settings.sig_figs = 6
            settings.angle_mode = "rad"
            print("Defaults restored")
        elif choice in ("q", "quit", "exit"):
            return
        else:
            print("Unknown option")


def _input_hook(prompt=""):
    try:
        line = _original_input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise
    if line.strip().lower() == "settings" or line.strip().lower() == "set":
        _settings_menu()
        return ""  # cause repl to skip this iteration
    return line


builtins.input = _input_hook


# Pretty-print numeric results according to significant figures.
_original_print = builtins.print


def _print_hook(*args, **kwargs):
    # Only format when printing a single numeric positional arg to stdout/stderr default
    file = kwargs.get("file", None)
    if (
        file is None
        and len(args) == 1
        and isinstance(args[0], (int, float))
        and not kwargs.get("end")
    ):
        val = args[0]
        if isinstance(val, float):
            s = format(val, f".{settings.sig_figs}g")
            _original_print(s, **kwargs)
            return
        _original_print(val, **kwargs)
        return
    _original_print(*args, **kwargs)


builtins.print = _print_hook


# Environment: whitelist math functions/constants + a few builtins
ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
ALLOWED_NAMES.update({"abs": abs, "round": round})


class Evaluator(ast.NodeVisitor):
    def __init__(self, env):
        self.env = env

    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Module(self, node):
        # For exec mode: allow single Assign or Expr
        if len(node.body) != 1:
            raise ValueError("Only single expressions or assignments allowed")
        stmt = node.body[0]
        if isinstance(stmt, ast.Assign):
            return self.handle_assign(stmt)
        elif isinstance(stmt, ast.Expr):
            return self.visit(stmt.value)
        else:
            raise ValueError("Only expressions or assignments allowed")

    def handle_assign(self, node: ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("Only simple assignments to a single variable are allowed")
        name = node.targets[0].id
        value = self.visit(node.value)
        self.env[name] = value
        return value

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        if node.value in FORMULAS:
            return node.value
        raise ValueError("Only numeric constants are allowed")

    # For Python <3.8 compatibility
    def visit_Num(self, node):
        return node.n

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id == "plot":
            print(
                "[yellow]Plotting syntax: [ plot (expr) from (start) to (end) samples N ][/yellow]"
            )
            return None
        if node.id in FORMULAS:
            return node.id
        raise NameError(f"Unknown name: {node.id}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls allowed")
        func_name = node.func.id
        if func_name not in self.env:
            raise NameError(f"Unknown function: {func_name}")
        func = self.env[func_name]
        args = [self.visit(a) for a in node.args]
        # disallow keywords, starargs, kwargs
        if node.keywords:
            raise ValueError("Keyword arguments not allowed")
        return func(*args)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression: {node.__class__.__name__}")


def safe_eval(expr: str, env: dict):
    """
    Evaluate an expression or assignment safely.
    Returns the resulting value.
    """
    try:
        # Try eval-mode first (single expression)
        tree = ast.parse(expr, mode="eval")
        evaluator = Evaluator(env)
        return evaluator.visit(tree)
    except SyntaxError:
        # Maybe it's an assignment or statement; allow a single assignment or expr
        tree = ast.parse(expr, mode="exec")
        evaluator = Evaluator(env)
        return evaluator.visit(tree)
    except Exception:
        # Re-raise to caller for handling
        raise


def repl():
    env = dict(ALLOWED_NAMES)  # copy
    history = []
    previous_result = None
    prompt = "calc> "

    print("\n\nSimple terminal calculator. Type [green]'help'[/green] for commands.\n")
    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "help":
            print(
                """[yellow]Enter expressions to evaluate. Commands[/yellow]:
[green]quit,
help,
plot <expr>,
formulas,
formula([ name_of_formula ])
history,
vars,
clear,
settings[/green]"""
            )
            continue
        if line == "vars":
            for k in sorted(env.keys()):
                if k in ALLOWED_NAMES:
                    continue
                print(f"{k} = {env[k]}")
            continue
        if line == "history":
            for i, h in enumerate(history, 1):
                print(f"{i}: {h}")
            continue
        if line == "clear":
            # clear user variables only
            for k in list(env.keys()):
                if k not in ALLOWED_NAMES:
                    del env[k]
            # also clear previous_result
            previous_result = None
            print("Cleared variables")
            continue

        if line == "formulas":
            _list_formulas()
            continue

        if line == "formula":
            print('[yellow]Usage: formula("name_of_formula")[/yellow]')
            continue

        # ans token handling: if user refers to 'ans' ensure previous result exists
        if re.search(r"\bans\b", line):
            if previous_result is None:
                print("Error: no previous answer available")
                continue
            env["ans"] = previous_result

        try:
            result = safe_eval(line, env)
            history.append(line)
            if result is not None:
                # Print numbers compactly
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                # update previous result (ans)
                previous_result = result
                env["ans"] = result
                print(result)
        except Exception as e:
            print("Error:", e)


def main():
    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
        # CLI has no previous result available
        if re.search(r"\bans\b", expr):
            print("Error: no previous answer available")
            sys.exit(1)
        try:
            res = safe_eval(expr, dict(ALLOWED_NAMES))
            if res is not None:
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                print(res)
        except Exception as e:
            print("Error:", e)
            sys.exit(1)
    else:
        repl()


if __name__ == "__main__":
    main()
