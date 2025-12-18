import ast
import math
import sys
from dataclasses import dataclass
import builtins
import re
from rich import print


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
#!/usr/bin/env python3
"""
Terminal-based calculator (calc.py)

Features:
- Safe expression evaluation using AST (no arbitrary code execution)
- Basic arithmetic, power, modulo, floor division
- math module functions/constants (sin, cos, pi, e, etc.)
- Variables assignment (e.g. x = 2 * pi)
- Commands: quit/exit, help, vars, history, clear, settings/set
- 'ans' token: refers to previous calculation result (error if none)
"""


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
        raise ValueError("Only numeric constants are allowed")

    # For Python <3.8 compatibility
    def visit_Num(self, node):
        return node.n

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
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

    print("Simple terminal calculator. Type [green]'help'[/green] for commands.\n")
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
                "[yellow]Enter expressions to evaluate. Commands[/yellow]: [green]quit, help, vars, history, clear, settings[/green]"
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
