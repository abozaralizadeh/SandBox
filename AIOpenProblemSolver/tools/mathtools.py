import os
import subprocess
import tempfile

from langchain_core.tools import tool

SANDBOX_TIMEOUT = int(os.getenv("AIOPS_SANDBOX_TIMEOUT", "120"))
MAX_OUTPUT_CHARS = 50000

MATH_PREAMBLE = """\
import math
import itertools
import functools
import collections
from fractions import Fraction
from decimal import Decimal
try:
    import numpy as np
except ImportError:
    pass
try:
    import sympy
    from sympy import *
    x, y, z, t, n, k, m, s = symbols('x y z t n k m s')
except ImportError:
    pass
try:
    import scipy
    from scipy import optimize, integrate, linalg, special, stats
except ImportError:
    pass
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass
"""


@tool
def python_math_sandbox(code: str) -> str:
    """Execute Python code for mathematical computation and exploration.

    Runs Python in an isolated subprocess with mathematical libraries pre-imported:
    sympy (with common symbols x,y,z,t,n,k,m,s), numpy (as np), scipy, math,
    itertools, functools, fractions, decimal, collections, and matplotlib.

    Use this tool to:
    - Test mathematical conjectures with concrete numerical examples
    - Perform symbolic algebra and calculus (sympy)
    - Run numerical simulations and searches (numpy/scipy)
    - Verify proof steps by checking specific cases
    - Search for patterns, counterexamples, or special values
    - Compute invariants, eigenvalues, series expansions, zeta zeros, etc.

    Args:
        code: Python code to execute. Math libraries are pre-imported. Use print() to display results.

    Returns:
        Combined stdout and stderr from execution, or error message.
    """
    full_code = MATH_PREAMBLE + "\n" + code
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        output = output.strip()
        if not output:
            output = "[No output produced. Use print() to see results.]"
        if len(output) > MAX_OUTPUT_CHARS:
            output = (
                output[:MAX_OUTPUT_CHARS]
                + "\n\n[Output truncated to stay within limits.]"
            )
        return output
    except subprocess.TimeoutExpired:
        return f"[Execution timed out after {SANDBOX_TIMEOUT} seconds. Simplify the computation or reduce iterations.]"
    except Exception as exc:
        return f"[Sandbox error: {exc}]"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


SYMPY_OPERATIONS = {
    "simplify",
    "expand",
    "factor",
    "solve",
    "integrate",
    "differentiate",
    "series",
    "limit",
    "latex",
    "evaluate",
}


@tool
def symbolic_calculator(
    expression: str,
    operation: str = "simplify",
    variable: str = "x",
) -> str:
    """Quick symbolic mathematics calculator using SymPy.

    Evaluate or transform a mathematical expression without writing full Python code.

    Args:
        expression: A SymPy-compatible math expression, e.g. "x**2 + 2*x + 1",
                    "sin(x)/x", "Sum(1/n**2, (n, 1, oo))".
        operation: One of: simplify, expand, factor, solve, integrate,
                   differentiate, series, limit, latex, evaluate.
        variable: The primary variable (default: "x").

    Returns:
        The result of applying the operation to the expression.
    """
    try:
        import sympy as sp
    except ImportError:
        return "[Error: sympy is not installed.]"

    op = operation.strip().lower()
    if op not in SYMPY_OPERATIONS:
        return f"[Unknown operation '{operation}'. Use one of: {', '.join(sorted(SYMPY_OPERATIONS))}]"

    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression, locals={variable: var})
    except Exception as exc:
        return f"[Could not parse expression: {exc}]"

    try:
        if op == "simplify":
            result = sp.simplify(expr)
        elif op == "expand":
            result = sp.expand(expr)
        elif op == "factor":
            result = sp.factor(expr)
        elif op == "solve":
            result = sp.solve(expr, var)
        elif op == "integrate":
            result = sp.integrate(expr, var)
        elif op == "differentiate":
            result = sp.diff(expr, var)
        elif op == "series":
            result = sp.series(expr, var, n=10)
        elif op == "limit":
            result = sp.limit(expr, var, sp.oo)
        elif op == "latex":
            result = sp.latex(expr)
        elif op == "evaluate":
            result = expr.evalf()
        else:
            result = expr

        output = str(result)
        if len(output) > MAX_OUTPUT_CHARS:
            output = (
                output[:MAX_OUTPUT_CHARS]
                + "\n[Output truncated.]"
            )
        return output
    except Exception as exc:
        return f"[Computation error: {exc}]"
