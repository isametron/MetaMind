"""Static checks on candidate source code, run before it is executed.

Violations reject the candidate (fitness 0, with a message the LLM loop can
feed back). Flags are allowed but recorded, for the exploit audit in the paper.

This raises the bar for an LLM trying to cheat the evaluator; it is not a
security sandbox. The harness's own defences (a plain-function objective,
harness-measured traces, budget enforcement, post-run checks) still apply.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Modules a candidate may import (submodules included, e.g. numpy.linalg).
ALLOWED_MODULES = {
    "numpy", "math", "random", "typing", "functools", "itertools", "collections", "heapq",
    "bisect", "dataclasses", "copy", "statistics", "numbers", "enum", "abc", "time", "__future__",
}

# Builtins that give access to objects, code or files the candidate shouldn't reach.
FORBIDDEN_NAMES = {
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "eval", "exec", "compile",
    "__import__", "open", "input", "breakpoint", "memoryview", "__builtins__",
}

# Attributes that lead into function internals, frames or the class hierarchy,
# plus numpy's file I/O.
FORBIDDEN_ATTRS = {
    "__closure__", "__globals__", "__code__", "__dict__", "__builtins__", "__subclasses__",
    "__mro__", "__bases__", "__base__", "__self__", "__func__", "__wrapped__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__defaults__", "__kwdefaults__", "__loader__", "__spec__",
    "cell_contents", "f_back", "f_globals", "f_locals", "f_builtins", "gi_frame", "tb_frame",
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap",
}

HARD_CODED_BOUNDS = {5, 5.0, 5.12}


@dataclass
class CodeAudit:
    violations: list[str] = field(default_factory=list)  # reject the candidate
    flags: list[str] = field(default_factory=list)       # allowed, but recorded

    @property
    def ok(self) -> bool:
        return not self.violations


def _is_bound_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value in HARD_CODED_BOUNDS


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def audit_code(code: str) -> CodeAudit:
    """Check candidate source. Raises SyntaxError if it doesn't parse."""
    tree = ast.parse(code, filename="<candidate>")
    audit = CodeAudit()

    def violation(node: ast.AST, msg: str) -> None:
        audit.violations.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def flag(node: ast.AST, msg: str) -> None:
        audit.flags.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in modules:
                if m.split(".")[0] not in ALLOWED_MODULES:
                    violation(node, f"import of '{m}' is not allowed (allowed: numpy, math, random and "
                                    "a few standard-library helpers)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violation(node, f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            violation(node, f"access to '.{node.attr}' is not allowed")
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "BaseException"):
                flag(node, "bare except / except BaseException")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ("seed", "default_rng", "RandomState", "Generator") and (node.args or node.keywords):
                flag(node, f"reseeds the random number generator ({name})")
            if name in ("uniform", "clip") and any(_is_bound_literal(a) for a in node.args):
                flag(node, f"hard-coded search bounds in {name}() instead of using `bounds`")
    return audit
