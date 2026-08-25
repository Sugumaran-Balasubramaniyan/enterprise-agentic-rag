import math

class CalculatorTool:
    name = "calculator"
    description = "Performs deterministic mathematical and cloud infrastructure sizing calculations."

    def execute(self, expression: str) -> float:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        clean_expr = "".join([c for c in expression if c in "0123456789+-*/()., eE"])
        try:
            return float(eval(clean_expr, {"__builtins__": None}, allowed_names))
        except Exception:
            return 0.0
