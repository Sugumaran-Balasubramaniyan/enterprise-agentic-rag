import ast
import math
import operator
import re
from typing import Dict, Any, Optional, Union


class CalculatorTool:
    """
    Performs deterministic mathematical, cloud infrastructure sizing, and capacity calculations.
    Supports safe arithmetic evaluation (AST-based) and standard cloud formulas:
    - Vector index RAM sizing: N * D * 4 bytes
    - Concurrency / Little's Law: QPS * latency_ms / 1000
    - Monthly storage & bandwidth cost: storage_gb * price_per_gb
    - Instance replica sizing: ceil(total_qps / qps_per_instance)
    """
    name = "calculator"
    description = (
        "Performs deterministic arithmetic calculations and cloud infrastructure sizing "
        "(RAM per vectors, QPS concurrency, storage costs, replica sizing)."
    )

    ALLOWED_MATH = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def vector_ram(
        self,
        num_vectors: Union[int, float],
        dimension: int,
        bytes_per_element: int = 4,
        overhead_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates memory required for embedding vectors.
        Formula: RAM = N * D * bytes_per_element * overhead_factor
        """
        num_vec = int(num_vectors)
        dim = int(dimension)
        raw_bytes = num_vec * dim * bytes_per_element
        total_bytes = raw_bytes * overhead_factor
        
        mb = total_bytes / (1024 ** 2)
        gib = total_bytes / (1024 ** 3)
        decimal_gb = total_bytes / 1e9

        return {
            "result": round(total_bytes, 2),
            "formatted": f"{gib:.2f} GiB ({decimal_gb:.2f} GB, {int(total_bytes):,} bytes)",
            "formula_type": "vector_ram",
            "details": {
                "num_vectors": num_vec,
                "dimension": dim,
                "bytes_per_element": bytes_per_element,
                "overhead_factor": overhead_factor,
                "bytes": int(total_bytes),
                "mb": round(mb, 2),
                "gib": round(gib, 2),
                "decimal_gb": round(decimal_gb, 2)
            },
            "success": True,
            "error": None
        }

    def concurrency(self, qps: float, latency_ms: float) -> Dict[str, Any]:
        """
        Calculates required system concurrency (Little's Law).
        Formula: Concurrency = QPS * (latency_ms / 1000)
        """
        conc = float(qps) * (float(latency_ms) / 1000.0)
        return {
            "result": round(conc, 4),
            "formatted": f"{conc:.2f} concurrent requests",
            "formula_type": "concurrency",
            "details": {
                "qps": float(qps),
                "latency_ms": float(latency_ms),
                "concurrent_requests": round(conc, 4)
            },
            "success": True,
            "error": None
        }

    def monthly_storage_cost(self, storage_gb: float, price_per_gb: float = 0.023) -> Dict[str, Any]:
        """
        Calculates monthly storage cost.
        Formula: Cost = storage_gb * price_per_gb
        """
        cost = float(storage_gb) * float(price_per_gb)
        return {
            "result": round(cost, 2),
            "formatted": f"${cost:.2f}/month",
            "formula_type": "monthly_storage_cost",
            "details": {
                "storage_gb": float(storage_gb),
                "price_per_gb": float(price_per_gb),
                "monthly_cost_usd": round(cost, 2)
            },
            "success": True,
            "error": None
        }

    def replicas_needed(self, total_qps: float, qps_per_instance: float) -> Dict[str, Any]:
        """
        Calculates required instances for given load.
        Formula: Replicas = ceil(total_qps / qps_per_instance)
        """
        inst_qps = max(float(qps_per_instance), 0.0001)
        count = math.ceil(float(total_qps) / inst_qps)
        return {
            "result": int(count),
            "formatted": f"{count} instances",
            "formula_type": "replicas_needed",
            "details": {
                "total_qps": float(total_qps),
                "qps_per_instance": inst_qps,
                "replicas": count
            },
            "success": True,
            "error": None
        }

    def _eval_ast(self, node: ast.AST) -> Union[float, int, Any]:
        """Recursively evaluates an AST node with strict whitelisting."""
        if isinstance(node, ast.Expression):
            return self._eval_ast(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")

        if isinstance(node, ast.Name):
            if node.id in self.ALLOWED_MATH:
                return self.ALLOWED_MATH[node.id]
            raise ValueError(f"Undefined identifier: {node.id}")

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self._OPERATORS:
                operand = self._eval_ast(node.operand)
                return self._OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self._OPERATORS:
                left = self._eval_ast(node.left)
                right = self._eval_ast(node.right)
                return self._OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")

        if isinstance(node, ast.Call):
            func = self._eval_ast(node.func)
            if not callable(func):
                raise ValueError(f"Called object is not callable: {func}")
            args = [self._eval_ast(arg) for arg in node.args]
            return func(*args)

        raise ValueError(f"Unsupported expression structure: {type(node).__name__}")

    def _parse_and_eval_arithmetic(self, expression: str) -> Dict[str, Any]:
        """Safely evaluates an arithmetic expression using AST parsing."""
        if not expression or not expression.strip():
            return {
                "result": 0.0,
                "formatted": "0.0",
                "formula_type": "arithmetic",
                "details": {},
                "success": False,
                "error": "Empty expression"
            }

        clean_expr = expression.strip().replace(",", "")
        try:
            parsed = ast.parse(clean_expr, mode="eval")
            val = self._eval_ast(parsed)
            num_val = float(val) if isinstance(val, (int, float)) else 0.0
            return {
                "result": num_val,
                "formatted": f"{num_val:.4f}".rstrip("0").rstrip(".") if "." in f"{num_val:.4f}" else str(num_val),
                "formula_type": "arithmetic",
                "details": {"expression": expression, "evaluated": clean_expr},
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "result": 0.0,
                "formatted": "0.0",
                "formula_type": "arithmetic",
                "details": {"expression": expression},
                "success": False,
                "error": f"Evaluation error: {str(e)}"
            }

    def _detect_cloud_formula(self, text: str) -> Optional[Dict[str, Any]]:
        """Detects domain-specific sizing parameters from text."""
        # 1. Vector RAM sizing detection
        # Match e.g. "5000000 vectors with 1536 dimensions" or "5M vectors, 768 dim"
        vec_match = re.search(
            r"(?:(\d[\d,_]*)\s*(?:million|m)?\s*vectors?|vectors?.*?(\d[\d,_]*)).*?(?:(\d+)\s*(?:dimensions?|dims?|dim)|(?:dimension|dim).*?(\d+))",
            text,
            re.IGNORECASE
        )
        if vec_match:
            g = vec_match.groups()
            vec_raw = g[0] or g[1]
            dim_raw = g[2] or g[3]
            if vec_raw and dim_raw:
                vec_val = float(vec_raw.replace(",", "").replace("_", ""))
                if re.search(r"\bmillion\b|\bm\b", text, re.IGNORECASE) and vec_val < 10000:
                    vec_val *= 1_000_000
                dim_val = int(dim_raw)
                return self.vector_ram(num_vectors=vec_val, dimension=dim_val)

        # 2. Concurrency / Little's Law
        # Match e.g. "500 QPS with 20ms latency"
        qps_match = re.search(
            r"(\d+(?:\.\d+)?)\s*qps.*?(\d+(?:\.\d+)?)\s*(?:ms|latency)",
            text,
            re.IGNORECASE
        )
        if not qps_match:
            qps_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:ms|latency).*?(\d+(?:\.\d+)?)\s*qps",
                text,
                re.IGNORECASE
            )
            if qps_match:
                lat = float(qps_match.group(1))
                qps = float(qps_match.group(2))
                return self.concurrency(qps=qps, latency_ms=lat)
        else:
            qps = float(qps_match.group(1))
            lat = float(qps_match.group(2))
            return self.concurrency(qps=qps, latency_ms=lat)

        # 3. Monthly Storage Cost
        # Match e.g. "500 GB storage at 0.023 per GB"
        storage_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:gb|tb)\s*(?:storage)?.*?(?:(\d+(?:\.\d+)?)\s*(?:per|\$|usd))?",
            text,
            re.IGNORECASE
        )
        if storage_match and ("storage" in text.lower() or "cost" in text.lower() or "$" in text):
            storage_amt = float(storage_match.group(1))
            if "tb" in text.lower():
                storage_amt *= 1024
            rate = float(storage_match.group(2)) if storage_match.group(2) else 0.023
            return self.monthly_storage_cost(storage_gb=storage_amt, price_per_gb=rate)

        return None

    def execute(
        self,
        expression: str = "",
        formula_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes a calculation either via explicit formula type or auto-detection.
        Returns structured dictionary with result, formatted string, formula_type, and details.
        """
        if formula_type == "vector_ram" or ("num_vectors" in kwargs and "dimension" in kwargs):
            return self.vector_ram(
                num_vectors=kwargs.get("num_vectors", 0),
                dimension=kwargs.get("dimension", 1536),
                bytes_per_element=kwargs.get("bytes_per_element", 4),
                overhead_factor=kwargs.get("overhead_factor", 1.0)
            )

        if formula_type == "concurrency" or ("qps" in kwargs and "latency_ms" in kwargs):
            return self.concurrency(
                qps=kwargs.get("qps", 0.0),
                latency_ms=kwargs.get("latency_ms", 0.0)
            )

        if formula_type == "monthly_storage_cost" or "storage_gb" in kwargs:
            return self.monthly_storage_cost(
                storage_gb=kwargs.get("storage_gb", 0.0),
                price_per_gb=kwargs.get("price_per_gb", 0.023)
            )

        if formula_type == "replicas_needed" or ("total_qps" in kwargs and "qps_per_instance" in kwargs):
            return self.replicas_needed(
                total_qps=kwargs.get("total_qps", 0.0),
                qps_per_instance=kwargs.get("qps_per_instance", 100.0)
            )

        # Check for natural language / domain formulas
        if expression:
            cloud_res = self._detect_cloud_formula(expression)
            if cloud_res:
                return cloud_res

            # Fallback to safe arithmetic evaluator
            return self._parse_and_eval_arithmetic(expression)

        return {
            "result": 0.0,
            "formatted": "0.0",
            "formula_type": "unknown",
            "details": {},
            "success": False,
            "error": "No expression or parameters supplied."
        }
