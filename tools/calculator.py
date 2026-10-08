import math
import re
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for evaluating mathematical expressions and arithmetic calculations.
    
    Args:
        expression: A string mathematical expression (e.g., '20000 * 12 * 5', '(15000 + 5000) / 2', '2 ** 10').
    """
    try:
        # Sanitize expression: allow numbers, basic math operators, parentheses, and spaces
        cleaned_expr = expression.strip()
        
        # Replace common user operators if any
        cleaned_expr = cleaned_expr.replace("^", "**").replace("x", "*").replace("X", "*")
        
        # Check for disallowed characters to prevent security issues
        if not re.match(r'^[0-9\.\+\-\*\/\(\)\s\%\*\*]+$', cleaned_expr):
            return f"Error: Expression '{expression}' contains invalid mathematical characters."

        # Evaluate safely in an isolated namespace
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "pow": pow
        }
        
        result = eval(cleaned_expr, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"
