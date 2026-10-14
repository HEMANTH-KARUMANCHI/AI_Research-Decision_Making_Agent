import pytest
from tools.calculator import calculator

def test_calculator_basic_arithmetic():
    """Test basic arithmetic operations."""
    assert calculator.invoke({"expression": "20000 * 12 * 5"}) == "1200000"
    assert calculator.invoke({"expression": "100 + 50 - 25"}) == "125"
    assert calculator.invoke({"expression": "(10 + 20) * 3"}) == "90"
    assert calculator.invoke({"expression": "150 / 3"}) == "50.0"

def test_calculator_power_and_operators():
    """Test exponentiation and alternate symbols (x, ^)."""
    assert calculator.invoke({"expression": "2 ^ 4"}) == "16"
    assert calculator.invoke({"expression": "5 x 10"}) == "50"

def test_calculator_division_by_zero():
    """Test division by zero returns an error string instead of crashing."""
    result = calculator.invoke({"expression": "100 / 0"})
    assert "Error" in result
    assert "division by zero" in result

def test_calculator_security_sanitization():
    """Test that arbitrary code injection is blocked."""
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert "Error: Expression" in result
    assert "contains invalid mathematical characters" in result
