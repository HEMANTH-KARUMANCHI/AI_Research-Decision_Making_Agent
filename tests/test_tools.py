import pytest
from tools.text_analysis import text_analysis
from tools.web_search import web_search

def test_text_analysis_normal_text():
    """Test text analysis returns word count, sentence count, and keywords."""
    sample = "Python and LangGraph are great for building AI agents. Python is powerful."
    result = text_analysis.invoke({"text": sample})
    assert "Word Count:" in result
    assert "Sentence Count:" in result
    assert "Top Keywords:" in result
    assert "python" in result.lower()

def test_text_analysis_empty_text():
    """Test text analysis handles empty strings safely."""
    result = text_analysis.invoke({"text": ""})
    assert "Error: Empty text" in result

def test_web_search_callable():
    """Test that web_search tool is invokable and returns a string."""
    result = web_search.invoke({"query": "Python programming language"})
    assert isinstance(result, str)
    assert len(result) > 0
