from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def web_search(query: str) -> str:
    """Useful for searching the web for current events, technical comparisons, specifications, and real-time information.
    
    Args:
        query: Clear search keywords (e.g., 'NVIDIA H100 GPU price specifications', 'PostgreSQL vs MongoDB differences').
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        if not results:
            return "No relevant search results found."
        return str(results)
    except Exception as e:
        return f"Web search error: {str(e)}"
