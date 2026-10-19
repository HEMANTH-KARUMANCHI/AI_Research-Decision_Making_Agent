from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Useful for searching the web for current events, technical comparisons, specifications, and real-time information.
    
    Args:
        query: Clear search keywords (e.g., 'NVIDIA H100 GPU price specifications', 'PostgreSQL vs MongoDB differences').
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=4)
        if not results:
            return "No relevant search results found."

        formatted_results = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted_results.append(f"Title: {title}\nSnippet: {body}\nURL: {href}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Web search error: {str(e)}"
