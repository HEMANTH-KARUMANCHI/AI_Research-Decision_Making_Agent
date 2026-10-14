import re
from collections import Counter
from langchain_core.tools import tool

@tool
def text_analysis(text: str) -> str:
    """Useful for analyzing text content to extract statistics, word counts, sentence metrics, and key terms.
    
    Args:
        text: The block of text or document content to analyze.
    """
    try:
        if not text or not text.strip():
            return "Error: Empty text provided for analysis."

        # Word count & Character count
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        total_chars = len(text)
        
        # Sentence count
        sentences = [s.strip() for s in re.split(r'[\.\!\?]+', text) if s.strip()]
        total_sentences = max(len(sentences), 1)

        # Average word length & reading time (approx 200 words/min)
        avg_word_length = round(sum(len(w) for w in words) / max(total_words, 1), 1)
        reading_time_mins = round(total_words / 200, 2)

        # Stop words filter for top keywords
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "should", "can", "could", "may", "might", "must",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }

        filtered_words = [w for w in words if len(w) > 2 and w not in stop_words]
        top_keywords = Counter(filtered_words).most_common(5)
        keywords_str = ", ".join([f"{kw} ({count})" for kw, count in top_keywords])

        summary = (
            f"=== Text Analysis Metrics ===\n"
            f"• Word Count: {total_words}\n"
            f"• Character Count: {total_chars}\n"
            f"• Sentence Count: {total_sentences}\n"
            f"• Avg Word Length: {avg_word_length} chars\n"
            f"• Est Reading Time: {reading_time_mins} minutes\n"
            f"• Top Keywords: {keywords_str if keywords_str else 'None'}"
        )

        return summary
    except Exception as e:
        return f"Error performing text analysis: {str(e)}"
