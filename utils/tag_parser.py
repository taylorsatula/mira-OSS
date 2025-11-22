"""
Tag parsing service for CNS.

Extracts semantic tags from assistant responses.
"""
import re
from typing import Dict, Any, Optional


class TagParser:
    """
    Service for parsing semantic tags from assistant responses.

    Handles memory references and other semantic markup in LLM responses.
    """

    # Tag patterns
    ERROR_ANALYSIS_PATTERN = re.compile(r'<error_analysis\s+error_id=["\']([^"\']+)["\']>(.*?)</error_analysis>', re.DOTALL | re.IGNORECASE)
    # Pattern for memory references: <mira:memory_ref="UUID" />
    MEMORY_REF_PATTERN = re.compile(
        r'<mira:memory_ref\s*=\s*["\']([a-f0-9-]{36})["\']?\s*/?>', 
        re.IGNORECASE
    )
    # Pattern for touchstone extraction: <mira:touchstone>content</mira:touchstone>
    TOUCHSTONE_PATTERN = re.compile(
        r'<mira:touchstone>(.*?)</mira:touchstone>',
        re.DOTALL | re.IGNORECASE
    )
    # Pattern for emotion emoji: <mira:my_emotion>emoji</mira:my_emotion>
    EMOTION_PATTERN = re.compile(
        r'<mira:my_emotion>\s*([^\s<]+)\s*</mira:my_emotion>',
        re.IGNORECASE
    )
    # Pattern for segment display title: <mira:display_title>title</mira:display_title>
    DISPLAY_TITLE_PATTERN = re.compile(
        r'<mira:display_title>(.*?)</mira:display_title>',
        re.DOTALL | re.IGNORECASE
    )
    # Pattern for segment complexity score: <mira:complexity>1-3</mira:complexity>
    COMPLEXITY_PATTERN = re.compile(
        r'<mira:complexity>\s*([123])\s*</mira:complexity>',
        re.IGNORECASE
    )
    
    def parse_response(self, response_text: str, preserve_tags: list = None) -> Dict[str, Any]:
        """
        Parse all tags from response text.

        Args:
            response_text: Assistant response to parse
            preserve_tags: Optional list of tag names to preserve in clean_text (e.g., ['my_emotion'])

        Returns:
            Dictionary with parsed tag information
        """
        # Extract error analysis
        error_analyses = []
        for match in self.ERROR_ANALYSIS_PATTERN.finditer(response_text):
            error_analyses.append({
                'error_id': match.group(1),
                'analysis': match.group(2).strip()
            })

        # Extract memory references
        memory_refs = []
        for match in self.MEMORY_REF_PATTERN.finditer(response_text):
            memory_refs.append(match.group(1))

        # Extract touchstone
        touchstone = None
        touchstone_match = self.TOUCHSTONE_PATTERN.search(response_text)
        if touchstone_match:
            touchstone_text = touchstone_match.group(1).strip()
            if touchstone_text:
                touchstone = touchstone_text

        # Extract emotion emoji
        emotion = None
        emotion_match = self.EMOTION_PATTERN.search(response_text)
        if emotion_match:
            emotion_text = emotion_match.group(1).strip()
            if emotion_text:
                emotion = emotion_text

        # Extract display title
        display_title = None
        display_title_match = self.DISPLAY_TITLE_PATTERN.search(response_text)
        if display_title_match:
            display_title_text = display_title_match.group(1).strip()
            if display_title_text:
                display_title = display_title_text

        # Extract complexity score
        complexity = None
        complexity_match = self.COMPLEXITY_PATTERN.search(response_text)
        if complexity_match:
            complexity = int(complexity_match.group(1))

        parsed = {
            'error_analysis': error_analyses,
            'referenced_memories': memory_refs,
            'touchstone': touchstone,
            'emotion': emotion,
            'display_title': display_title,
            'complexity': complexity,
            'clean_text': self.remove_all_tags(response_text, preserve_tags=preserve_tags)
        }

        return parsed

    def remove_all_tags(self, text: str, preserve_tags: list = None) -> str:
        """
        Remove all semantic tags from text for clean display.

        Args:
            text: Text with tags
            preserve_tags: Optional list of tag names to preserve (e.g., ['my_emotion', 'touchstone'])

        Returns:
            Text with tags removed (except preserved ones)
        """
        preserve_tags = preserve_tags or []

        if preserve_tags:
            # Build pattern to match all tags EXCEPT preserved ones
            preserve_pattern = '|'.join(re.escape(tag) for tag in preserve_tags)

            # Remove paired mira tags that are NOT in preserve list
            text = re.sub(
                r'<mira:([^>\/\s]+)(?:\s[^>]*)?>[\s\S]*?</mira:\1>',
                lambda m: m.group(0) if m.group(1).lower() in [t.lower() for t in preserve_tags] else '',
                text,
                flags=re.IGNORECASE
            )

            # Remove self-closing mira tags that are NOT in preserve list
            text = re.sub(
                r'<mira:([^>\s\/]+)[^>]*\/>',
                lambda m: m.group(0) if m.group(1).lower() in [t.lower() for t in preserve_tags] else '',
                text,
                flags=re.IGNORECASE
            )

            # Remove any remaining malformed mira tags (but not preserved ones)
            text = re.sub(
                r'</?mira:([^>\s]+)[^>]*>',
                lambda m: m.group(0) if m.group(1).lower() in [t.lower() for t in preserve_tags] else '',
                text,
                flags=re.IGNORECASE
            )
        else:
            # Remove all paired mira tags with their content
            text = re.sub(r'<mira:([^>\/\s]+)(?:\s[^>]*)?>[\s\S]*?</mira:\1>', '', text, flags=re.IGNORECASE)

            # Remove all self-closing mira tags
            text = re.sub(r'<mira:[^>]*\/>', '', text, flags=re.IGNORECASE)

            # Remove any remaining malformed mira tags
            text = re.sub(r'</?mira:[^>]*>', '', text, flags=re.IGNORECASE)

        # Remove specific error analysis patterns
        text = self.ERROR_ANALYSIS_PATTERN.sub('', text)

        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove blank lines
        text = text.strip()

        return text