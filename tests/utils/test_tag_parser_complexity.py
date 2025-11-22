"""
Tests for tag_parser.py complexity extraction.

Focus: Real contract guarantees for complexity tag parsing.
"""
import pytest

from utils.tag_parser import TagParser


class TestComplexityExtraction:
    """Tests enforce complexity tag parsing guarantees."""

    def test_extracts_complexity_one(self):
        """CONTRACT: Parser extracts complexity=1 from valid tag."""
        parser = TagParser()
        text = "Summary text\n<mira:complexity>1</mira:complexity>"

        parsed = parser.parse_response(text)

        assert parsed['complexity'] == 1

    def test_extracts_complexity_two(self):
        """CONTRACT: Parser extracts complexity=2 from valid tag."""
        parser = TagParser()
        text = "Summary text\n<mira:complexity>2</mira:complexity>"

        parsed = parser.parse_response(text)

        assert parsed['complexity'] == 2

    def test_extracts_complexity_three(self):
        """CONTRACT: Parser extracts complexity=3 from valid tag."""
        parser = TagParser()
        text = "Summary text\n<mira:complexity>3</mira:complexity>"

        parsed = parser.parse_response(text)

        assert parsed['complexity'] == 3

    def test_handles_whitespace_around_value(self):
        """CONTRACT: Parser handles whitespace around complexity value."""
        parser = TagParser()
        text = "Summary text\n<mira:complexity>  2  </mira:complexity>"

        parsed = parser.parse_response(text)

        assert parsed['complexity'] == 2

    def test_returns_none_when_tag_missing(self):
        """CONTRACT: Parser returns None when complexity tag is missing."""
        parser = TagParser()
        text = "Summary text\n<mira:display_title>Test</mira:display_title>"

        parsed = parser.parse_response(text)

        assert parsed['complexity'] is None

    def test_ignores_invalid_values(self):
        """CONTRACT: Parser returns None for complexity values outside 1-3 range."""
        parser = TagParser()

        # Test 0
        text_zero = "Summary\n<mira:complexity>0</mira:complexity>"
        assert parser.parse_response(text_zero)['complexity'] is None

        # Test 4
        text_four = "Summary\n<mira:complexity>4</mira:complexity>"
        assert parser.parse_response(text_four)['complexity'] is None

        # Test non-numeric
        text_invalid = "Summary\n<mira:complexity>high</mira:complexity>"
        assert parser.parse_response(text_invalid)['complexity'] is None

    def test_extracts_complexity_with_other_tags(self):
        """CONTRACT: Complexity extraction works alongside other tags."""
        parser = TagParser()
        text = """
Summary paragraph here with details.

<mira:display_title>Test Title</mira:display_title>
<mira:complexity>3</mira:complexity>
<mira:my_emotion>😊</mira:my_emotion>
"""

        parsed = parser.parse_response(text)

        assert parsed['complexity'] == 3
        assert parsed['display_title'] == "Test Title"
        assert parsed['emotion'] == "😊"
        assert "Summary paragraph" in parsed['clean_text']

    def test_case_insensitive_tag_name(self):
        """CONTRACT: Complexity tag parsing is case-insensitive."""
        parser = TagParser()

        # Uppercase
        text_upper = "Summary\n<MIRA:COMPLEXITY>2</MIRA:COMPLEXITY>"
        assert parser.parse_response(text_upper)['complexity'] == 2

        # Mixed case
        text_mixed = "Summary\n<MiRa:CoMpLeXiTy>1</MiRa:CoMpLeXiTy>"
        assert parser.parse_response(text_mixed)['complexity'] == 1

    def test_handles_multiple_complexity_tags(self):
        """CONTRACT: When multiple tags present, parser extracts first valid one."""
        parser = TagParser()
        text = """
Summary text
<mira:complexity>1</mira:complexity>
More text
<mira:complexity>3</mira:complexity>
"""

        parsed = parser.parse_response(text)

        # Should get the first one
        assert parsed['complexity'] == 1
