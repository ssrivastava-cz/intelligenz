from app.utils.text import looks_like_heading, title_from_filename


def test_title_from_filename_humanizes_separators():
    assert title_from_filename("release_notes-v2.pdf") == "Release Notes V2"


def test_title_from_filename_falls_back_to_original_when_cleaned_stem_is_empty():
    assert title_from_filename("___.pdf") == "___.pdf"


def test_all_caps_line_looks_like_heading():
    assert looks_like_heading("ROLE PERMISSIONS") is True


def test_colon_terminated_line_looks_like_heading():
    assert looks_like_heading("Validation Rules:") is True


def test_title_case_line_looks_like_heading():
    assert looks_like_heading("Role Permissions") is True
    assert looks_like_heading("Contact Creation") is True


def test_ordinary_sentence_does_not_look_like_heading():
    assert looks_like_heading("This is a normal sentence.") is False


def test_sentence_fragment_ending_in_comma_does_not_look_like_heading():
    assert looks_like_heading("Please review the following,") is False


def test_single_capitalized_word_without_punctuation_is_not_a_heading():
    """A lone capitalized word is too weak a signal on its own — needs
    ALL CAPS, a trailing colon, or at least two Title Case words."""
    assert looks_like_heading("Overview") is False


def test_lowercase_single_word_is_not_a_heading():
    assert looks_like_heading("overview") is False


def test_sentence_with_only_first_word_capitalized_is_not_title_case():
    assert looks_like_heading("Thanks for waiting on this one today") is False


def test_empty_or_blank_line_is_not_a_heading():
    assert looks_like_heading("") is False
    assert looks_like_heading("   ") is False


def test_overly_long_line_is_not_a_heading():
    long_line = "Role Permissions " * 10  # way over 80 chars, still title-case-ish
    assert looks_like_heading(long_line) is False
