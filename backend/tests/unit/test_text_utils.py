from app.utils.text import is_blank_row, is_blank_value, looks_like_heading, title_from_filename


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


# --- data-like lines must not be misclassified as headings ---
# Confirmed against the real Source of Truth PDFs, which contain
# exactly these lines today, misclassified before this fix.


def test_id_value_pair_is_not_a_heading():
    assert looks_like_heading("Patient ID P10001") is False
    assert looks_like_heading("Member ID M10001") is False


def test_hyphenated_code_value_is_not_a_heading():
    assert looks_like_heading("Plan ID PLAN-A") is False
    assert looks_like_heading("Site ID SITE-01") is False


def test_short_acronym_inside_a_genuine_heading_still_works():
    """The stricter per-word check must not break a real heading that
    legitimately contains a short all-caps acronym."""
    assert looks_like_heading("API Configuration") is True
    assert looks_like_heading("Patient ID Overview") is True


def test_numeric_only_token_alone_does_not_disqualify_a_heading():
    """A pure-numeric token (no leading letter at all) is ignored by the
    existing word filter, exactly like before this fix — only a
    letter-starting word containing digits (an ID/code) is rejected."""
    assert looks_like_heading("Phase 2 Rollout") is True


# --- is_blank_value / is_blank_row ---


def test_is_blank_value_treats_none_and_whitespace_as_blank():
    assert is_blank_value(None) is True
    assert is_blank_value("") is True
    assert is_blank_value("   ") is True


def test_is_blank_value_treats_meaningful_falsy_values_as_not_blank():
    assert is_blank_value(0) is False
    assert is_blank_value(0.0) is False
    assert is_blank_value(False) is False
    assert is_blank_value("0") is False


def test_is_blank_value_treats_real_content_as_not_blank():
    assert is_blank_value("some value") is False
    assert is_blank_value("Patient ID") is False


def test_is_blank_row_true_only_when_every_value_is_blank():
    assert is_blank_row(["", "", "", ""]) is True
    assert is_blank_row(["   ", " ", "", "   "]) is True
    assert is_blank_row([None, None]) is True
    assert is_blank_row([]) is True


def test_is_blank_row_false_when_any_value_is_meaningful():
    assert is_blank_row(["", "", "some value", ""]) is False
    assert is_blank_row(["Patient ID", "", "", ""]) is False
    assert is_blank_row([0, "", ""]) is False
    assert is_blank_row([False, "", ""]) is False
