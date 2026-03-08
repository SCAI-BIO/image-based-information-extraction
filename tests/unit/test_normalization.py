"""Tests for normalize()."""

import pytest

from covid_ndd_extraction.utils.embeddings import normalize


class TestNormalize:
    def test_lowercase(self):
        assert normalize("COVID-19") == "covid 19"

    def test_underscore_to_space(self):
        assert normalize("astrocyte_activation") == "astrocyte activation"

    def test_hyphen_to_space(self):
        assert normalize("blood-brain-barrier") == "blood brain barrier"

    def test_punctuation_stripped(self):
        # Commas, dots, exclamation marks should be removed
        assert normalize("SARS-CoV-2, infection!") == "sars cov 2 infection"

    def test_excess_whitespace_collapsed(self):
        assert normalize("  foo   bar  ") == "foo bar"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_nan_returns_empty(self):
        import pandas as pd
        assert normalize(pd.NA) == ""
        assert normalize(float("nan")) == ""

    def test_none_returns_empty(self):
        # pd.isna(None) is True
        import pandas as pd
        assert normalize(None) == ""  # type: ignore[arg-type]

    def test_mixed_separators(self):
        result = normalize("IL_6-induced inflammation")
        assert result == "il 6 induced inflammation"

    def test_already_clean(self):
        assert normalize("neuroinflammation") == "neuroinflammation"
