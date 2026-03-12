"""Tests for parse_mechanisms_and_triples()."""

import pytest

from covid_ndd_extraction.utils.parsing import parse_mechanisms_and_triples


VALID_OUTPUT = """\
Pathophysiological Process: Astrocyte_Activation
Triples:
SARS-CoV-2_infection|triggers|astrocyte_activation
cytokine_storm|promotes|neuroinflammation

Pathophysiological Process: BBB_Disruption
Triples:
SARS-CoV-2|disrupts|blood-brain_barrier
"""

MALFORMED_TRIPLES = """\
Pathophysiological Process: Some_Process
Triples:
subject_only
A|B|C|D
"""

REFUSAL_TEXT = "I cannot process this image due to content policy restrictions."


class TestParseMechanismsAndTriples:
    def test_valid_two_blocks(self):
        blocks = parse_mechanisms_and_triples(VALID_OUTPUT)
        assert len(blocks) == 2

    def test_first_block_mechanism(self):
        blocks = parse_mechanisms_and_triples(VALID_OUTPUT)
        assert blocks[0]["mechanism"] == "Astrocyte_Activation"

    def test_first_block_triples(self):
        blocks = parse_mechanisms_and_triples(VALID_OUTPUT)
        triples = blocks[0]["triples"]
        assert len(triples) == 2
        assert triples[0] == ("SARS-CoV-2_infection", "triggers", "astrocyte_activation")

    def test_second_block_mechanism(self):
        blocks = parse_mechanisms_and_triples(VALID_OUTPUT)
        assert blocks[1]["mechanism"] == "BBB_Disruption"

    def test_malformed_triples_ignored(self):
        """Lines that don't match `s|p|o` exactly should be silently skipped."""
        blocks = parse_mechanisms_and_triples(MALFORMED_TRIPLES)
        assert len(blocks) == 1
        # "subject_only" and "A|B|C|D" are both malformed
        assert blocks[0]["triples"] == []

    def test_empty_string(self):
        blocks = parse_mechanisms_and_triples("")
        assert blocks == []

    def test_none_treated_as_empty(self):
        blocks = parse_mechanisms_and_triples(None)  # type: ignore[arg-type]
        assert blocks == []

    def test_refusal_text_yields_no_blocks(self):
        """Refusal text has no mechanism headers so should return empty."""
        blocks = parse_mechanisms_and_triples(REFUSAL_TEXT)
        assert blocks == []

    def test_whitespace_stripped_from_mechanism(self):
        text = "Pathophysiological Process:  Foo_Bar  \nTriples:\nA|B|C\n"
        blocks = parse_mechanisms_and_triples(text)
        assert blocks[0]["mechanism"] == "Foo_Bar"

    def test_triple_whitespace_stripped(self):
        text = "Pathophysiological Process: X\nTriples:\n  alpha | beta | gamma  \n"
        blocks = parse_mechanisms_and_triples(text)
        assert blocks[0]["triples"] == [("alpha", "beta", "gamma")]
