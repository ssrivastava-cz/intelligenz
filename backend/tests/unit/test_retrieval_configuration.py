"""Unit tests for RetrievalConfiguration/SourceRetrievalConfig — the
structured replacement for a single flat `top_k`, letting each
knowledge source independently tune its candidate pool size and final
selection size.
"""
from app.models.retrieval_configuration import RetrievalConfiguration, SourceRetrievalConfig


def test_source_retrieval_config_holds_independent_candidate_and_final_counts():
    config = SourceRetrievalConfig(candidate_chunks=20, final_chunks=5)

    assert config.candidate_chunks == 20
    assert config.final_chunks == 5


def test_retrieval_configuration_holds_one_config_per_knowledge_source():
    configuration = RetrievalConfiguration(
        workflow=SourceRetrievalConfig(candidate_chunks=20, final_chunks=5),
        historical_test_cases=SourceRetrievalConfig(candidate_chunks=20, final_chunks=8),
        historical_issues=SourceRetrievalConfig(candidate_chunks=15, final_chunks=5),
        uploaded_documents=SourceRetrievalConfig(candidate_chunks=10, final_chunks=3),
    )

    assert configuration.workflow.final_chunks == 5
    assert configuration.historical_test_cases.final_chunks == 8
    assert configuration.historical_issues.candidate_chunks == 15
    assert configuration.uploaded_documents.candidate_chunks == 10


def test_uniform_applies_the_same_count_to_every_source_as_both_candidate_and_final():
    configuration = RetrievalConfiguration.uniform(5)

    for source in (
        configuration.workflow,
        configuration.historical_test_cases,
        configuration.historical_issues,
        configuration.uploaded_documents,
    ):
        assert source.candidate_chunks == 5
        assert source.final_chunks == 5


def test_uniform_produces_independent_config_instances_per_source():
    configuration = RetrievalConfiguration.uniform(5)

    assert configuration.workflow is not configuration.historical_test_cases
