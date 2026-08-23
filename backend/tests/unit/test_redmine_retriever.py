"""RedmineRetriever is an interface-only stub — these tests just pin
down that behavior so it can't silently start "working" (e.g. returning
`[]`, which would be mistaken for "no results" instead of "not
implemented") without a deliberate code change.
"""
import pytest

from app.retrievers.base import Retriever
from app.retrievers.redmine_retriever import RedmineRetriever


def test_redmine_retriever_is_a_retriever():
    assert issubclass(RedmineRetriever, Retriever)


def test_retrieve_raises_not_implemented_error():
    retriever = RedmineRetriever()

    with pytest.raises(NotImplementedError):
        retriever.retrieve([1.0, 0.0], top_k=5)
