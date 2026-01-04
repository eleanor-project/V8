import pytest

from api.schemas import DeliberationRequest


def test_deliberation_request_sanitizes_input():
    req = DeliberationRequest(input=" hello\x00world ")
    assert req.input == "helloworld"


def test_deliberation_request_context_depth_validation():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
    with pytest.raises(ValueError):
        DeliberationRequest(input="hi", context=deep)

    shallow = {"a": {"b": {"c": 1}}}
    req = DeliberationRequest(input="hi", context=shallow)
    assert req.context["a"]["b"]["c"] == 1
