import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SOPHON_WARMUP", "0")
import duplex_harness as h  # noqa: E402


@pytest.fixture(params=h.VARIANTS)
def variant(request):
    return request.param


@pytest.fixture
def new(variant):
    return h.load(variant)


@pytest.fixture
def old(variant):
    try:
        return h.load(variant, old=True)
    except FileNotFoundError as e:
        pytest.skip(str(e))
