from typing import Any

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            ("authorization", "DUMMY_API_KEY"),
        ],
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
    }
