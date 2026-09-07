import pytest

from store.db import Store


@pytest.fixture()
def store() -> Store:
    s = Store(db_path=":memory:")
    s.seed_from_fixtures()
    return s
