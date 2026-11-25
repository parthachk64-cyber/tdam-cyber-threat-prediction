from __future__ import annotations

import pytest

from tdam.config import Config
from tdam.data.dataset import build_dataset
from tdam.data.fixture import FixtureSpec, generate
from tdam.data.io import event_from_record, incident_from_record

SMALL = FixtureSpec(n_hosts=6, days=3.0, seed=3, incidents_per_host_day=1.5, outage_hosts=1)


@pytest.fixture(scope="session")
def small_records():
    return generate(SMALL)


@pytest.fixture(scope="session")
def small_dataset(small_records):
    events, incidents, coverage = small_records
    return build_dataset(
        "small",
        "synthetic",
        [event_from_record(e) for e in events],
        [incident_from_record(i) for i in incidents],
        coverage,
    )


@pytest.fixture()
def cfg():
    return Config()
