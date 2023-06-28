from pathlib import Path

import pytest

from pix_framework.discovery.resource_pools import discover_resource_pools
from pix_framework.input import read_csv_log
from pix_framework.log_ids import APROMORE_LOG_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.integration
@pytest.mark.parametrize('log_name', ['Resource_profiles_test.csv'])
def test_discover_resource_pools(log_name):
    log_path = assets_dir / log_name
    log_ids = APROMORE_LOG_IDS
    log = read_csv_log(log_path, log_ids)
    # Discover differentiated profiles
    pools = discover_resource_pools(log=log, log_ids=log_ids)
    # Assert discovered pools is two
    assert pools is not None
    assert len(pools) == 2
    # Assert the resources are the ones from the log
    for pool in pools:
        if len(pools[pool]) == 1:
            assert pools[pool] == ["Pucci-000001"]
        else:
            assert sorted(pools[pool]) == sorted(["Jotaro-000001", "Jolyne-000001"])
