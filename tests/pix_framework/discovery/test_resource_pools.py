from pathlib import Path

import pytest
from pix_framework.discovery.resource_pools import discover_resource_pools
from pix_framework.input import read_csv_log
from pix_framework.log_ids import DEFAULT_XES_IDS

assets_dir = Path(__file__).parent.parent / "assets"


@pytest.mark.smoke
def test_discover_resource_pools():
    log_path = assets_dir / "PurchasingExample.csv.gz"
    log_ids = DEFAULT_XES_IDS
    log = read_csv_log(log_path, log_ids)

    result = discover_resource_pools(log, log_ids)

    assert result is not None
    assert len(result) >= 4
