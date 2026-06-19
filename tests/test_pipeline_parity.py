"""
Pipeline-output parity guard.

The original P0 incident was the Listed chain drifting a month behind the Sold
chain (they were hand-run independently). run_pipeline now runs both in one
ordered pass, but this test is a cheap structural guard that catches the drift
directly from the outputs — and adds the pipeline-output coverage the rest of
the suite lacks.

Skips automatically when the (gitignored) processed data isn't present, e.g. in
CI — so it runs as a real check locally after a pipeline run, without breaking
the data-free CI suite.
"""
import os

import pandas as pd
import pytest

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "processed", "tableau")
SOLD_MONTHLY = os.path.join(TAB, "monthly_market.csv")
LISTED_MONTHLY = os.path.join(TAB, "monthly_new_listings.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SOLD_MONTHLY) and os.path.exists(LISTED_MONTHLY)),
    reason="processed monthly data not present (e.g. CI without the gitignored data)",
)


def test_sold_and_listed_cover_the_same_months():
    """Both chains must span the same month range — catches one chain drifting
    behind the other."""
    sold = pd.read_csv(SOLD_MONTHLY)
    listed = pd.read_csv(LISTED_MONTHLY)
    assert sold["yr_mo"].max() == listed["yr_mo"].max(), (
        f"latest month differs — sold={sold['yr_mo'].max()} "
        f"listed={listed['yr_mo'].max()} (a chain drifted)")
    assert sold["yr_mo"].min() == listed["yr_mo"].min(), (
        f"start month differs — sold={sold['yr_mo'].min()} "
        f"listed={listed['yr_mo'].min()}")
