from __future__ import annotations

from typing import Dict, List

FUNDAMENTAL_PARQUET_COLUMNS: List[str] = [
    "asset_id",
    "report_type",
    "fiscal_date_end",
    "reported_date",
    "revenue",
    "net_income",
    "operating_cash_flow",
    "total_shareholder_equity",
    "total_liabilities",
    "source",
]

# pandas dtypes for stable parquet writing
# NOTE: fiscal_date_end/reported_date handled separately as datetime64[ns, UTC]
FUNDAMENTAL_PARQUET_DTYPES: Dict[str, str] = {
    "asset_id": "string",
    "report_type": "string",
    "revenue": "float64",
    "net_income": "float64",
    "operating_cash_flow": "float64",
    "total_shareholder_equity": "float64",
    "total_liabilities": "float64",
    "source": "string",
}
