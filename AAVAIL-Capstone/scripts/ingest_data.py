"""Normalize the course dataset into data/revenue.csv.

Expected source columns: InvoiceDate/date, Country/country, Revenue/revenue.
If the source contains Quantity and UnitPrice instead of Revenue, revenue is
calculated as their product. Review this mapping against your course files.
"""
import argparse
from pathlib import Path
import pandas as pd

def ingest(source, destination):
    df = pd.read_csv(source)
    lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    def find(*names):
        for name in names:
            if name in lower:
                return lower[name]
        return None
    date_col = find("date", "invoicedate", "invoice_date")
    country_col = find("country")
    revenue_col = find("revenue", "total_revenue", "sales")
    if not date_col or not country_col:
        raise ValueError("Source must contain date/InvoiceDate and Country columns.")
    if not revenue_col:
        qty, price = find("quantity"), find("unitprice", "unit_price")
        if not qty or not price:
            raise ValueError("Provide Revenue or both Quantity and UnitPrice.")
        df["revenue"] = pd.to_numeric(df[qty], errors="coerce") * pd.to_numeric(df[price], errors="coerce")
        revenue_col = "revenue"
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "country": df[country_col].astype(str),
        "revenue": pd.to_numeric(df[revenue_col], errors="coerce"),
    }).dropna()
    out = out.groupby(["date", "country"], as_index=False)["revenue"].sum()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", default="data/revenue.csv")
    args = parser.parse_args()
    result = ingest(args.source, args.output)
    print(f"Saved {len(result)} daily-country rows to {args.output}")
