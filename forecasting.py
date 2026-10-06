"""Reusable helpers for the first retail sales forecasting experiment.

Forecasts represent positive-quantity, positive-price, non-cancelled sales,
not unmet demand. Exact duplicate removal is an explicit cleaning assumption.
Unknown calendar weeks remain missing; they are never compressed away.
The experiment uses three regression lags and a six-week moving average.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def clean_transactions(raw):
    """Apply the original experiment's filters while preserving the input."""
    cancelled = raw["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    valid_sale = (
        (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
        & ~cancelled
    )
    sales = raw.loc[valid_sale].copy().sort_values("InvoiceDate")
    return sales.drop_duplicates().copy()


def build_weekly_sales(sales_data, product_code, week_ends, unknown_weeks=()):
    """Build Monday-Sunday totals on an explicitly supplied weekly calendar.

    The calendar must cover fully observed weeks in the source dataset.
    A zero means no qualifying transactions were recorded, not no demand.
    Unknown weeks must be supplied separately and are represented as NaN.
    """
    calendar = pd.DatetimeIndex(week_ends)
    if calendar.empty or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("week_ends must be a non-empty, unique, increasing calendar")
    if not (calendar.dayofweek == 6).all():
        raise ValueError("Each week must be labelled by its ending Sunday")
    if len(calendar) > 1 and not (
        calendar.to_series().diff().dropna() == pd.Timedelta(days=7)
    ).all():
        raise ValueError("Keep every consecutive calendar week, including unknown weeks")

    product_rows = sales_data.loc[
        sales_data["StockCode"].astype(str) == str(product_code)
    ].copy()
    if product_rows.empty:
        raise ValueError(f"No sales found for product {product_code}")

    weekly = (
        product_rows.resample("W-SUN", on="InvoiceDate")["Quantity"]
        .sum()
        .rename("UnitsSold")
    )
    weekly = weekly.reindex(calendar, fill_value=0).astype(float)
    weekly.index.name = "WeekEnding"
    unknown_dates = pd.to_datetime(list(unknown_weeks))
    weekly.loc[weekly.index.isin(unknown_dates)] = float("nan")
    return weekly


def walk_forward_forecasts(weekly, evaluation_dates):
    """Refit three-lag regression for each one-week-ahead prediction.

    Only earlier target observations are used for fitting. Earlier evaluation
    outcomes become available for subsequent forecasts, as in weekly operation.
    The fixed six-week average requires six observed preceding calendar weeks.
    Regression predictions are clipped at zero, matching the original experiment.
    """
    if not isinstance(weekly.index, pd.DatetimeIndex):
        raise ValueError("weekly must have a DatetimeIndex")
    if not weekly.index.is_unique or not weekly.index.is_monotonic_increasing:
        raise ValueError("weekly dates must be unique and increasing")
    if len(weekly) > 1 and not (
        weekly.index.to_series().diff().dropna() == pd.Timedelta(days=7)
    ).all():
        raise ValueError("Preserve consecutive weekly dates, including missing values")

    dates = pd.DatetimeIndex(evaluation_dates)
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("evaluation_dates must be non-empty, unique, and increasing")
    if not dates.isin(weekly.index).all():
        raise ValueError("Every evaluation date must be present in the weekly calendar")

    data = pd.DataFrame({"Target": weekly})
    features = ["Lag1", "Lag2", "Lag3"]
    for lag in [1, 2, 3]:
        data[f"Lag{lag}"] = weekly.shift(lag)
    ma6 = weekly.shift(1).rolling(window=6, min_periods=6).mean()
    records = []

    for date in dates:
        history = data.loc[data.index < date].dropna(subset=["Target"] + features)
        X_next = data.loc[[date], features]
        prediction = float("nan")
        if not history.empty and not X_next.isna().any().any():
            model = LinearRegression()
            model.fit(history[features], history["Target"])
            prediction = max(0.0, float(model.predict(X_next)[0]))
        records.append({
            "WeekEnding": date,
            "Actual": weekly.loc[date],
            "LinearRegression": prediction,
            "MA6": ma6.loc[date],
        })
    return pd.DataFrame(records).set_index("WeekEnding")


def score_forecasts(predictions, models=("LinearRegression", "MA6")):
    """Compute each model's MAE on the same complete rows and report the count."""
    paired = predictions.dropna(subset=["Actual", *models])
    if paired.empty:
        raise ValueError("No evaluation weeks have actual values and all model forecasts")
    scores = {}
    for name in models:
        scores[name] = {
            "MAE": (paired["Actual"] - paired[name]).abs().mean(),
            "WeeksScored": len(paired),
        }
    result = pd.DataFrame.from_dict(scores, orient="index")
    result.index.name = "Model"
    return result
