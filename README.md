# Retail Sales Forecasting Dashboard

A Python and Streamlit project that forecasts one product's weekly recorded sales, compares simple baselines with linear regression, and investigates forecast errors at invoice level.

The first experiment uses product **85123A — WHITE HANGING HEART T-LIGHT HOLDER** from the UCI Online Retail dataset. The dashboard also lets users explore nine other products using the same fixed forecasting methods.

## Features

- Interactive weekly sales and forecast charts.
- A product shortlist ranked by invoice count before 3 October 2011.
- Three-lag linear regression and a six-week moving-average baseline.
- Chronological validation and test evaluation with weekly updates.
- Invoice-level analysis of large sales spikes.
- Prediction CSV downloads and reusable Python functions.

## Dataset and attribution

Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).

Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.

The source contains **541,909 transaction rows and eight columns**, covering 1 December 2010 to 9 December 2011. It describes a UK-based online retailer and includes transactions from multiple countries; this experiment does not restrict the data to UK customers.

The dataset is distributed under **CC BY 4.0**. Download the original Excel file from UCI and place it at `data/Online Retail.xlsx`. The raw dataset is not required in the GitHub repository.

## Forecasting target and cleaning

The target is weekly units in **positive-quantity, positive-price, non-cancelled transactions** for a selected product. This represents recorded qualifying sales, not unmet demand or net sales after returns.

Cleaning rules:

1. Exclude negative or zero quantities, non-positive prices, and invoice identifiers beginning with C.
2. Retain one copy of each completely identical transaction row. This is a documented assumption: identical rows may represent duplicate records or legitimate repeated lines.
3. Retain missing customer IDs and descriptions because product codes, dates, and quantities are available for forecasting.

The original filtering retained **530,104 rows**. Removing **5,226 additional identical copies** left **524,878 rows**.

Weeks run Monday–Sunday and are labelled by their ending Sunday. The calendar starts with 6–12 December 2010 and ends with 28 November–4 December 2011, excluding the dataset's partial boundary weeks.

The week ending **2 January 2011** has no records anywhere in the source dataset. Its sales value is marked unknown (`NaN`) rather than treated as confirmed zero. Its date remains in the calendar. Unknown values are preserved when lag features are created.

## Evaluation design

The complete calendar contains **52 weeks**. The earlier 43 calendar weeks provide development history, with their last eight weeks used for validation; the final nine weeks form the test period.

| Period | Calendar coverage | Purpose |
|---|---|---|
| Initial fitting history | 6 December 2010–7 August 2011 | Earlier examples available before validation |
| Validation | 8 August–2 October 2011 | Compare methods and moving-average windows |
| Test | 3 October–4 December 2011 | Evaluate the frozen methods in a later period |

For each prediction date, regression is refitted using completed weeks strictly before that date. Earlier validation or test outcomes become available for subsequent weeks, matching a repeated one-week-ahead forecasting workflow. This is not a nine-week forecast made at one fixed origin.

The regression uses sales from one, two, and three weeks earlier. Negative regression predictions are clipped at zero. The moving average uses the six preceding calendar weeks and requires all six values to be observed.

Rows with unknown targets or required lag inputs are excluded from fitting after lag creation. Initially, regression has **28 usable fitting examples**. Scores compare both methods on the same available evaluation weeks.

Moving-average windows of 2, 3, 4, and 6 weeks were compared on the eight validation weeks. The six-week window had the lowest validation MAE among those baselines. Both it and the three-lag regression were frozen before test evaluation.

The product shortlist used transactions before 3 October 2011, including the validation period. Validation results are therefore exploratory; they should not be presented as an entirely untouched product-selection benchmark.

## Results from the first experiment

| Method | Validation MAE, units (8 weeks) | Test MAE, units (9 weeks) |
|---|---:|---:|
| Linear regression | 176.22 | 355.10 |
| Six-week moving average | 177.21 | 345.91 |

MAE is the average absolute difference between predicted and actual sales, measured in units. Lower is better; it is not an accuracy percentage.

Regression's validation advantage was less than one unit of MAE. The six-week baseline performed slightly better on the later test period. These small evaluation samples do not establish a consistent advantage for either method.

### Invoice-level error analysis

The week ending **6 November 2011** recorded **1,702 units across 51 invoices**. Invoice **574293** contributed **992 units (58.3%)**; the top three invoices contributed **73.3%** of the week's sales.

The models substantially underpredicted this week. The concentration supports the finding that a large invoice dominated the total, but does not establish why the order occurred or make it invalid. The large sale remains in the evaluation target.

The high-sales weeks ending 6 and 20 November account for approximately **63%** of the six-week baseline's total absolute test error.

## Project files

| File or directory | Purpose |
|---|---|
| `forecasting.py` | Cleaning, weekly aggregation, walk-forward forecasts, and paired MAE scoring |
| `retail_dashboard.py` | Streamlit interface and invoice analysis |
| `01_explore_data.ipynb` | Original experiment and learning notes |
| `02_reusable_pipeline.ipynb` | Reproducible workflow using the Python module |
| `requirements.txt` | Dependency versions from the working project environment |
| `data/Online Retail.xlsx` | Locally downloaded source dataset |
| `outputs/` | Exported metrics, predictions, and forecast chart |

The reusable module exposes `clean_transactions`, `build_weekly_sales`, `walk_forward_forecasts`, and `score_forecasts`.

## Run locally

The recorded environment uses Python 3.14. Clone the repository and run the installation and launch commands from its root directory:

```bash
git clone https://github.com/fouad-2004/retail-demand-forecasting.git
cd retail-demand-forecasting
```

Download the Excel dataset from [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), create a `data` folder, and save it as `data/Online Retail.xlsx`. The source Excel file is excluded from Git tracking.

```bash
python -m pip install -r requirements.txt
python -m streamlit run retail_dashboard.py
```

To reproduce the experiment, run `02_reusable_pipeline.ipynb` from top to bottom using that Python environment. The first notebook records the original exploration and baseline comparisons.

## Reproducibility checks

These checks were performed during development; automated test files are not included in this repository.

- The function-based weekly series and test predictions matched the original notebook outputs.
- Both notebooks were run from the beginning in the development environment and reproduced the recorded scores.
- The helper module was checked using synthetic examples for calendar boundaries, preserving source data, unknown-week handling, isolation from future observations, and paired scoring.
- Dashboard checks with synthetic data covered the missing-file screen, evaluation-period selection, product switching, and eight/nine-week scoring.
- The dashboard was run locally with the original dataset.

## Limitations

- Approximately one year of history and short validation/test periods.
- The original benchmark covers one product. Other dashboard products are exploratory uses of the fixed methods.
- No inventory availability, promotion schedule, or advance-order information.
- No proof that identical records are accidental duplicates.
- No established cause for the source-wide empty January week.
- Observed sales do not measure demand that could not be fulfilled.
- Changes informed by the inspected test period require fresh evaluation data before claiming an improvement.

## Possible next steps

Collect additional history or another evaluation dataset; investigate duplicate sensitivity; examine performance across a predefined product set; and incorporate inventory or promotion information if available before each forecast date.
