# Application Interface Contract

## Interface type

The active product is a **Streamlit web application**, not a REST API. It has no FastAPI endpoints and no programmatic prediction endpoint.

## Runtime routes

| Route | Purpose | Expected response |
|---|---|---|
| `/` | Streamlit dashboard shell; the browser establishes the Streamlit session to run the UI. | HTTP 200 |
| `/_stcore/health` | Streamlit server health check. | `ok` |

The health route verifies server availability only; it is not a cancer-prediction API.

## Prediction input contract

The dashboard accepts one numeric value for each WDBC feature in this order:

1. `radius_mean` through `fractal_dimension_mean`
2. `radius_se` through `fractal_dimension_se`
3. `radius_worst` through `fractal_dimension_worst`

The exact canonical list is `WDBC_FEATURE_COLUMNS` in `cdss/src/pipeline/data_loader.py`. Inputs are assembled into a one-row DataFrame and passed directly to the saved sklearn pipeline.

## Prediction output contract

The UI displays:

- A class label: `Benign` or `Malignant`.
- The model's malignant probability.
- The complementary benign probability.

There are no response payloads, authentication rules, database records, or external service contracts.

## Future API work

Adding a REST or MCP interface is out of scope. It requires explicit user approval and updates to `PRD.md`, `DATABASE.md`, `TESTING.md`, `PLAN.md`, and this file before implementation.
