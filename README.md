# Demand Inventory Optimizer

> A Python-first, end-to-end data product for globally benchmarkable demand forecasting, inventory planning, and stockout simulation and optimization.

## Project status

**Repository bootstrap / project definition.** Implementation has not started yet. This README is the initial project contract and will be updated as the system becomes real. The public dataset is selected during the source audit, before any code.

## Problem

Inventory decisions are a constant trade-off: too little stock means stockouts and lost sales; too much means holding costs and waste. Most demand forecasts look better than they are because they are evaluated with leakage or on unrealistic splits, and simple baselines are never reported. The result is inventory policies tuned to an over-optimistic forecast.

This project will build a reproducible, benchmarkable forecasting and inventory-optimization pipeline on a public dataset, with honest temporal evaluation and the business trade-offs made visible.

## Product goal

Build a small production-style data platform that answers questions such as:

- How well do simple baselines forecast demand, and how much better is any advanced model?
- What inventory policy (reorder point, order-up-to level) minimizes total cost at a chosen service level?
- How do stockouts and holding costs trade off as the service level changes?
- Where does forecast error actually hurt the business?

The product must make the business trade-off explicit: service level, stockout cost, and holding cost are inputs to the decision, not hidden constants.

## Intended users

- Inventory planners and operations analysts exploring policy trade-offs.
- Students and candidates demonstrating forecasting and optimization skills.
- Anyone evaluating benchmarkable forecasting methods on public data.

## What the system will do

1. **Select a public dataset during the source audit** (candidates: retail demand competitions such as M5 Accuracy and Store Sales/Favorita, or an equivalent benchmarkable set — the final choice is decided by the audit).
2. **Ingest and validate** the data with a documented schema.
3. **Split temporally** with walk-forward, out-of-sample evaluation and no leakage (per product/store group and final time horizon).
4. **Establish baselines** (naive, seasonal naive, moving average, exponential smoothing) before any advanced model.
5. **Report forecast error** (MAE, RMSE, WMAPE, bias, interval coverage).
6. **Simulate inventory policies** (for example, reorder-point/order-up-to) against realized demand with explicit stockout and holding costs.
7. **Optimize** policy parameters for a stated service level or cost objective.
8. **Expose** results through an interactive scenario demo.
9. **Report** assumptions, limitations, and failure behavior.

## Demo vision

The final demo should be understandable in under one minute and show more than a happy-path chart:

- A forecast view with actuals, baseline, and advanced model on the hold-out horizon.
- An error-comparison view: which method wins, and by how much, on which products.
- A scenario simulator with sliders for service level, lead time, and cost assumptions.
- A stockout-vs-cost frontier showing the trade-off at a glance.
- A data-quality page showing the latest successful ingestion, source status, and known limitations.
- An empty-result or failure state that explains what happened instead of silently failing.

The project will have a public deployment when hosting constraints are understood. Until then, a reproducible local demo and a short recorded walkthrough will be required.

## Scope

### Minimum viable product

- One audited public dataset with a documented schema.
- Reproducible ingestion using saved fixtures for offline development.
- Walk-forward temporal splits with a hold-out horizon.
- Simple baselines with standard forecast-error metrics.
- One inventory policy simulator with explicit stockout and holding costs.
- An interactive Streamlit scenario demo.
- Automated tests for splitting, metrics, and simulation logic.
- A README that documents data lineage, metrics, limitations, and setup.

### Version 2

- A measured advanced model (for example, gradient boosting or a probabilistic method) only after baselines are reported.
- Interval and quantile forecasts with coverage evaluation.
- Policy optimization under constraints (budget, service level).
- Additional benchmark datasets and cross-dataset comparison.
- FastAPI service and Dockerized local deployment.

### Explicitly out of scope

- Private or proprietary demand data.
- Claims of universal superiority for any single method.
- Optimizing for costs that are not stated and defended.
- Real procurement system integration.
- Building an opaque ML forecast without a baseline and evaluation.

## Proposed technical stack

| Layer | Initial choice | Purpose |
| --- | --- | --- |
| Language | Python | Core implementation and analysis |
| Environment | `uv` | Reproducible dependency and environment management |
| Quality | Ruff, pytest, Pandera | Formatting/linting, tests, and data contracts |
| Ingestion | Python source adapters, `httpx` | Collect data without coupling the domain to one source |
| Raw storage | JSON and Parquet fixtures | Preserve source evidence and enable offline development |
| Analytics | pandas and DuckDB | Cleaning, SQL analysis, and local analytical queries |
| Baselines | statsmodels | Naive, seasonal, and exponential-smoothing baselines |
| ML extension | scikit-learn, later | Measured forecast improvements after the baseline |
| Optimization | scipy, OR-Tools later | Policy parameter search under constraints |
| Demo | Streamlit for the MVP | Fast, readable, interactive public demonstration |
| API | FastAPI and Pydantic, later | Typed service boundary for the application |
| Delivery | Docker and GitHub Actions, later | Reproducible execution and automated checks |

Technology choices are staged deliberately: baselines and honest splits come before models, and the business trade-off is defined before optimization.

## Skills demonstrated

### Data science

- Problem framing and metric definition (WMAPE, bias, coverage).
- Temporal validation: walk-forward splits and leakage prevention.
- Baseline-first model evaluation.
- Uncertainty and interval forecasting (in version 2).

### Data engineering

- Source adapters and resilient ingestion.
- Raw, cleaned, and analytical data layers.
- Schema design and data contracts.
- Failure handling, freshness, and idempotency.
- SQL analytics with DuckDB.

### AI and ML

- Simple-baseline discipline: advanced models must justify their cost.
- Probabilistic forecasting as a measured extension.
- Honest hold-out reporting per product and store group.

### Optimization and business analysis

- Inventory policy simulation (reorder point, order-up-to).
- Explicit cost models: stockouts, holding, and service level.
- Trade-off frontiers instead of single-number answers.

### Software and full stack

- Typed Python modules and service boundaries.
- Interactive user experience through Streamlit.
- Tests, documentation, configuration, logging, and reproducible execution.
- CI and deployment readiness.

### Professional communication

- Clear problem statement and stakeholder value.
- Metrics, trade-offs, limitations, and failure analysis.
- A short demo video and a LinkedIn post focused on the result rather than the tool list.

## Evaluation plan

The project will report measurable evidence, including:

- Baseline forecast error (MAE, RMSE, WMAPE, bias) on the hold-out horizon.
- Improvement of any advanced model over the best baseline, using the same splits.
- Simulated stockout rate and total cost at stated service levels.
- Cost frontier: total cost across service levels (for example, 90/95/98 percent).
- Reproducibility from a clean environment using fixtures.
- At least one intentionally induced failure (for example, missing sales days) and the system's response.

A strong result is an honest, benchmarkable comparison with visible business trade-offs — not a single model that claims to be best everywhere.

## Data and ethics principles

- Use only public datasets whose licenses permit analysis, reproduction, and publication.
- Attribute dataset sources and respect their terms.
- Document the dataset's real-world limitations; do not present it as Ecuador-specific unless the audit says otherwise.
- State all cost and service-level assumptions explicitly.
- Keep a fixture dataset so the project remains runnable if a source changes or disappears.
- Never present simulated results as actual business recommendations for a specific company.

## Planned repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── inventory_optimizer/
│       ├── ingestion/
│       ├── forecasting/
│       ├── simulation/
│       ├── optimization/
│       └── config.py
├── tests/
├── data/
│   └── sample/
├── notebooks/
├── app/
├── docs/
└── .github/
    └── workflows/
```

Large raw datasets, credentials, and local runtime state must not be committed.

## Roadmap

- [ ] Audit candidate public datasets and their licenses.
- [ ] Select the dataset and define the canonical schema.
- [ ] Create fixtures and the first ingestion adapter.
- [ ] Implement temporal splits and the validation harness.
- [ ] Implement and evaluate simple baselines.
- [ ] Implement the inventory policy simulator and cost model.
- [ ] Add data-quality checks and failure-path tests.
- [ ] Build the Streamlit scenario demo.
- [ ] Add an advanced model only after baselines are reported.
- [ ] Add policy optimization under constraints.
- [ ] Deploy the demo or publish a recorded walkthrough.
- [ ] Write the final technical case study and LinkedIn publication.

## Definition of done

The project is ready for portfolio publication when:

- A new contributor can run it from documented instructions.
- The demo answers a clear inventory question with traceable evidence.
- Forecasts are evaluated on temporal splits against simple baselines.
- The business trade-off (service level versus cost) is visible and adjustable.
- Tests and CI pass on the reviewed revision.
- The README explains the architecture, trade-offs, limitations, and next steps.
- A recruiter can understand the problem, result, and demo link in under two minutes.

## License

To be decided after the dataset and dependency licenses are audited.
