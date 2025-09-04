# Trading strategy
### Problem

### Data Sources
- Take S&P500 tickers
- Take data for past 10 years (2010-01-01 - Now)
- take only tickers that was in the index for more than 20 years
- add tickers that appeared in past 5 years
- add macro indicators
- add transformations with prices

Total: 300 tickers S&P500 (with longest history + newest) -> 1_101_534 rows
file: notebooks/collect_data.ipynb

### Data Transformations
- 153 columns - remove correlated features + add dummy features
- 103 numeric features and 39 binary features

let's first predict is_positive_growth_30d_future and growth_future_30d \
file: notebooks/eda.ipynb

### Modeling
- Train data: 2010-01-04 - ... ; time split validation on 5 folds
- Test data: 2024-01-01 - 2025-01-08

### Trading Simulation

### Automation
