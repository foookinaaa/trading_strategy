# Trading strategy
### Problem
Predict trend in the next 30 days will go up or not for S&P500 tickers that was in the index for more than 20 years and appeared in past 5 years

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
- Train data: 2010-01-04 - 2023-12-31; time split validation on 5 folds
- Test data: 2024-01-01 - 2025-01-08
for is_positive_growth_30d_future 30-45% -> 0 and 55-70% -> 1 \
file: notebooks/eda.ipynb \

##### Baseline
Simple strategy: buy when SMA10 is lower and intersects SMA20, sell when they intersect again in the upper side \
Realisation through growing_moving_average -> is_positive_growth_30d_future
 * accuracy: 0.504
 * roc_auc: 0.495
 * precision: 0.57
 * recall: 0.555

##### Desicion tree
MAX_DEPTH = 10 + drop unimportant features
* accuracy: 0.527
* roc_auc: 0.508
* precision: 0.581
* recall: 0.631

##### Random Forest
MAX_DEPTH = 10 + drop unimportant features
* accuracy: 0.573
* roc_auc: 0.506
* precision: 0.578
* recall: 0.956

Let's continue with random forest for now

### Trading Simulation
Simple simulation with maximising CAGR in 4 years
