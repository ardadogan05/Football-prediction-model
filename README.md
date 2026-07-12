# Football Match Prediction

This project predicts football results with two Poisson regressions:

- one model predicts the expected home goals, `lambda_home`;
- one model predicts the expected away goals, `lambda_away`.

The project covers the Premier League, Bundesliga, Ligue 1, Serie A, and La Liga.
One pooled model is used across all five competitions. Competition is included as
a categorical input, so each league can have a different average scoring level.

## Workflow

```text
historical matches
-> rolling and season-to-date features
-> home and away Poisson regressions
-> tune rolling window and regularization
-> freeze and save the selected model
-> evaluate once on the untouched 2025/26 season
-> compare with a competition-average baseline
-> predict future fixtures
```

There are no hand-written favourite boosts, lambda multipliers, or subjective
adjustments after prediction.

## Data roles

Two data sources have different chronological roles:

1. StatsBomb Open Data supplies the older training matches.
2. football-data.org 2024/25 selects the rolling window and regularization.
3. football-data.org 2025/26 is the untouched final test season.

The final test season must not influence the feature list, model structure,
rolling-window choice, alpha choice, or baseline design. Earlier results within
2025/26 may update features for later test matches. This represents making a new
prediction each match week while keeping the fitted model frozen.

The synchronization, API, caching, and parsing code is kept separately under
`src/football_prediction/data/`. The educational modelling workflow does not hide
inside that data code.

## Features

For each match, the feature builder calculates:

- rolling goals scored and conceded by both teams;
- season-to-date goals scored and conceded by both teams;
- the competition.

The rolling window controls how many recent matches represent current form. A
window of 3 reacts quickly, while a window of 8 is steadier.

Features must be shifted: the goals from the match being predicted cannot be used
to predict that same match. The code therefore calculates every match on a date
before adding any results from that date to team history. This also prevents one
same-date match from leaking into another.

Both teams need at least three earlier matches in the current season. Earlier rows
are marked unsupported and excluded instead of being filled by a complicated
fallback hierarchy.

## Poisson regressions

The two models use a small scikit-learn pipeline:

1. `StandardScaler` scales the eight numerical features.
2. `OneHotEncoder` converts competition IDs into categorical columns.
3. `PoissonRegressor` fits expected goal counts.

The regularization value `alpha` controls how strongly coefficients are pulled
toward zero. This helps prevent the model from fitting noise. The tuning grid is:

```text
rolling window: 3, 5, 8
alpha:          0.01, 0.1, 1.0
```

Every one of the nine validation results is printed. Once the best settings are
selected, the model is refitted on the older training data plus 2024/25 and saved.

## Final selected model and test result

The validation grid selected:

```text
rolling window: 3
alpha: 0.1
validation log loss: 1.0119
```

The frozen model was then evaluated once on 1,606 supported matches from 2025/26,
covering 12 September 2025 through 24 May 2026.

| Metric | Fitted model | Competition-average baseline |
|---|---:|---:|
| Multiclass log loss | 1.0151 | 1.0742 |
| Multiclass Brier score | 0.6073 | 0.6498 |
| 1X2 accuracy | 50.25% | 43.59% |
| Home-goal MAE | 0.9682 | 1.0258 |
| Away-goal MAE | 0.8512 | 0.8999 |
| Combined goal MAE | 0.9097 | 0.9628 |

The fitted model beat the baseline on every reported metric. These test results
were recorded after the model settings were frozen; the model was not retuned after
viewing them.

## Result probabilities

Each fitted model produces a Poisson lambda, which is its expected number of goals.
The probability code calculates all scorelines from 0-0 through 10-10. Adding the
relevant cells gives:

- home-win probability;
- draw probability;
- away-win probability.

The cells are normalized to account for the tiny probability above ten goals. The
largest cell is the most likely scoreline. The calculation is deterministic.

## Baseline and final backtest

The baseline is deliberately simple. Before each match it calculates the earlier
competition average for:

- goals scored by home teams;
- goals scored by away teams.

The current match and later matches are never included. The fitted model and
baseline use the same Poisson probability function and exactly the same test rows.

The final report contains:

- multiclass log loss;
- multiclass Brier score;
- 1X2 accuracy;
- home-goal MAE;
- away-goal MAE;
- combined goal MAE.

Lower log loss, Brier score, and MAE are better. Higher accuracy is better. The
metrics JSON states honestly whether the fitted model beats the baseline on log
loss.

## Installation

Python 3.10 or newer is required.

```powershell
python -m pip install -e ".[dev]"
```

For football-data.org downloads, create an ignored `.env` file in the project root:

```text
FOOTBALL_DATA_API_KEY=your_own_token
```

Never commit or print this token.

## Commands

Download or refresh StatsBomb data:

```powershell
python -m football_prediction.cli update-data
```

Download football-data.org 2024/25 and 2025/26:

```powershell
python -m football_prediction.cli update-football-data --seasons 2024 2025
```

Inspect the two data manifests:

```powershell
python -m football_prediction.cli data-status
python -m football_prediction.cli football-data-status
```

Tune all nine configurations, refit, and save the selected model:

```powershell
python -m football_prediction.cli tune-model
```

`train` is an alias for the same operation. Run one command, not both:

```powershell
python -m football_prediction.cli train
```

Evaluate the saved model once on 2025/26:

```powershell
python -m football_prediction.cli backtest
```

The backtest refuses to overwrite existing reports. This discourages repeatedly
checking the test season and then changing the model around its result.

Predict one future fixture:

```powershell
python -m football_prediction.cli predict `
  --home-team "Arsenal FC" `
  --away-team "Chelsea FC" `
  --competition "Premier League"
```

Run all tests:

```powershell
python -m pytest
```

## Output files

Only three final artifacts are written:

```text
models/model.pkl
reports/backtest_predictions.csv
reports/metrics.json
```

The saved model dictionary contains the home model, away model, selected window,
selected alpha, feature columns, training cutoff date, and supported competitions.

## Main files

- `features.py`: builds shifted rolling and season features. Read the date loop and
  notice that history updates happen only after a date's features are complete.
- `model.py`: defines the visible scikit-learn pipeline, fits both regressions, tries
  the 3x3 grid, and refits the selected model without using the test targets.
- `probabilities.py`: converts the two lambdas into the 0-10 score matrix and 1X2
  probabilities.
- `backtest.py`: creates the causal competition baseline, match-level predictions,
  metrics, and the two report files.
- `prediction.py`: saves and loads the model dictionary and predicts one future
  fixture from the latest available team histories.
- `cli.py`: connects the data, training, backtest, and prediction commands.

## Limitations

- Goals are noisy and may react slowly to changes in team performance.
- Home and away goals are treated as independent Poisson variables.
- Injuries, line-ups, transfers, red cards, and bookmaker odds are not features.
- StatsBomb coverage is uneven between competitions and seasons.
- The strict history rule leaves no supported older Bundesliga training rows in
  the partial StatsBomb sample. Bundesliga keeps a fixed competition column and
  gains fitted league information when 2024/25 joins the final refit.
- Team IDs are kept separate between data providers rather than guessed across
  providers.
- A team needs three earlier matches in the current season before prediction.
- The future prediction function expects exact team and competition names from the
  recent processed data.
- The Streamlit interface is still a later, separate step.

Data sources: [StatsBomb Open Data](https://github.com/statsbomb/open-data) and
[football-data.org](https://www.football-data.org/).
