# Football Match Prediction Model

This project predicts football results with two fitted Poisson regression models:
one for home goals and one for away goals. It currently covers the Premier League,
Bundesliga, Ligue 1, Serie A, and La Liga.

The rebuild uses two data sources:

- StatsBomb Open Data supplies the older training matches.
- football-data.org supplies recent results for validation and final testing.

The earlier FBRef, Brave Search, hand-adjusted lambda, and desktop GUI files are
still in the repository for reference. They are not part of the rebuilt model.

## Current progress

Phase 1 and the modelling work in Phase 2 are complete.

- StatsBomb: 2,169 processed matches from 27 competition-seasons, ending on
  18 May 2024.
- football-data.org: 3,501 finished regular-time matches from the 2024/25 and
  2025/26 seasons, ending on 24 May 2026.
- All five leagues are included.
- The best validation settings are an 8-match rolling window and `alpha=0.1`.
- Validation 1X2 log loss is `1.0274`.
- The 1,751 supported matches in 2025/26 remain separate from model selection.

That last season has not been scored yet. Its one-time comparison against a simple
baseline belongs to Phase 3.

## Why the model now uses goals instead of xG

StatsBomb provides xG, but football-data.org provides results without xG. A single
xG feature model therefore cannot use both sources fairly.

Both sources do provide final goals. The shared model now builds the same goal-form
features from both sources. StatsBomb xG is still retained in its processed data,
so it can be used in a separate experiment later.

## How the modelling works

For every match, the feature builder looks only at matches from earlier dates. It
calculates:

- recent goals scored and conceded by both teams;
- season-to-date goals scored and conceded by both teams;
- the number of earlier matches behind each average;
- the competition in which the match was played.

If a team has no earlier match, the code uses the league's earlier goal average.
If the league also has no earlier match, that row is marked unsupported. All games
on one date are calculated before that date updates the histories, which prevents
same-day leakage.

Team IDs are kept separate for each provider. The code does not try to guess that
a StatsBomb team ID and a football-data.org team ID are the same club.

The data roles are explicit:

1. Fit candidate models on the 2,169 StatsBomb matches.
2. Choose the rolling window and regularization using football-data.org 2024/25.
3. Refit the chosen model on StatsBomb plus 2024/25.
4. Keep football-data.org 2025/26 for the final test.

Earlier results within the final season may update the form used for later matches.
This copies how predictions would work week by week, while the fitted model and its
settings remain frozen.

The two regressions produce `lambda_home` and `lambda_away`: the model's expected
number of goals for each team. The probability code then calculates every scoreline
probability from the two Poisson distributions. Adding the relevant score cells
gives home-win, draw, and away-win probabilities. The largest cell is the most
likely scoreline.

There are no hand-written favourite boosts or adjustments after prediction.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

Create an ignored `.env` file in the project root for football-data.org:

```text
FOOTBALL_DATA_API_KEY=your_own_token
```

Never commit this file or paste the token into source code.

## Commands you can run

Download or refresh StatsBomb data:

```bash
python -m football_prediction.cli update-data
```

Download the two recent football-data.org seasons:

```bash
python -m football_prediction.cli update-football-data --seasons 2024 2025
```

The downloaded JSON is cached. Running the same command again processes the local
cache without making repeat API calls unless `--refresh` is added.

Inspect both data summaries:

```bash
python -m football_prediction.cli data-status
python -m football_prediction.cli football-data-status
```

Run the real Phase 2 tuning workflow:

```bash
python -m football_prediction.cli tune-model
```

This prints the best settings, all nine validation results, and the dates and size
of the untouched final-test period. It does not report final-test performance.

Run all tests:

```bash
python -m pytest
```

## Main modelling files

- `src/football_prediction/features.py` builds leakage-safe pre-match features.
- `src/football_prediction/model.py` fits and tunes the two Poisson regressions.
- `src/football_prediction/probabilities.py` converts expected goals into exact
  score and 1X2 probabilities.
- `src/football_prediction/data/football_data.py` handles the recent API data.
- `src/football_prediction/cli.py` contains the commands shown above.

## Current limitations

- Goals are noisier than xG and may react more slowly to changes in performance.
- Home and away goal counts are treated as independent Poisson variables.
- Injuries, line-ups, transfers, and red-card information are not model features.
- StatsBomb coverage is uneven across leagues and seasons.
- Clubs are not matched across the two providers; only the learned model
  coefficients and competition categories are shared.
- Final test metrics, a competition-average baseline, saved model artifacts, and
  the new interface are later phases.

Data sources: [StatsBomb Open Data](https://github.com/statsbomb/open-data) and
[football-data.org](https://www.football-data.org/).
