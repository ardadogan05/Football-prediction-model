import json
from pathlib import Path

import pandas as pd
import streamlit as st

from football_prediction.config import ProjectPaths
from football_prediction.data.football_data import load_football_matches
from football_prediction.prediction import load_model, predict_match


PROJECT_ROOT = Path(__file__).resolve().parent
PATHS = ProjectPaths.from_root(PROJECT_ROOT)

LEAGUE_NAMES = {
    "Premier League": "Premier League",
    "Bundesliga": "1. Bundesliga",
    "Ligue 1": "Ligue 1",
    "Serie A": "Serie A",
    "La Liga": "La Liga",
}

st.set_page_config(
    page_title="Touchline AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown(
    """
    <style>
    :root {
        --navy: #07111f;
        --panel: rgba(16, 31, 50, 0.88);
        --panel-light: rgba(25, 45, 70, 0.72);
        --text: #f5f8fc;
        --muted: #9fb0c5;
        --green: #34d399;
        --blue: #60a5fa;
        --amber: #fbbf24;
        --red: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.18), transparent 34%),
            radial-gradient(circle at 95% 18%, rgba(16, 185, 129, 0.12), transparent 28%),
            linear-gradient(145deg, #050b14 0%, #091523 52%, #07101d 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0b1929 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.13);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, label, [data-testid="stMetricLabel"] {
        color: var(--text) !important;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #2563eb, #34d399);
        box-shadow: 0 12px 36px rgba(37, 99, 235, 0.28);
        font-size: 1.35rem;
    }

    .brand-name {
        font-size: 1.15rem;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    .brand-subtitle, .eyebrow, .muted {
        color: var(--muted) !important;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: clamp(1.9rem, 4vw, 3.2rem);
        line-height: 1.03;
        letter-spacing: -0.055em;
        font-weight: 820;
        margin: 0 0 0.6rem 0;
        max-width: 850px;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 0.96rem;
        max-width: 720px;
        margin-bottom: 1.1rem;
    }

    .glass-card, [data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--panel), var(--panel-light));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(14px);
    }

    [data-testid="stMetric"] {
        min-height: 92px;
        padding: 0.9rem 1rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 760;
        letter-spacing: -0.04em;
    }

    .team-name {
        font-size: 1rem;
        font-weight: 720;
        text-align: center;
        margin-top: 0.4rem;
    }

    .versus {
        display: grid;
        place-items: center;
        min-height: 105px;
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.2em;
        font-weight: 750;
    }

    .score-hero {
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.26), rgba(16, 185, 129, 0.18));
        border: 1px solid rgba(96, 165, 250, 0.25);
    }

    .score-value {
        font-size: 2.7rem;
        line-height: 1;
        font-weight: 820;
        letter-spacing: -0.06em;
    }

    .score-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.65rem;
    }

    .probability-card {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        background: rgba(15, 29, 47, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.14);
        min-height: 120px;
    }

    .probability-label {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
    }

    .probability-value {
        font-size: 1.75rem;
        font-weight: 790;
        letter-spacing: -0.05em;
        margin: 0.25rem 0;
    }

    .odds-value {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .bar-track {
        height: 6px;
        width: 100%;
        background: rgba(148, 163, 184, 0.14);
        border-radius: 100px;
        overflow: hidden;
        margin-top: 0.85rem;
    }

    .bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #2563eb, #34d399);
    }

    .accuracy-card {
        padding: 1.25rem;
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(37, 99, 235, 0.22), rgba(16, 185, 129, 0.13));
        border: 1px solid rgba(52, 211, 153, 0.24);
        min-height: 160px;
    }

    .accuracy-value {
        font-size: 3.2rem;
        font-weight: 820;
        letter-spacing: -0.07em;
        color: #ecfdf5;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        background: rgba(52, 211, 153, 0.13);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.22);
        font-size: 0.76rem;
        font-weight: 700;
    }

    .note-card {
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: var(--muted);
        background: rgba(15, 29, 47, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.11);
        font-size: 0.88rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 13px;
        min-height: 3.05rem;
        border: 0;
        color: white;
        font-weight: 720;
        background: linear-gradient(100deg, #2563eb 0%, #0ea5e9 55%, #10b981 115%);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.23);
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        color: white;
        border: 0;
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.32);
    }

    [data-baseweb="select"] > div,
    [data-testid="stDataFrame"] {
        border-radius: 13px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_matches():
    return load_football_matches(PATHS.football_data_matches_file)


@st.cache_resource
def load_saved_model():
    return load_model(PATHS.model_file)


@st.cache_data
def load_metrics():
    return json.loads(PATHS.metrics_file.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions():
    predictions = pd.read_csv(PATHS.backtest_predictions_file)
    predictions["match_date"] = pd.to_datetime(predictions["match_date"])
    return predictions


@st.cache_data
def load_team_crests():
    team_crests = {}
    raw_directory = PATHS.football_data_raw_directory

    for path in raw_directory.glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for match in payload.get("matches", []):
            for side in ["homeTeam", "awayTeam"]:
                team = match.get(side, {})
                if team.get("name") and team.get("crest"):
                    team_crests[team["name"]] = team["crest"]

    return team_crests


def implied_odds(probability):
    return 1 / probability


def display_logo(url, fallback="⚽", width=92):
    if url:
        st.image(url, width=width)
    else:
        st.markdown(f"<div style='font-size:3rem;text-align:center'>{fallback}</div>", unsafe_allow_html=True)


def probability_card(label, probability):
    percentage = probability * 100
    st.markdown(
        f"""
        <div class="probability-card">
            <div class="probability-label">{label}</div>
            <div class="probability-value">{percentage:.1f}%</div>
            <div class="odds-value">Model-implied odds&nbsp; {implied_odds(probability):.2f}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{percentage:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_header(eyebrow, title, copy):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def prediction_page(matches, model, team_crests):
    app_header(
        "Match intelligence",
        "Turn form into a match forecast.",
        "Choose a fixture to see expected goals, 1X2 probabilities, the most likely score and transparent model-implied odds.",
    )

    display_league = st.selectbox("Competition", list(LEAGUE_NAMES))
    competition = LEAGUE_NAMES[display_league]

    latest_season = matches.loc[
        matches["competition_name"] == competition, "season_name"
    ].iloc[-1]
    current_matches = matches.loc[
        (matches["competition_name"] == competition)
        & (matches["season_name"] == latest_season)
    ]
    teams = sorted(
        set(current_matches["home_team_name"])
        | set(current_matches["away_team_name"])
    )

    default_home = 0
    default_away = 1 if len(teams) > 1 else 0
    if competition == "Premier League":
        if "Arsenal FC" in teams:
            default_home = teams.index("Arsenal FC")
        if "Chelsea FC" in teams:
            default_away = teams.index("Chelsea FC")

    home_column, versus_column, away_column = st.columns([1, 0.25, 1])
    with home_column:
        home_team = st.selectbox("Home team", teams, index=default_home)
        display_logo(team_crests.get(home_team), width=68)
        st.markdown(f"<div class='team-name'>{home_team}</div>", unsafe_allow_html=True)
    with versus_column:
        st.markdown("<div class='versus'>VS</div>", unsafe_allow_html=True)
    with away_column:
        away_options = [team for team in teams if team != home_team]
        away_index = 0
        if default_away < len(teams) and teams[default_away] in away_options:
            away_index = away_options.index(teams[default_away])
        away_team = st.selectbox("Away team", away_options, index=away_index)
        display_logo(team_crests.get(away_team), width=68)
        st.markdown(f"<div class='team-name'>{away_team}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    run_prediction = st.button("Generate match forecast", type="primary")

    if run_prediction:
        try:
            result = predict_match(
                home_team,
                away_team,
                competition,
                matches,
                model,
            )
        except ValueError as error:
            st.error(str(error))
            return

        home_goals = result["lambda_home"]
        away_goals = result["lambda_away"]
        score = result["most_likely_score"]
        st.markdown(
            f"""
            <div class="score-hero">
                <div class="score-label">Most likely score</div>
                <div class="score-value">{score[0]} — {score[1]}</div>
                <div class="muted" style="margin-top:0.6rem">
                    Expected goals&nbsp; {home_goals:.2f} — {away_goals:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        home_card, draw_card, away_card = st.columns(3, gap="medium")
        with home_card:
            probability_card("Home win", result["home_probability"])
        with draw_card:
            probability_card("Draw", result["draw_probability"])
        with away_card:
            probability_card("Away win", result["away_probability"])

        with st.expander("How to read this forecast"):
            st.write(
                "The exact score is the single most likely scoreline. The 1X2 favourite "
                "uses all home-win, draw, or away-win scorelines added together, so the "
                "two can point in different directions. Model-implied odds are 1 divided "
                "by probability and are not bookmaker odds or betting advice."
            )


def performance_page(metrics, predictions):
    app_header(
        "Frozen test season · 2025/26",
        "Performance you can audit.",
        "The fitted model was evaluated once on the untouched season and compared on the exact same matches with a causal competition-average baseline.",
    )

    model_metrics = metrics["fitted_model"]
    baseline_metrics = metrics["competition_average_baseline"]
    accuracy_gain = (model_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100

    accuracy_column, summary_column = st.columns([1.05, 1.95], gap="large")
    with accuracy_column:
        st.markdown(
            f"""
            <div class="accuracy-card">
                <div class="eyebrow">1X2 accuracy</div>
                <div class="accuracy-value">{model_metrics['accuracy'] * 100:.2f}%</div>
                <div class="status-pill">+{accuracy_gain:.2f} percentage points vs baseline</div>
                <div class="muted" style="margin-top:1rem">{model_metrics['matches']:,} supported matches</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_column:
        metric_one, metric_two = st.columns(2)
        metric_one.metric(
            "Log loss",
            f"{model_metrics['multiclass_log_loss']:.3f}",
            f"-{baseline_metrics['multiclass_log_loss'] - model_metrics['multiclass_log_loss']:.3f}",
            delta_color="inverse",
        )
        metric_two.metric(
            "Combined goal MAE",
            f"{model_metrics['combined_goal_mae']:.3f}",
            f"-{baseline_metrics['combined_goal_mae'] - model_metrics['combined_goal_mae']:.3f}",
            delta_color="inverse",
        )
        st.caption("Lower is better. The fitted model beat the baseline on every reported metric.")

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy (%)",
                "Log loss",
                "Brier score",
                "Home-goal MAE",
                "Away-goal MAE",
                "Combined-goal MAE",
            ],
            "Fitted model": [
                model_metrics["accuracy"] * 100,
                model_metrics["multiclass_log_loss"],
                model_metrics["multiclass_brier_score"],
                model_metrics["home_goal_mae"],
                model_metrics["away_goal_mae"],
                model_metrics["combined_goal_mae"],
            ],
            "Baseline": [
                baseline_metrics["accuracy"] * 100,
                baseline_metrics["multiclass_log_loss"],
                baseline_metrics["multiclass_brier_score"],
                baseline_metrics["home_goal_mae"],
                baseline_metrics["away_goal_mae"],
                baseline_metrics["combined_goal_mae"],
            ],
        }
    ).set_index("Metric")
    with st.expander("View all model metrics"):
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")

    model_rows = predictions.loc[predictions["model_name"] == "fitted_model"].copy()
    model_rows["correct"] = (
        model_rows["predicted_result"] == model_rows["actual_result"]
    )
    league_accuracy = (
        model_rows.groupby("competition_name")["correct"].mean().mul(100).sort_values()
    )
    with st.expander("View accuracy by competition"):
        st.bar_chart(league_accuracy, horizontal=True, color="#34d399")


def explorer_page(predictions):
    app_header(
        "Backtest explorer",
        "Inspect every prediction.",
        "Browse the untouched test season. Open filters or probability details only when you need them.",
    )

    with st.expander("Filters"):
        filter_one, filter_two, filter_three = st.columns(3)
        with filter_one:
            competition_options = ["All competitions"] + sorted(
                predictions["competition_name"].unique().tolist()
            )
            competition = st.selectbox("Competition", competition_options)
        with filter_two:
            model_label = st.selectbox(
                "Prediction source",
                ["Fitted model", "Competition average"],
            )
        with filter_three:
            result_filter = st.selectbox(
                "Actual result",
                ["All results", "Home", "Draw", "Away"],
            )

    model_name = (
        "fitted_model" if model_label == "Fitted model" else "competition_average"
    )
    filtered = predictions.loc[predictions["model_name"] == model_name].copy()
    if competition != "All competitions":
        filtered = filtered.loc[filtered["competition_name"] == competition]
    if result_filter != "All results":
        filtered = filtered.loc[filtered["actual_result"] == result_filter.lower()]

    filtered["correct"] = filtered["predicted_result"] == filtered["actual_result"]
    card_one, card_two = st.columns(2)
    card_one.metric("Matches", f"{len(filtered):,}")
    card_two.metric("Accuracy", f"{filtered['correct'].mean() * 100:.2f}%")

    show_probabilities = st.toggle("Show probability details", value=False)
    filtered["fixture"] = (
        filtered["home_team_name"] + " vs " + filtered["away_team_name"]
    )

    display_columns = [
        "match_date",
        "competition_name",
        "fixture",
        "most_likely_score",
        "predicted_result",
        "actual_result",
        "correct",
    ]
    if show_probabilities:
        display_columns[4:4] = [
            "home_probability",
            "draw_probability",
            "away_probability",
        ]
    table = filtered[display_columns].sort_values("match_date", ascending=False)
    if show_probabilities:
        for column in ["home_probability", "draw_probability", "away_probability"]:
            table[column] = table[column].map(lambda value: f"{value:.1%}")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "match_date": st.column_config.DateColumn("Date"),
            "competition_name": "Competition",
            "fixture": "Fixture",
            "most_likely_score": "Most likely exact score",
            "home_probability": "Home %",
            "draw_probability": "Draw %",
            "away_probability": "Away %",
            "predicted_result": "1X2 prediction",
            "actual_result": "Actual",
            "correct": "1X2 correct",
        },
    )
    st.caption(
        "The most likely exact score can be a draw while the 1X2 pick is home, because "
        "the home probability adds together every home-winning scoreline."
    )


def required_files_exist():
    required = [
        PATHS.football_data_matches_file,
        PATHS.model_file,
        PATHS.metrics_file,
        PATHS.backtest_predictions_file,
    ]
    return all(path.is_file() for path in required)


def main():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">⚽</div>
                <div>
                    <div class="brand-name">Touchline AI</div>
                    <div class="brand-subtitle">Poisson match intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Match predictor", "Model performance", "Backtest explorer"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("<div class='status-pill'>● Model ready</div>", unsafe_allow_html=True)

    if not required_files_exist():
        st.error(
            "The UI needs the processed recent data, saved model and backtest reports. "
            "Run tune-model and backtest first."
        )
        st.stop()

    matches = load_matches()
    model = load_saved_model()
    metrics = load_metrics()
    predictions = load_predictions()
    team_crests = load_team_crests()

    if page == "Match predictor":
        prediction_page(matches, model, team_crests)
    elif page == "Model performance":
        performance_page(metrics, predictions)
    else:
        explorer_page(predictions)


if __name__ == "__main__":
    main()
