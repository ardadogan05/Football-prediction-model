import pandas as pd


COMPETITIONS = [
    (2, "Premier League"),
    (9, "Bundesliga"),
    (7, "Ligue 1"),
    (12, "Serie A"),
    (11, "La Liga"),
]


def make_season(source, season_name, season_id, start_date, id_offset=0):
    matches = []
    start_date = pd.Timestamp(start_date)

    for competition_id, competition_name in COMPETITIONS:
        for index in range(20):
            home_number = (index % 4) + 1
            away_number = ((index + 1) % 4) + 1
            home_id = competition_id * 100 + home_number
            away_id = competition_id * 100 + away_number
            matches.append(
                {
                    "source": source,
                    "match_id": id_offset + competition_id * 1000 + index,
                    "match_date": start_date + pd.Timedelta(index * 7, unit="D"),
                    "competition_id": competition_id,
                    "competition_name": competition_name,
                    "season_id": season_id,
                    "season_name": season_name,
                    "home_team_id": home_id,
                    "home_team_name": f"{competition_name} Team {home_number}",
                    "away_team_id": away_id,
                    "away_team_name": f"{competition_name} Team {away_number}",
                    "home_goals": (index + competition_id) % 4,
                    "away_goals": (index * 2 + competition_id) % 3,
                    "eligible_for_model": True,
                }
            )

    return pd.DataFrame(matches)


def training_and_recent_matches():
    training = make_season(
        "statsbomb", "2023/2024", 23, "2023-08-01", id_offset=0
    )
    validation = make_season(
        "football_data", "2024/2025", 24, "2024-08-01", id_offset=100000
    )
    test = make_season(
        "football_data", "2025/2026", 25, "2025-08-01", id_offset=200000
    )
    recent = pd.concat([validation, test], ignore_index=True)
    return training, recent
