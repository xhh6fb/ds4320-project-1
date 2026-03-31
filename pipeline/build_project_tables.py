import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl


# --------------------------------------------------
# logging setup
# --------------------------------------------------

Path("pipeline/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="pipeline/logs/build_project_tables.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("started build_project_tables.py")


# --------------------------------------------------
# helper functions
# --------------------------------------------------

def safe_to_pandas(df_like):
    """
    Convert a Polars DataFrame or pandas DataFrame into pandas.
    This keeps the rest of the project consistent.
    """
    try:
        if hasattr(df_like, "to_pandas"):
            return df_like.to_pandas()
        return pd.DataFrame(df_like)
    except Exception as e:
        logging.exception("failed to convert raw data to pandas")
        raise RuntimeError("could not convert raw data into pandas format") from e


def validate_required_columns(df, required_cols, df_name):
    """
    Check that a DataFrame contains the columns needed for the project.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error("missing required columns in %s: %s", df_name, missing)
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def add_game_id(df):
    """
    Create a readable unique game identifier.
    """
    df = df.copy()
    df["game_id"] = (
        df["season"].astype(str)
        + "_W"
        + df["week"].astype(str).str.zfill(2)
        + "_"
        + df["away_team"].astype(str)
        + "_at_"
        + df["home_team"].astype(str)
    )
    return df


def save_table_both_formats(df, output_dir, base_name):
    """
    Save a table as both CSV and parquet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{base_name}.csv"
    parquet_path = output_dir / f"{base_name}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logging.info("saved table %s with %s rows", base_name, len(df))
    print(f"saved: {csv_path}")
    print(f"saved: {parquet_path}")


def add_rolling_team_features(df):
    """
    Compute leakage-safe pregame rolling features for each team.
    Only games before the current row are used.
    """
    df = df.copy()
    df = df.sort_values(["team_id", "season", "gameday", "week", "game_id"]).reset_index(drop=True)

    grouped = df.groupby("team_id", group_keys=False)

    df["games_played_before"] = grouped.cumcount()

    df["cum_wins_before"] = grouped["win"].cumsum().shift(1).fillna(0)
    df["cum_losses_before"] = df["games_played_before"] - df["cum_wins_before"]

    df["cum_points_for_before"] = grouped["points_for"].cumsum().shift(1).fillna(0)
    df["cum_points_against_before"] = grouped["points_against"].cumsum().shift(1).fillna(0)

    gp = df["games_played_before"].replace(0, np.nan)

    # use neutral defaults for a team's first game in the data
    df["pregame_win_pct"] = (df["cum_wins_before"] / gp).fillna(0.5)
    df["pregame_points_for_pg"] = (df["cum_points_for_before"] / gp).fillna(0.0)
    df["pregame_points_against_pg"] = (df["cum_points_against_before"] / gp).fillna(0.0)
    df["pregame_point_diff_pg"] = df["pregame_points_for_pg"] - df["pregame_points_against_pg"]

    df["prev_gameday"] = grouped["gameday"].shift(1)
    df["days_rest"] = (df["gameday"] - df["prev_gameday"]).dt.days
    df["days_rest"] = df["days_rest"].fillna(7)

    df = df.drop(columns=["prev_gameday"])

    return df


# --------------------------------------------------
# main build function
# --------------------------------------------------

def main():
    """
    Build the four project tables:
    teams, games, team_games, and matchups.
    """
    try:
        output_dir = Path("data")

        seasons = list(range(2010, 2025))

        teams_raw = safe_to_pandas(nfl.load_teams())
        schedules_raw = safe_to_pandas(nfl.load_schedules())
        logging.info("successfully loaded raw nflreadpy data")

        required_schedule_cols = [
            "season", "game_type", "week", "gameday",
            "home_team", "away_team", "home_score", "away_score"
        ]
        validate_required_columns(schedules_raw, required_schedule_cols, "schedules_raw")

        # ------------------------------
        # build games table
        # ------------------------------
      
        games = schedules_raw.loc[
            (schedules_raw["season"].isin(seasons)) &
            (schedules_raw["game_type"] == "REG")
        ].copy()

        games = games.loc[
            games["home_score"].notna() &
            games["away_score"].notna()
        ].copy()

        games["season"] = games["season"].astype(int)
        games["week"] = pd.to_numeric(games["week"], errors="coerce").astype("Int64")
        games["gameday"] = pd.to_datetime(games["gameday"])

        games = games.loc[games["week"].notna()].copy()
        games["week"] = games["week"].astype(int)

        games = add_game_id(games)

        games = games[
            ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]
        ].copy()

        games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
        games["winner_team"] = np.where(games["home_win"] == 1, games["home_team"], games["away_team"])

        games = games.sort_values(["season", "gameday", "week", "game_id"]).reset_index(drop=True)

        # ------------------------------
        # build teams table
        # ------------------------------
      
        teams = teams_raw.copy()

        rename_map = {}
        if "team_abbr" in teams.columns:
            rename_map["team_abbr"] = "team_id"
        elif "team" in teams.columns:
            rename_map["team"] = "team_id"

        if "team_name" in teams.columns:
            rename_map["team_name"] = "team_name"
        if "team_nick" in teams.columns:
            rename_map["team_nick"] = "team_nick"
        if "team_conf" in teams.columns:
            rename_map["team_conf"] = "team_conf"
        elif "team_conference" in teams.columns:
            rename_map["team_conference"] = "team_conf"

        if "team_division" in teams.columns:
            rename_map["team_division"] = "team_division"
        elif "team_div" in teams.columns:
            rename_map["team_div"] = "team_division"

        teams = teams.rename(columns=rename_map).copy()

        team_cols = [c for c in ["team_id", "team_name", "team_nick", "team_conf", "team_division"] if c in teams.columns]
        teams = teams[team_cols].drop_duplicates().copy()

        validate_required_columns(teams, ["team_id"], "teams")
        teams = teams.sort_values("team_id").reset_index(drop=True)

        # ------------------------------
        # build team_games table
        # ------------------------------
      
        home_side = games.copy()
        home_side["team_id"] = home_side["home_team"]
        home_side["opponent_team"] = home_side["away_team"]
        home_side["is_home"] = 1
        home_side["points_for"] = home_side["home_score"]
        home_side["points_against"] = home_side["away_score"]
        home_side["win"] = (home_side["home_score"] > home_side["away_score"]).astype(int)

        away_side = games.copy()
        away_side["team_id"] = away_side["away_team"]
        away_side["opponent_team"] = away_side["home_team"]
        away_side["is_home"] = 0
        away_side["points_for"] = away_side["away_score"]
        away_side["points_against"] = away_side["home_score"]
        away_side["win"] = (away_side["away_score"] > away_side["home_score"]).astype(int)

        team_games = pd.concat([home_side, away_side], ignore_index=True)

        team_games = team_games[
            [
                "game_id", "season", "week", "gameday",
                "team_id", "opponent_team", "is_home",
                "points_for", "points_against", "win"
            ]
        ].copy()

        team_games = add_rolling_team_features(team_games)

        # ------------------------------
        # build matchups table
        # ------------------------------
      
        home_features = team_games.loc[team_games["is_home"] == 1].copy()
        away_features = team_games.loc[team_games["is_home"] == 0].copy()

        home_features = home_features[
            [
                "game_id", "team_id",
                "pregame_win_pct", "pregame_points_for_pg",
                "pregame_points_against_pg", "pregame_point_diff_pg",
                "days_rest"
            ]
        ].rename(columns={
            "team_id": "home_team",
            "pregame_win_pct": "home_pregame_win_pct",
            "pregame_points_for_pg": "home_pregame_points_for_pg",
            "pregame_points_against_pg": "home_pregame_points_against_pg",
            "pregame_point_diff_pg": "home_pregame_point_diff_pg",
            "days_rest": "home_days_rest"
        })

        away_features = away_features[
            [
                "game_id", "team_id",
                "pregame_win_pct", "pregame_points_for_pg",
                "pregame_points_against_pg", "pregame_point_diff_pg",
                "days_rest"
            ]
        ].rename(columns={
            "team_id": "away_team",
            "pregame_win_pct": "away_pregame_win_pct",
            "pregame_points_for_pg": "away_pregame_points_for_pg",
            "pregame_points_against_pg": "away_pregame_points_against_pg",
            "pregame_point_diff_pg": "away_pregame_point_diff_pg",
            "days_rest": "away_days_rest"
        })

        matchups = games.merge(home_features, on=["game_id", "home_team"], how="left")
        matchups = matchups.merge(away_features, on=["game_id", "away_team"], how="left")

        matchups["pregame_win_pct_diff"] = matchups["home_pregame_win_pct"] - matchups["away_pregame_win_pct"]
        matchups["pregame_points_for_pg_diff"] = matchups["home_pregame_points_for_pg"] - matchups["away_pregame_points_for_pg"]
        matchups["pregame_points_against_pg_diff"] = matchups["home_pregame_points_against_pg"] - matchups["away_pregame_points_against_pg"]
        matchups["pregame_point_diff_pg_diff"] = matchups["home_pregame_point_diff_pg"] - matchups["away_pregame_point_diff_pg"]
        matchups["days_rest_diff"] = matchups["home_days_rest"] - matchups["away_days_rest"]
        matchups["target_home_win"] = matchups["home_win"]

        # ------------------------------
        # save tables
        # ------------------------------
      
        save_table_both_formats(teams, output_dir, "teams")
        save_table_both_formats(games, output_dir, "games")
        save_table_both_formats(team_games, output_dir, "team_games")
        save_table_both_formats(matchups, output_dir, "matchups")

        logging.info("finished successfully")
        print("\nAll project tables were created successfully.")

    except Exception as e:
        logging.exception("Project table creation failed.")
        raise RuntimeError("Failed to build project tables.") from e


if __name__ == "__main__":
    main()
