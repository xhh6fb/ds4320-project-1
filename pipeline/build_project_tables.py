from pathlib import Path
import logging

import numpy as np
import pandas as pd
import nflreadpy as nfl
import duckdb

# --------------------------------------------------
# logging
# --------------------------------------------------

Path("pipeline/logs").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="pipeline/logs/build_project_tables.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Started build_project_tables.py file.")


# --------------------------------------------------
# helper functions
# --------------------------------------------------

def safe_to_pandas(df_like):
    """
    Convert a Polars DataFrame or pandas DataFrame into pandas.
    """
    try:
        if hasattr(df_like, "to_pandas"):
            return df_like.to_pandas()
        return pd.DataFrame(df_like)
    except Exception as e:
        logging.exception("Failed to convert raw data to pandas.")
        raise RuntimeError("Could not convert raw data into pandas format.") from e


def validate_required_columns(df, required_cols, df_name):
    """
    Check that a DataFrame contains the required columns.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error("Missing required columns in %s: %s.", df_name, missing)
        raise ValueError(f"{df_name} is missing required columns: {missing}.")


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


def moneyline_to_implied_prob(series):
    """
    Convert American moneyline odds to implied probability.
    """
    s = pd.to_numeric(series, errors="coerce")

    prob = pd.Series(np.nan, index=s.index, dtype="float64")

    pos_mask = s > 0
    neg_mask = s < 0

    prob.loc[pos_mask] = 100 / (s.loc[pos_mask] + 100)
    prob.loc[neg_mask] = (-s.loc[neg_mask]) / ((-s.loc[neg_mask]) + 100)

    return prob


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

    # full-history pregame features
    df["pregame_win_pct"] = (df["cum_wins_before"] / gp).fillna(0.5)
    df["pregame_points_for_pg"] = (df["cum_points_for_before"] / gp).fillna(0.0)
    df["pregame_points_against_pg"] = (df["cum_points_against_before"] / gp).fillna(0.0)
    df["pregame_point_diff_pg"] = df["pregame_points_for_pg"] - df["pregame_points_against_pg"]

    # recent-form features based on previous 3 games
    df["pregame_last3_points_for_pg"] = (
        grouped["points_for"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    df["pregame_last3_points_against_pg"] = (
        grouped["points_against"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    df["pregame_last3_win_pct"] = (
        grouped["win"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.5)
    )

    df["pregame_last3_point_diff_pg"] = (
        df["pregame_last3_points_for_pg"] - df["pregame_last3_points_against_pg"]
    )

    # rest computed from dates
    df["prev_gameday"] = grouped["gameday"].shift(1)
    df["days_rest_calc"] = (df["gameday"] - df["prev_gameday"]).dt.days
    df["days_rest_calc"] = df["days_rest_calc"].fillna(7)

    df = df.drop(columns=["prev_gameday"])

    return df


def export_duckdb_tables_to_parquet(con):
    """
    Export final DuckDB tables to parquet files.
    """
    con.execute("COPY teams TO 'data/teams.parquet' (FORMAT PARQUET)")
    con.execute("COPY games TO 'data/games.parquet' (FORMAT PARQUET)")
    con.execute("COPY team_games TO 'data/team_games.parquet' (FORMAT PARQUET)")
    con.execute("COPY matchups TO 'data/matchups.parquet' (FORMAT PARQUET)")

    logging.info("Exported duckdb tables to parquet files.")
    print("Saved: data/teams.parquet")
    print("Saved: data/games.parquet")
    print("Saved: data/team_games.parquet")
    print("Saved: data/matchups.parquet")


# --------------------------------------------------
# main build function
# --------------------------------------------------
def main():
    """
    Build the four project tables, load them into DuckDB,
    and export the final tables from DuckDB as parquet.
    """
    try:
        seasons = list(range(2010, 2025))

        teams_raw = safe_to_pandas(nfl.load_teams())
        schedules_raw = safe_to_pandas(nfl.load_schedules(seasons=seasons))
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

        optional_cols = [
            "home_rest", "away_rest",
            "home_moneyline", "away_moneyline",
            "spread_line", "total_line",
            "div_game", "roof", "surface", "temp", "wind"
        ]
        keep_optional = [c for c in optional_cols if c in games.columns]

        games = games[
            ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"] + keep_optional
        ].copy()

        games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
        games["winner_team"] = np.where(games["home_win"] == 1, games["home_team"], games["away_team"])

        # market-implied probability features
        if "home_moneyline" in games.columns:
            games["market_home_implied_prob"] = moneyline_to_implied_prob(games["home_moneyline"])
        else:
            games["market_home_implied_prob"] = np.nan

        if "away_moneyline" in games.columns:
            games["market_away_implied_prob"] = moneyline_to_implied_prob(games["away_moneyline"])
        else:
            games["market_away_implied_prob"] = np.nan

        games["market_implied_prob_diff"] = (
            games["market_home_implied_prob"] - games["market_away_implied_prob"]
        )

        if "home_rest" in games.columns and "away_rest" in games.columns:
            games["sched_rest_diff"] = games["home_rest"] - games["away_rest"]
        else:
            games["sched_rest_diff"] = np.nan

        games = games.sort_values(["season", "gameday", "week", "game_id"]).reset_index(drop=True)

        # ------------------------------
        # build teams table
        # ------------------------------
        teams = teams_raw.copy()

        rename_map = {}

        if "team_id" not in teams.columns:
            if "team_abbr" in teams.columns:
                rename_map["team_abbr"] = "team_id"
            elif "team" in teams.columns:
                rename_map["team"] = "team_id"

        if "team_name" not in teams.columns and "team_full_name" in teams.columns:
            rename_map["team_full_name"] = "team_name"

        if "team_conf" not in teams.columns and "team_conference" in teams.columns:
            rename_map["team_conference"] = "team_conf"

        if "team_division" not in teams.columns and "team_div" in teams.columns:
            rename_map["team_div"] = "team_division"

        teams = teams.rename(columns=rename_map).copy()
        teams = teams.loc[:, ~teams.columns.duplicated()].copy()

        team_cols = [c for c in ["team_id", "team_name", "team_nick", "team_conf", "team_division"] if c in teams.columns]
        teams = teams[team_cols].copy()

        validate_required_columns(teams, ["team_id"], "teams")
        teams = teams.drop_duplicates(subset=["team_id"]).sort_values("team_id").reset_index(drop=True)

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
                "games_played_before", "cum_wins_before", "cum_losses_before",
                "pregame_win_pct", "pregame_points_for_pg",
                "pregame_points_against_pg", "pregame_point_diff_pg",
                "pregame_last3_win_pct", "pregame_last3_point_diff_pg",
                "days_rest_calc"
            ]
        ].rename(columns={
            "team_id": "home_team",
            "games_played_before": "home_games_played_before",
            "cum_wins_before": "home_cum_wins_before",
            "cum_losses_before": "home_cum_losses_before",
            "pregame_win_pct": "home_pregame_win_pct",
            "pregame_points_for_pg": "home_pregame_points_for_pg",
            "pregame_points_against_pg": "home_pregame_points_against_pg",
            "pregame_point_diff_pg": "home_pregame_point_diff_pg",
            "pregame_last3_win_pct": "home_pregame_last3_win_pct",
            "pregame_last3_point_diff_pg": "home_pregame_last3_point_diff_pg",
            "days_rest_calc": "home_days_rest_calc"
        })

        away_features = away_features[
            [
                "game_id", "team_id",
                "games_played_before", "cum_wins_before", "cum_losses_before",
                "pregame_win_pct", "pregame_points_for_pg",
                "pregame_points_against_pg", "pregame_point_diff_pg",
                "pregame_last3_win_pct", "pregame_last3_point_diff_pg",
                "days_rest_calc"
            ]
        ].rename(columns={
            "team_id": "away_team",
            "games_played_before": "away_games_played_before",
            "cum_wins_before": "away_cum_wins_before",
            "cum_losses_before": "away_cum_losses_before",
            "pregame_win_pct": "away_pregame_win_pct",
            "pregame_points_for_pg": "away_pregame_points_for_pg",
            "pregame_points_against_pg": "away_pregame_points_against_pg",
            "pregame_point_diff_pg": "away_pregame_point_diff_pg",
            "pregame_last3_win_pct": "away_pregame_last3_win_pct",
            "pregame_last3_point_diff_pg": "away_pregame_last3_point_diff_pg",
            "days_rest_calc": "away_days_rest_calc"
        })

        matchups = games.merge(home_features, on=["game_id", "home_team"], how="left")
        matchups = matchups.merge(away_features, on=["game_id", "away_team"], how="left")

        # rolling difference features
        matchups["pregame_win_pct_diff"] = matchups["home_pregame_win_pct"] - matchups["away_pregame_win_pct"]
        matchups["pregame_points_for_pg_diff"] = matchups["home_pregame_points_for_pg"] - matchups["away_pregame_points_for_pg"]
        matchups["pregame_points_against_pg_diff"] = matchups["home_pregame_points_against_pg"] - matchups["away_pregame_points_against_pg"]
        matchups["pregame_point_diff_pg_diff"] = matchups["home_pregame_point_diff_pg"] - matchups["away_pregame_point_diff_pg"]

        matchups["games_played_before_diff"] = matchups["home_games_played_before"] - matchups["away_games_played_before"]
        matchups["cum_wins_before_diff"] = matchups["home_cum_wins_before"] - matchups["away_cum_wins_before"]
        matchups["cum_losses_before_diff"] = matchups["home_cum_losses_before"] - matchups["away_cum_losses_before"]

        matchups["last3_win_pct_diff"] = matchups["home_pregame_last3_win_pct"] - matchups["away_pregame_last3_win_pct"]
        matchups["last3_point_diff_pg_diff"] = matchups["home_pregame_last3_point_diff_pg"] - matchups["away_pregame_last3_point_diff_pg"]

        matchups["calc_rest_diff"] = matchups["home_days_rest_calc"] - matchups["away_days_rest_calc"]

        matchups["target_home_win"] = matchups["home_win"]

        # ------------------------------
        # load into duckdb
        # ------------------------------
        con = duckdb.connect("data/project_1.duckdb")

        con.register("teams_df", teams)
        con.register("games_df", games)
        con.register("team_games_df", team_games)
        con.register("matchups_df", matchups)

        con.execute("CREATE OR REPLACE TABLE teams AS SELECT * FROM teams_df")
        con.execute("CREATE OR REPLACE TABLE games AS SELECT * FROM games_df")
        con.execute("CREATE OR REPLACE TABLE team_games AS SELECT * FROM team_games_df")
        con.execute("CREATE OR REPLACE TABLE matchups AS SELECT * FROM matchups_df")

        logging.info("loaded final pandas tables into duckdb")

        export_duckdb_tables_to_parquet(con)

        con.close()

        logging.info("Finished successfully.")
        print("\nAll project tables were created successfully.")

    except Exception as e:
        logging.exception("Project table creation failed.")
        raise RuntimeError("Failed to build project tables.") from e


if __name__ == "__main__":
    main()
