from pathlib import Path
import logging

import numpy as np
import pandas as pd
import nflreadpy as nfl
import duckdb


# --------------------------------------------------
# setup folders and logging
# --------------------------------------------------

# make sure the logs folder exists
Path("pipeline/logs").mkdir(parents=True, exist_ok=True)

# make sure the data folder exists
Path("data").mkdir(parents=True, exist_ok=True)

# create the log file for this script
logging.basicConfig(
    filename="pipeline/logs/build_project_tables.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("started build_project_tables.py file")


# --------------------------------------------------
# helper functions
# --------------------------------------------------

def safe_to_pandas(df_like):
    """
    convert a polars dataframe or pandas dataframe into pandas
    so the rest of the script can use one dataframe type
    """
    try:
        # nflreadpy may return a polars dataframe
        if hasattr(df_like, "to_pandas"):
            return df_like.to_pandas()

        # if it is already pandas-like, convert directly
        return pd.DataFrame(df_like)

    except Exception as e:
        logging.exception("failed to convert raw data to pandas")
        raise RuntimeError("could not convert raw data into pandas format") from e


def validate_required_columns(df, required_cols, df_name):
    """
    check whether a dataframe has the columns we need
    before we continue building tables
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        logging.error("missing required columns in %s: %s", df_name, missing)
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def add_game_id(df):
    """
    create a readable game id using season, week, away team, and home team
    """
    df = df.copy()

    # example: 2023_w01_hou_at_bal
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
    convert american moneyline odds into implied probability

    positive odds:
    prob = 100 / (odds + 100)

    negative odds:
    prob = abs(odds) / (abs(odds) + 100)
    """
    # force the series into numeric form
    s = pd.to_numeric(series, errors="coerce")

    # create an empty float series to store probabilities
    prob = pd.Series(np.nan, index=s.index, dtype="float64")

    # positive american odds
    pos_mask = s > 0

    # negative american odds
    neg_mask = s < 0

    # fill positive moneyline probabilities
    prob.loc[pos_mask] = 100 / (s.loc[pos_mask] + 100)

    # fill negative moneyline probabilities
    prob.loc[neg_mask] = (-s.loc[neg_mask]) / ((-s.loc[neg_mask]) + 100)

    return prob


def add_rolling_team_features(df):
    """
    compute leakage-safe rolling features for each team

    the key rule is:
    only use games that happened before the current row
    """
    df = df.copy()

    # sort so rolling calculations happen in true game order
    df = df.sort_values(
        ["team_id", "season", "gameday", "week", "game_id"]
    ).reset_index(drop=True)

    # group by team so each team gets its own rolling history
    grouped = df.groupby("team_id", group_keys=False)

    # count how many games the team had already played
    df["games_played_before"] = grouped.cumcount()

    # cumulative wins before the current game
    df["cum_wins_before"] = grouped["win"].cumsum().shift(1).fillna(0)

    # cumulative losses before the current game
    df["cum_losses_before"] = df["games_played_before"] - df["cum_wins_before"]

    # cumulative points scored before the current game
    df["cum_points_for_before"] = grouped["points_for"].cumsum().shift(1).fillna(0)

    # cumulative points allowed before the current game
    df["cum_points_against_before"] = grouped["points_against"].cumsum().shift(1).fillna(0)

    # replace 0 with nan to avoid dividing by 0
    gp = df["games_played_before"].replace(0, np.nan)

    # full-history pregame win rate
    df["pregame_win_pct"] = (df["cum_wins_before"] / gp).fillna(0.5)

    # full-history average points scored entering the game
    df["pregame_points_for_pg"] = (df["cum_points_for_before"] / gp).fillna(0.0)

    # full-history average points allowed entering the game
    df["pregame_points_against_pg"] = (df["cum_points_against_before"] / gp).fillna(0.0)

    # full-history point differential entering the game
    df["pregame_point_diff_pg"] = (
        df["pregame_points_for_pg"] - df["pregame_points_against_pg"]
    )

    # average points scored across the previous 3 games
    df["pregame_last3_points_for_pg"] = (
        grouped["points_for"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    # average points allowed across the previous 3 games
    df["pregame_last3_points_against_pg"] = (
        grouped["points_against"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    # win percentage across the previous 3 games
    df["pregame_last3_win_pct"] = (
        grouped["win"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.5)
    )

    # recent-form point differential
    df["pregame_last3_point_diff_pg"] = (
        df["pregame_last3_points_for_pg"] - df["pregame_last3_points_against_pg"]
    )

    # get the date of the team's previous game
    df["prev_gameday"] = grouped["gameday"].shift(1)

    # calculate days of rest from previous game
    df["days_rest_calc"] = (df["gameday"] - df["prev_gameday"]).dt.days

    # for a team's first row, use 7 as a neutral default
    df["days_rest_calc"] = df["days_rest_calc"].fillna(7)

    # drop helper column we no longer need
    df = df.drop(columns=["prev_gameday"])

    return df


def export_duckdb_tables_to_parquet(con):
    """
    export the final duckdb tables to parquet files in data/
    """
    # export each final table from duckdb into parquet
    con.execute("copy teams to 'data/teams.parquet' (format parquet)")
    con.execute("copy games to 'data/games.parquet' (format parquet)")
    con.execute("copy team_games to 'data/team_games.parquet' (format parquet)")
    con.execute("copy matchups to 'data/matchups.parquet' (format parquet)")

    logging.info("exported duckdb tables to parquet files")

    print("saved: data/teams.parquet")
    print("saved: data/games.parquet")
    print("saved: data/team_games.parquet")
    print("saved: data/matchups.parquet")


# --------------------------------------------------
# main build function
# --------------------------------------------------

def main():
    """
    build the four project tables, load them into duckdb,
    and export the final tables from duckdb as parquet
    """
    try:
        # choose the nfl seasons to use
        seasons = list(range(2010, 2025))

        # load raw team metadata
        teams_raw = safe_to_pandas(nfl.load_teams())

        # load raw schedule data for the seasons we want
        schedules_raw = safe_to_pandas(nfl.load_schedules(seasons=seasons))

        logging.info("successfully loaded raw nflreadpy data")

        # make sure the source schedule file has the columns we need
        required_schedule_cols = [
            "season", "game_type", "week", "gameday",
            "home_team", "away_team", "home_score", "away_score"
        ]
        validate_required_columns(schedules_raw, required_schedule_cols, "schedules_raw")

        # ------------------------------
        # build games table
        # ------------------------------

        # keep only nfl regular-season games in our season range
        games = schedules_raw.loc[
            (schedules_raw["season"].isin(seasons)) &
            (schedules_raw["game_type"] == "REG")
        ].copy()

        # keep only completed games with final scores
        games = games.loc[
            games["home_score"].notna() &
            games["away_score"].notna()
        ].copy()

        # convert important columns to the right types
        games["season"] = games["season"].astype(int)
        games["week"] = pd.to_numeric(games["week"], errors="coerce").astype("Int64")
        games["gameday"] = pd.to_datetime(games["gameday"])

        # drop rows where week could not be interpreted
        games = games.loc[games["week"].notna()].copy()
        games["week"] = games["week"].astype(int)

        # create a readable primary key for each game
        games = add_game_id(games)

        # optional columns may or may not exist depending on the source file
        optional_cols = [
            "home_rest", "away_rest",
            "home_moneyline", "away_moneyline",
            "spread_line", "total_line",
            "div_game", "roof", "surface", "temp", "wind"
        ]
        keep_optional = [c for c in optional_cols if c in games.columns]

        # keep the core columns plus whatever optional context exists
        games = games[
            ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"] + keep_optional
        ].copy()

        # binary target from the final score
        games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

        # store the winning team id
        games["winner_team"] = np.where(
            games["home_win"] == 1,
            games["home_team"],
            games["away_team"]
        )

        # create market-implied home probability if moneyline exists
        if "home_moneyline" in games.columns:
            games["market_home_implied_prob"] = moneyline_to_implied_prob(games["home_moneyline"])
        else:
            games["market_home_implied_prob"] = np.nan

        # create market-implied away probability if moneyline exists
        if "away_moneyline" in games.columns:
            games["market_away_implied_prob"] = moneyline_to_implied_prob(games["away_moneyline"])
        else:
            games["market_away_implied_prob"] = np.nan

        # difference between the two implied probabilities
        games["market_implied_prob_diff"] = (
            games["market_home_implied_prob"] - games["market_away_implied_prob"]
        )

        # create rest difference if both source rest variables exist
        if "home_rest" in games.columns and "away_rest" in games.columns:
            games["sched_rest_diff"] = games["home_rest"] - games["away_rest"]
        else:
            games["sched_rest_diff"] = np.nan

        # sort in chronological order
        games = games.sort_values(
            ["season", "gameday", "week", "game_id"]
        ).reset_index(drop=True)

        # ------------------------------
        # build teams table
        # ------------------------------

        teams = teams_raw.copy()

        # rename possible source columns into one standard set of names
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

        # remove duplicate column labels if they exist
        teams = teams.loc[:, ~teams.columns.duplicated()].copy()

        # keep only the columns we need
        team_cols = [
            c for c in ["team_id", "team_name", "team_nick", "team_conf", "team_division"]
            if c in teams.columns
        ]
        teams = teams[team_cols].copy()

        # make sure the key exists
        validate_required_columns(teams, ["team_id"], "teams")

        # keep one row per team id
        teams = teams.drop_duplicates(subset=["team_id"]).sort_values("team_id").reset_index(drop=True)

        # ------------------------------
        # build team_games table
        # ------------------------------

        # create the home-team perspective
        home_side = games.copy()
        home_side["team_id"] = home_side["home_team"]
        home_side["opponent_team"] = home_side["away_team"]
        home_side["is_home"] = 1
        home_side["points_for"] = home_side["home_score"]
        home_side["points_against"] = home_side["away_score"]
        home_side["win"] = (home_side["home_score"] > home_side["away_score"]).astype(int)

        # create the away-team perspective
        away_side = games.copy()
        away_side["team_id"] = away_side["away_team"]
        away_side["opponent_team"] = away_side["home_team"]
        away_side["is_home"] = 0
        away_side["points_for"] = away_side["away_score"]
        away_side["points_against"] = away_side["home_score"]
        away_side["win"] = (away_side["away_score"] > away_side["home_score"]).astype(int)

        # stack both perspectives together
        team_games = pd.concat([home_side, away_side], ignore_index=True)

        # keep only the columns needed for rolling calculations
        team_games = team_games[
            [
                "game_id", "season", "week", "gameday",
                "team_id", "opponent_team", "is_home",
                "points_for", "points_against", "win"
            ]
        ].copy()

        # add leakage-safe rolling features
        team_games = add_rolling_team_features(team_games)

        # ------------------------------
        # build matchups table
        # ------------------------------

        # split team_games into home rows and away rows
        home_features = team_games.loc[team_games["is_home"] == 1].copy()
        away_features = team_games.loc[team_games["is_home"] == 0].copy()

        # rename home-side rolling features
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

        # rename away-side rolling features
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

        # merge the home features into the games table
        matchups = games.merge(home_features, on=["game_id", "home_team"], how="left")

        # merge the away features into the games table
        matchups = matchups.merge(away_features, on=["game_id", "away_team"], how="left")

        # create home-minus-away rolling difference features
        matchups["pregame_win_pct_diff"] = matchups["home_pregame_win_pct"] - matchups["away_pregame_win_pct"]
        matchups["pregame_points_for_pg_diff"] = matchups["home_pregame_points_for_pg"] - matchups["away_pregame_points_for_pg"]
        matchups["pregame_points_against_pg_diff"] = matchups["home_pregame_points_against_pg"] - matchups["away_pregame_points_against_pg"]
        matchups["pregame_point_diff_pg_diff"] = matchups["home_pregame_point_diff_pg"] - matchups["away_pregame_point_diff_pg"]

        # create home-minus-away cumulative count features
        matchups["games_played_before_diff"] = matchups["home_games_played_before"] - matchups["away_games_played_before"]
        matchups["cum_wins_before_diff"] = matchups["home_cum_wins_before"] - matchups["away_cum_wins_before"]
        matchups["cum_losses_before_diff"] = matchups["home_cum_losses_before"] - matchups["away_cum_losses_before"]

        # create home-minus-away recent-form features
        matchups["last3_win_pct_diff"] = matchups["home_pregame_last3_win_pct"] - matchups["away_pregame_last3_win_pct"]
        matchups["last3_point_diff_pg_diff"] = matchups["home_pregame_last3_point_diff_pg"] - matchups["away_pregame_last3_point_diff_pg"]

        # create calculated rest difference
        matchups["calc_rest_diff"] = matchups["home_days_rest_calc"] - matchups["away_days_rest_calc"]

        # final binary target
        matchups["target_home_win"] = matchups["home_win"]

        # ------------------------------
        # load into duckdb
        # ------------------------------

        # connect to a local duckdb database file
        con = duckdb.connect("data/project_1.duckdb")

        # register the pandas dataframes so duckdb can query them
        con.register("teams_df", teams)
        con.register("games_df", games)
        con.register("team_games_df", team_games)
        con.register("matchups_df", matchups)

        # create duckdb tables from the pandas dataframes
        con.execute("create or replace table teams as select * from teams_df")
        con.execute("create or replace table games as select * from games_df")
        con.execute("create or replace table team_games as select * from team_games_df")
        con.execute("create or replace table matchups as select * from matchups_df")

        logging.info("loaded final pandas tables into duckdb")

        # export the final duckdb tables to parquet
        export_duckdb_tables_to_parquet(con)

        # close the connection
        con.close()

        logging.info("finished successfully")
        print("\nall project tables were created successfully")

    except Exception as e:
        logging.exception("project table creation failed")
        raise RuntimeError("failed to build project tables") from e


if __name__ == "__main__":
    main()
