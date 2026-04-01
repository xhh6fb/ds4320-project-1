# DS 4320 Project 1: Predicting NFL Home-Team Win Probability

## Executive Summary

This repository contains my DS 4320 Project on predicting NFL regular-season game outcomes using a relational secondary dataset built from nflverse data accessed through the `nflreadpy` Python package. The project refines the broad problem of predicting sports game outcomes into a specific pregame classification task: estimating the probability that the home team will win an NFL regular-season game using only information available before kickoff. The repository includes project documentation, a press release, background readings, a linked data folder, SQL queries, Python code for data creation, and a reproducible modeling pipeline in Python, Markdown, and SQL.

<br>

||Details|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | FIX THIS [https://doi.org/10.1000/182](https://doi.org/10.1000/182) |
| Press Release | [How NFL Fans Can Turn Pregame Data into Game-Day Predictions](press_release.md) |
| Data | FIX THIS [UVA OneDrive data folder](REPLACE-onedrive-link) |
| Pipeline | [Notebook](pipeline/project_1_pipeline.ipynb) & [Markdown](pipeline/project_1_pipeline.md)   |
| License | [MIT](LICENSE) |

<br>

## Problem Definition

### Initial General Problem & Refined Specific Problem Statement

**Initial General Problem:** Predicting sports game outcomes.

**Refined Specific Problem Statement:** Develop a data-driven system that predicts whether the home team will win an NFL regular-season game using only pregame information derived from historical team form, recent team form, schedule context, and pregame market information. The final prediction target is a binary variable equal to 1 when the home team wins and 0 otherwise.

### Rationale for Refinement

I refined the general problem of predicting sports game outcomes into predicting NFL regular-season home-team wins because the original problem was too broad to support a clear dataset, a coherent relational structure, and a realistic prediction target. “Sports games” could refer to many different leagues and sports, each with different rules, schedules, scoring systems, and useful predictors. That would make it difficult to define consistent variables, construct meaningful linked tables, and explain what the model is trying to predict.

By narrowing the project to NFL regular-season games, the problem becomes much more focused and manageable. The NFL has a standardized season format, clearly recorded game outcomes, a fixed set of teams, and public schedule-level data that can be transformed into pregame features. Limiting the task to pregame prediction also makes the project more realistic because the goal is to estimate the likely outcome before kickoff rather than describe what already happened after the game ended.

### Motivation

I chose this project because football is one of the subjects I care most about, and I want to build toward a future career in the NFL. More specifically, I hope to work toward becoming a coach in the league, and ideally even the first female head coach in the NFL. Because of that, this project is not just an academic exercise for me. It connects directly to something I genuinely care about and want to understand at a deeper level.

This project also matters to me because football is often discussed emotionally and based on intuition, but successful teams increasingly use analytics to support preparation, game planning, player evaluation, and decision-making. I want to understand that side of the sport better. Building a project around NFL game prediction helps me learn which factors actually matter when evaluating teams before a game, how to think critically about performance rather than relying only on record or reputation, and how data can support strategic football thinking.

### Press Release
 
[How NFL Fans Can Turn Pregame Data into Game-Day Predictions](press_release.md)

<br>

## Domain Exposition

### Terminology

| Term | Meaning in this project | Why it matters |
|---|---|---|
| NFL | National Football League | The league this project focuses on |
| regular-season game | An NFL game that is part of the standard season schedule, excluding playoffs | Keeps the scope and target consistent |
| home-team win | A binary target equal to 1 if the home team scores more points than the away team | This is the final prediction outcome |
| target variable | The value the model is trying to predict | Here it is whether the home team wins |
| feature | An input variable used by the model | The model uses only pregame predictors |
| pregame feature | A variable computed only from information available before kickoff | Prevents leakage |
| leakage | Using current-game or future information to predict the current outcome | Would make the model misleadingly strong |
| relational model | A data design built from linked tables with keys and joins | Required by the project rubric |
| team reference table | A lookup table with team identifiers and metadata | Supports joins and documentation |
| game table | One row per completed NFL regular-season game | Core fact table for outcomes |
| team-game table | One row per team per game | Allows rolling pregame features to be computed safely |
| matchup table | One row per game with final pregame features | Final modeling table |
| pregame win percentage | A team’s win rate before the current game | A simple measure of prior form |
| pregame point differential per game | Prior average points scored minus prior average points allowed | Summarizes overall team strength |
| recent form | A team’s performance over the most recent few games | Helps capture momentum or short-term performance changes |
| spread line | Pregame betting spread | A strong public market signal about expected relative team strength |
| moneyline | Pregame betting odds for each side | Can be converted into implied win probabilities |
| implied probability | Probability implied by betting odds | Useful as a market-based predictor |
| divisional game | A game between two teams from the same division | These games may behave differently from non-division games |
| days of rest | Time since the previous game | Captures scheduling context |
| DuckDB | An in-process analytical database used here for parquet loading and SQL analysis | Supports reproducibility and SQL-based analysis |
| logistic regression | A classification model that estimates the probability of a binary outcome | Interpretable baseline model |
| random forest | An ensemble tree-based classification model | More flexible nonlinear comparison model |
| ROC AUC | A ranking-based metric for binary classification | Helps compare model performance beyond raw accuracy |
| confusion matrix | A table of correct and incorrect classification outcomes | Helps interpret model behavior |

### Domain

This project lives in the domain of **sports analytics**, and more specifically in football analytics and predictive modeling. Sports analytics uses data to better understand team performance, strategy, and likely future outcomes. Within that broader domain, football analytics focuses on measuring what leads to success in football games and how team strength can be estimated more accurately than by wins and losses alone.

Predicting NFL game outcomes is a natural problem in this domain because NFL games have clearly recorded results, a stable schedule structure, and a large amount of historical information. Even at the schedule level, useful pregame signals exist, including team form, recent scoring trends, days of rest, division-game status, and market expectations from betting lines. This project belongs in the football analytics domain because it turns those measurable pregame signals into a reproducible prediction system rather than relying only on narrative, reputation, or intuition.

### Background Reading FIX

The `background_reading/` folder contains readings that help explain the football analytics context of this project.

| Index | Title | Brief description | File in folder |
|---|---|---|---|
| 1 | Predicting the Outcome of NFL Games Using Logistic Regression | Honors thesis focused directly on NFL game outcome prediction and logistic-regression model framing | `background_reading/01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf` |
| 2 | Modeling NFL Football Outcomes | Paper discussing statistical models for NFL outcome prediction | `background_reading/02_modeling_nfl_football_outcomes.pdf` |
| 3 | The Effect of Attendance on Home Field Advantage in the NFL | Study of home-field effects and how attendance relates to them | `background_reading/03_nfl_home_field_advantage.pdf` |
| 4 | Is the NFL Betting Market Efficient? | Economics paper on whether NFL betting prices are efficient | `background_reading/04_nfl_betting_market_efficiency.pdf` |
| 5 | Assessing the Convergence of the Elo Ranking Model | Paper on Elo-model convergence and ranking stability | `background_reading/05_assessing_the_convergence_of_the_elo_ranking_model.pdf` |

<br>

## Data Creation

### Provenance

The raw data for this project were obtained from the nflverse ecosystem using the `nflreadpy` Python package. I used `nflreadpy` to access NFL schedules and team reference information, then filtered those source data to completed regular-season games across multiple seasons. The package provided the original source data, while all cleaning, filtering, reshaping, feature engineering, relational structuring, DuckDB loading, querying, and modeling steps were performed by my own code.

After loading the source data, I transformed them into a relational secondary dataset with four linked tables. First, I created a `teams` table that stores team identifiers and basic metadata. Second, I created a `games` table with one row per completed regular-season NFL game, including raw schedule-level context and pregame market variables when available. Third, I created a `team_games` table with one row per team per game so that leakage-safe rolling features could be computed separately for each team. Fourth, I created a `matchups` table with one row per game that combines home-team and away-team pregame features into a final modeling dataset. This makes the project dataset a reproducible secondary data product derived from the original nflverse schedule and team information.

### Data Creation Code FIX

The code below shows the python file used to create the secondary dataset and support the project pipeline.

| File | What it does | Path |
|---|---|---|
| `pipeline/build_project_tables.py` | Loads raw data from `nflreadpy`, filters to completed regular-season games, creates the `teams`, `games`, `team_games`, and `matchups` tables, loads them into DuckDB, and exports the final tables as parquet files | [pipeline/build_project_tables.py](pipeline/build_project_tables.py) |

### Bias Identification

Bias can enter this dataset through both the source data and the chosen predictors. Team outcomes are influenced by injuries, quarterback availability, coaching changes, travel burden, weather, and many other contextual factors that are not fully captured in schedule-level data. As a result, even a well-constructed model may miss important game-specific information.

Another important source of bias is temporal leakage. If any feature accidentally uses information from the game being predicted or from later games, the model would appear stronger than it really is. There is also a selection decision built into the project because it focuses only on completed NFL regular-season games. That is appropriate for the stated prediction target, but it means the model is not designed for playoff forecasting, college football, or other sports settings.

Finally, market-based features such as spread and moneyline are powerful predictors, but they reflect the betting market’s aggregated expectations rather than purely team-level football performance. That is useful for prediction, but it means the project partly studies how well public pregame expectations align with outcomes, not just how well rolling team statistics do on their own.

### Bias Mitigation

To reduce bias, all team-level rolling features are constructed using only information from games that occurred **before** the game being predicted. That means pregame win percentage, scoring averages, point differential, and recent-form summaries are all leakage-safe. I also exclude non-regular-season games so that the dataset stays aligned with the project’s specific problem statement.

Bias is further reduced by using a relational design with saved intermediate tables, which makes the feature engineering process easier to inspect and audit. In addition, I evaluate the model using a **time-based split** rather than a random shuffle so that later seasons are held out as future-like validation and test data. That better reflects how a real forecasting system would be used and reduces the risk of hidden leakage across train/test boundaries.

### Rationale for Critical Decisions

A major critical decision was to structure the project as four linked tables because the assignment requires a relational dataset and because that structure makes the workflow much easier to understand. The `teams`, `games`, `team_games`, and `matchups` tables separate lookup data, raw game outcomes, team-level historical summaries, and final modeling inputs into clear steps. In particular, the `team_games` table is essential because it provides a safe way to compute rolling team features before each game.

Another important decision was to predict **home-team win probability** rather than exact score or betting spread. That makes the target easier to explain, easier to evaluate, and less noisy than score prediction. It also fits naturally with binary classification models such as logistic regression and random forest.

A third important decision was to combine rolling football-performance features with schedule and market context from the raw schedule data. The earlier version of the project relied mainly on rolling averages and recent form, but those signals alone were not strong enough. By adding pregame spread, moneyline-based implied probabilities, rest measures, divisional-game status, and weather context, the project better reflects the real information available before an NFL game begins.

Finally, I chose to compare logistic regression and random forest. Logistic regression provides an interpretable baseline for a binary target, while random forest offers a more flexible nonlinear model that can capture interactions and threshold effects among the pregame variables.

<br>

## Metadata

### Schema ER Diagram FIX

```text
teams
-----
team_id (pk)
team_name
team_nick
team_conf
team_division

games
-----
game_id (pk)
season
week
gameday
home_team (fk -> teams.team_id)
away_team (fk -> teams.team_id)
home_score
away_score
home_win
winner_team

team_games
----------
game_id (fk -> games.game_id)
team_id (fk -> teams.team_id)
opponent_team (fk -> teams.team_id)
season
week
gameday
is_home
points_for
points_against
win
games_played_before
cum_wins_before
cum_losses_before
cum_points_for_before
cum_points_against_before
pregame_win_pct
pregame_points_for_pg
pregame_points_against_pg
pregame_point_diff_pg
days_rest

matchups
--------
game_id (pk, fk -> games.game_id)
season
week
gameday
home_team (fk -> teams.team_id)
away_team (fk -> teams.team_id)
home_score
away_score
home_win
winner_team
home_pregame_win_pct
away_pregame_win_pct
home_pregame_points_for_pg
away_pregame_points_for_pg
home_pregame_points_against_pg
away_pregame_points_against_pg
home_pregame_point_diff_pg
away_pregame_point_diff_pg
home_days_rest
away_days_rest
pregame_win_pct_diff
pregame_points_for_pg_diff
pregame_points_against_pg_diff
pregame_point_diff_pg_diff
days_rest_diff
target_home_win
```

### Data Table FIX

| Table | Description | CSV file |
|---|---|---|
| `teams` | Team reference table containing identifiers and metadata | `data/teams.csv` |
| `games` | One row per completed NFL regular-season game | `data/games.csv` |
| `team_games` | One row per team per game with rolling leakage-safe pregame features | `data/team_games.csv` |
| `matchups` | One row per game for final prediction modeling | `data/matchups.csv` |

### Data Dictionary

#### `teams`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `team_id` | string | Team abbreviation used as key | `BAL` |
| `team_name` | string | Full team name | `Baltimore Ravens` |
| `team_nick` | string | Team nickname | `Ravens` |
| `team_conf` | string | Conference | `AFC` |
| `team_division` | string | Division | `AFC North` |

#### `games`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Unique game identifier | `2023_W01_HOU_at_BAL` |
| `season` | integer | NFL season year | `2023` |
| `week` | integer | Regular-season week number | `1` |
| `gameday` | date | Date of game | `2023-09-10` |
| `home_team` | string | Home team ID | `BAL` |
| `away_team` | string | Away team ID | `HOU` |
| `home_score` | integer | Home final score | `25` |
| `away_score` | integer | Away final score | `9` |
| `home_win` | integer | 1 if home team won, else 0 | `1` |
| `winner_team` | string | Winner team ID | `BAL` |

#### `team_games`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Foreign key to game | `2023_W01_HOU_at_BAL` |
| `team_id` | string | Team for the row | `BAL` |
| `opponent_team` | string | Opponent team ID | `HOU` |
| `season` | integer | Season year | `2023` |
| `week` | integer | Week number | `1` |
| `gameday` | date | Game date | `2023-09-10` |
| `is_home` | integer | 1 if home, 0 if away | `1` |
| `points_for` | integer | Points scored by team | `25` |
| `points_against` | integer | Points allowed by team | `9` |
| `win` | integer | 1 if team won, else 0 | `1` |
| `games_played_before` | integer | Number of prior games entering this game | `0` |
| `cum_wins_before` | integer | Prior cumulative wins | `0` |
| `cum_losses_before` | integer | Prior cumulative losses | `0` |
| `cum_points_for_before` | integer | Prior cumulative points scored | `0` |
| `cum_points_against_before` | integer | Prior cumulative points allowed | `0` |
| `pregame_win_pct` | float | Win percentage before this game | `0.500` |
| `pregame_points_for_pg` | float | Prior scoring average | `0.0` |
| `pregame_points_against_pg` | float | Prior allowed average | `0.0` |
| `pregame_point_diff_pg` | float | Prior point differential average | `0.0` |
| `days_rest` | float | Days since previous game | `7.0` |

#### `matchups`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Unique game identifier | `2023_W01_HOU_at_BAL` |
| `season` | integer | Season year | `2023` |
| `week` | integer | Week number | `1` |
| `gameday` | date | Game day | `2023-09-10` |
| `home_team` | string | Home team ID | `BAL` |
| `away_team` | string | Away team ID | `HOU` |
| `home_score` | integer | Home final score | `25` |
| `away_score` | integer | Away final score | `9` |
| `home_win` | integer | 1 if home team won, else 0 | `1` |
| `winner_team` | string | Winner team ID | `BAL` |
| `home_pregame_win_pct` | float | Home-team win percentage entering game | `0.500` |
| `away_pregame_win_pct` | float | Away-team win percentage entering game | `0.500` |
| `home_pregame_points_for_pg` | float | Home-team prior scoring average | `0.0` |
| `away_pregame_points_for_pg` | float | Away-team prior scoring average | `0.0` |
| `home_pregame_points_against_pg` | float | Home-team prior points allowed average | `0.0` |
| `away_pregame_points_against_pg` | float | Away-team prior points allowed average | `0.0` |
| `home_pregame_point_diff_pg` | float | Home-team prior point differential average | `0.0` |
| `away_pregame_point_diff_pg` | float | Away-team prior point differential average | `0.0` |
| `home_days_rest` | float | Home-team days since previous game | `7.0` |
| `away_days_rest` | float | Away-team days since previous game | `7.0` |
| `pregame_win_pct_diff` | float | Home minus away pregame win percentage | `0.0` |
| `pregame_points_for_pg_diff` | float | Home minus away prior scoring average | `0.0` |
| `pregame_points_against_pg_diff` | float | Home minus away prior allowed average | `0.0` |
| `pregame_point_diff_pg_diff` | float | Home minus away prior point differential average | `0.0` |
| `days_rest_diff` | float | Home minus away days of rest | `0.0` |
| `target_home_win` | integer | Final prediction target | `1` |

### Quantification of Uncertainty for Numerical Features

| Numerical feature | Source of uncertainty | Quantitative discussion |
|---|---|---|
| `home_score / away_score` | Officially recorded outcomes | These are observed final values, so measurement uncertainty is very low, but they are not prediction inputs in the final model because they occur after kickoff |
| `pregame_win_pct` | Small-sample instability early in the season | When `games_played_before` is small, one result changes the estimate sharply; after 1 prior game the value can only be 0 or 1 |
| `pregame_points_for_pg` | Sampling variation across prior games | Early-season estimates are noisier because one unusually high- or low-scoring game strongly changes the average |
| `pregame_points_against_pg` | Sampling variation across prior games | Same issue as above; defensive estimates stabilize only after more games are observed |
| `pregame_point_diff_pg` | Uncertainty inherited from two averages | Because it is based on prior points scored and prior points allowed, it compounds uncertainty from both values |
| `days_rest` | Exact from dates, incomplete as a fatigue proxy | The value is measured deterministically from game dates, but it does not capture travel difficulty, injury recovery, or practice intensity |
| `difference features` | Propagated uncertainty from both teams | Each home-minus-away feature combines uncertainty from two team-level estimates, especially in early weeks |
