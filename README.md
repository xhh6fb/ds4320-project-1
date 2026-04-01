# DS 4320 Project 1: Predicting NFL Home-Team Win Probability

## Executive Summary

This repository contains my DS 4320 Project on predicting NFL regular-season game outcomes using a relational secondary dataset built from nflverse data accessed through the `nflreadpy` Python package. The project refines the broad problem of predicting sports game outcomes into a specific pregame classification task: estimating the probability that the home team will win an NFL regular-season game using only information available before kickoff. The repository includes project documentation, a press release, background readings, a linked data folder, SQL queries, Python code for data creation, and a reproducible modeling pipeline in Python, Markdown, and SQL.

<br>

|---|---|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | FIX THIS [https://doi.org/10.1000/182](https://doi.org/10.1000/182) |
| Press Release | [FIX HEADER](press_release.md) |
| Data | FIX THIS [UVA OneDrive data folder](REPLACE-onedrive-link) |
| Pipeline | [Notebook](pipeline/project_1_pipeline.ipynb) & [Markdown](pipeline/project_1_pipeline.md)   |
| License | [MIT](LICENSE.md) |

<br>

## Problem Definition

### Initial General Problem & Refined Specific Problem Statement

**Initial General Problem:** Predicting sports game outcomes.

**Refined Specific Problem Statement:** Develop a data-driven system that predicts whether the home team will win an NFL regular-season game using only pregame information derived from historical team form, recent team form, schedule context, and pregame market information. The final prediction target is a binary variable equal to 1 when the home team wins and 0 otherwise.

### Rationale for Refinement

I refined the general problem of predicting sports game outcomes into predicting NFL regular-season home-team wins because the original problem was too broad to support a clear relational dataset and a focused predictive pipeline. “Sports games” could refer to many different leagues and sports with different rules, schedules, scoring systems, and useful statistics, which would make it difficult to define consistent variables, build a coherent relational model, and explain what exactly the model is predicting.

By narrowing the project topic simply to NFL regular-season games, the problem becomes much more realistic and better structured for analysis. The NFL has a standardized season format, a manageable number of teams, clearly recorded game outcomes, and lots of historical data that can be used to construct pregame features. Limiting the project to pregame prediction also keeps the project meaningful, since the goal is to estimate the likely outcome before it begins and not to explain a result in hindsight after the game is over. This refinement makes the project more rigorous, more reproducible, and better aligned with the kind of forecasting problem that teams, analysts, and fans actually care about.

### Motivation

I chose this project because football is one of the subjects I care most about, and I want to build toward a future career in the NFL. More specifically, I hope to work toward becoming a coach in the league, and ideally even the first female head coach in the NFL. Because of that, this project is not just an academic exercise for me. It connects directly to something I genuinely care about and want to understand at a deeper level.

This project also matters to me because football is often discussed emotionally and based on intuition, but successful teams increasingly use analytics to support preparation, game planning, player evaluation, and decision-making. I want to understand that side of the sport better. Building a project around NFL game prediction would help me learn which factors actually matter when evaluating teams before a game, how to think critically about performance rather than relying only on record or reputation, and how data can support strategic football thinking. It would also help me practice turning a personal interest of mine into a structured data science problem with a reproducible solution.

### Press Release Headline & Link FIX

**Headline:** *New NFL Data Pipeline Turns Pregame Team Form Into Game-day Predictions*  
**Link to Press Release:** [press_release.md](press_release.md)

## Domain Exposition

### Terminology

| Term | Meaning in this project | Why it matters |
|---|---|---|
| NFL | National Football League | The league this project focuses on |
| regular-season game | An NFL game that is part of the standard season schedule, excluding playoffs | Keeps the prediction target consistent |
| home-team win | A binary target equal to 1 if the home team scores more points than the away team | This is the prediction outcome |
| target variable | The value the model is trying to predict | Here it is whether the home team wins |
| feature | An input variable used by the model | These are the pregame predictors |
| pregame feature | A variable computed only from information available before kickoff | Prevents data leakage |
| leakage | Using information from the current game or future games to predict the outcome | Would make the model misleadingly strong |
| team reference table | A lookup table with team identifiers and metadata | Supports relational joins and documentation |
| game table | One row per completed NFL regular-season game | Core fact table for outcomes |
| team-game table | One row per team per game | Allows rolling pregame features to be calculated |
| matchup table | One row per game with home and away pregame features | Final modeling table |
| pregame win percentage | A team’s win rate before the current game | A simple measure of prior form |
| pregame points for per game | A team’s average prior points scored entering the game | Captures offensive form |
| pregame points against per game | A team’s average prior points allowed entering the game | Captures defensive form |
| pregame point differential per game | Prior average points scored minus prior average points allowed | Summarizes overall prior strength |
| days of rest | Days since the team’s previous game | Helps capture scheduling effects |
| relational model | A data design built from linked tables with keys and joins | Required by the project rubric |
| DuckDB | An in-process analytical database used here for loading CSV files and querying them with SQL | Satisfies the pipeline requirement and supports reproducibility |
| logistic regression | A classification model that estimates the probability of a binary outcome | Baseline model for home-win prediction |
| random forest | An ensemble machine-learning classification model built from many decision trees | More flexible comparison model |
| ROC AUC | A metric that measures how well predicted probabilities rank positive cases over negative cases | Useful beyond raw accuracy |
| confusion matrix | A table showing true positives, false positives, true negatives, and false negatives | Helps interpret model behavior |

### Domain

This project lives in the domain of **sports analytics**, and more specifically in football analytics and predictive modeling. Sports analytics uses data to better understand team performance, player performance, strategy, and likely future outcomes. Within that larger domain, football analytics focuses on measuring what leads to success in football games and how team strength can be estimated more accurately than by wins and losses alone.

Predicting NFL game outcomes is a natural problem in this domain because NFL games have clear results, a fixed schedule structure, and a large amount of recorded historical information. Even simple game-level measures such as prior win percentage, scoring margin, and rest can reveal useful signals about team strength. This project belongs in the football analytics domain because it turns those measurable patterns into a reproducible pregame forecasting system. Instead of relying entirely on narrative or opinion, the project uses historical data to estimate how likely the home team is to win before a game begins.

### Background Reading FIX

https://drive.google.com/drive/folders/1DmNz53sf4Q_i9nTIDYbz9nB2Qzlk0tSc?usp=sharing  
The `background_reading/` folder contains readings that help explain the football analytics context of this project.

| Index | Title | Brief description | File in folder |
|---|---|---|---|
| 1 | Predicting the Outcome of NFL Games Using Logistic Regression | Honors thesis focused directly on NFL game outcome prediction and logistic-regression model framing | `background_reading/01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf` |
| 2 | Modeling NFL Football Outcomes | Paper discussing statistical models for NFL outcome prediction | `background_reading/02_modeling_nfl_football_outcomes.pdf` |
| 3 | The Effect of Attendance on Home Field Advantage in the NFL | Study of home-field effects and how attendance relates to them | `background_reading/03_nfl_home_field_advantage.pdf` |
| 4 | Is the NFL Betting Market Efficient? | Economics paper on whether NFL betting prices are efficient | `background_reading/04_nfl_betting_market_efficiency.pdf` |
| 5 | Elo Model Convergence | Paper comparing Elo-style prediction ideas in sports forecasting context | `background_reading/05_elo_model_convergence_nfl.pdf` |

## Data Creation

### Provenance

I obtained the raw data for this project from the nflverse ecosystem using the `nflreadpy` Python package. I used team-level reference data and NFL schedule/results data to create a secondary dataset for modeling. The package supplied the original source data, while all cleaning, filtering, reshaping, feature engineering, relational structuring, SQL loading, and modeling steps were performed by my own code.

I filtered the schedule data to completed **regular-season** NFL games from selected seasons, then transformed those raw data into a relational dataset with four linked tables. First, I created a `teams` table for team identifiers and metadata. Second, I created a `games` table with one row per completed regular-season game. Third, I created a `team_games` table with one row per team per game so that rolling pregame summaries could be calculated separately for each team. Fourth, I created a `matchups` table with one row per game containing the home and away pregame features plus the binary target variable. This dataset is therefore a **secondary data product** derived from the original nflverse releases.

### Data Creation Code FIX

The code below shows the files used to create the secondary dataset and support the project pipeline.

| File | What it does | Link/path |
|---|---|---|
| `pipeline/build_project_tables.py` | Loads raw data from `nflreadpy`, filters to completed regular-season games, creates the `teams`, `games`, `team_games`, and `matchups` tables, and saves them as CSV and parquet files | [pipeline/build_project_tables.py](pipeline/build_project_tables.py) |
| `pipeline/query_project_data.sql` | Contains example SQL queries used to inspect and summarize the relational tables after loading them into DuckDB | [pipeline/query_project_data.sql](pipeline/query_project_data.sql) |
| `pipeline/project_1_pipeline.ipynb` | Full notebook pipeline that loads the saved tables into DuckDB, runs SQL queries, trains models, evaluates performance, and creates visualizations | [pipeline/project_1_pipeline.ipynb](pipeline/project_1_pipeline.ipynb) |
| `pipeline/project_1_pipeline.md` | Markdown export of the notebook pipeline | [pipeline/project_1_pipeline.md](pipeline/project_1_pipeline.md) |

### Bias Identification

Bias can enter this dataset through both the source data and the simplified feature choices. Team outcomes are affected by injuries, quarterback changes, weather, coaching adjustments, travel burden, and opponent quality, but not all of those factors are represented directly in a schedule-based dataset. As a result, the model may assign too much importance to simple prior scoring or win-percentage summaries while missing important contextual factors that influence real NFL games.

Another major source of bias is temporal leakage. If any feature accidentally includes information from the current game or from future games, the model would appear stronger than it really is. There is also a selection decision built into the project because it only analyzes regular-season NFL games rather than playoff games, college football games, or other football competitions. That is appropriate for the project goal, but it means the resulting model is not automatically generalizable outside that setting.

### Bias Mitigation

To reduce bias, I constructed all rolling team features using only information from games that happened **before** the game being predicted. That means pregame win percentage, pregame points scored per game, pregame points allowed per game, and pregame point differential per game are all leakage-safe summaries. I also exclude non-regular-season games so that the dataset remains aligned with the stated prediction target.

Bias is also reduced by using a relational design with saved intermediate tables, which makes the feature-engineering process easier to inspect and audit. In addition, I evaluate the model using a **time-based split** instead of a random shuffle so that later seasons are held out as future-like test data. This can better reflect how a real forecasting system would be used and reduces the risk of information bleeding across the train/test boundary.

### Rationale for Critical Decisions

A major critical decision was to structure the data into four linked tables because the project rubric requires the dataset to be constructed using the relational model with at least four tables. I chose `teams`, `games`, `team_games`, and `matchups` because that structure both satisfies the rubric and cleanly supports pregame feature engineering. In particular, the `team_games` table is essential because it provides a safe way to compute team-level rolling summaries before each game.

Another important decision was to predict **home-team win probability** rather than exact score or betting spread. That makes the target easier to explain, easier to evaluate, and less noisy. I also decided to begin with transparent pregame schedule-derived features rather than more advanced inputs that would require play-by-play engineering or external injury data. That keeps the project interpretable and reproducible while still producing a meaningful football prediction pipeline.

A third critical decision was to use both a logistic-regression baseline and a random-forest comparison model. Logistic regression provides a clear and interpretable starting point for a binary outcome, while random forest offers a more flexible machine-learning comparison that can capture nonlinear interactions among the pregame features. This balances simplicity, interpretability, and predictive strength.

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
