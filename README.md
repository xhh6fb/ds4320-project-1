# DS 4320 Project 1: Predicting NFL Home-Team Wins

## Executive Summary

this repository contains my ds 4320 project 1 on predicting nfl regular-season game outcomes using a relational secondary dataset built from nflverse source data. the project refines the broad problem of predicting sports game outcomes into a specific pregame classification task: predicting whether the home team will win, using only information that would have been available before kickoff. the repository includes the project documentation, background readings, a separate press release, a linked data folder, and a reproducible pipeline in python, markdown, and sql.

**Name:** Jolie Ng  
**NetID:** xhh6fb  
**DOI:** FIX THIS
**Press Release:** [press_release.md](press_release.md)  
**Data:** FIX THIS [UVA OneDrive data folder](REPLACE-onedrive-link)  
**Pipeline:** FIX THIS [pipeline/project_1_pipeline.ipynb](pipeline/project_1_pipeline.ipynb)  
**License:** [MIT license](LICENSE)

## Problem Definition

### Initial General Problem & Refined Specific Problem Statement

**Initial General Problem:** Predicting sports game win/loss outcomes.

**Refined Specific Problem Statement:** Develop a data-driven system that predicts the outcomes of NFL regular-season games by estimating which team is more likely to win based on historical team performance, opponent quality, game location, rest, scoring efficiency, turnover trends, and other football-related indicators available before kickoff. This refined problem focuses specifically on NFL regular-season games rather than all sports or even all football games. It limits the project to one league with a fixed structure, a consistent schedule format, and a large amount of historical data. It also clearly defines the target of the prediction: the winner of a game, or more generally each team's probability of winning before the game begins.

### Rationale for Refinement

I refined the general problem of predicting sports game outcomes into predicting NFL regular-season game outcomes because the original problem was too broad to be practical for a focused data science project. “Sports games” could include football, basketball, baseball, soccer, hockey, and many others, all of which have different rules, scoring systems, season structures, and statistical patterns. Trying to build one project that applies to all sports would make it hard to define useful variables, select a consistent dataset, or explain what “success” means in a clear way.

By narrowing the topic to the NFL, the project becomes much more realistic and better defined. The NFL has a standardized schedule structure, a manageable number of teams, a clear win/loss outcome for each game, and a strong existing analytics culture. The league also has many accessible team- and game-level statistics that can be used to study performance before a game takes place. In addition, football analytics increasingly uses advanced measures such as Expected Points Added (EPA), win probability, tracking-based metrics from Next Gen Stats, and opponent-adjusted efficiency measures such as DVOA. These metrics make it possible to move beyond simple win-loss records and better capture how strong a team actually is. That makes the NFL a strong and well-supported environment for a prediction project like this.

### Motivation

I chose this project because football is one of the subjects I care most about, and I want to build toward a future career in the NFL. More specifically, I hope to work toward becoming a coach in the league, and ideally even the first female head coach in the NFL. Because of that, this project is not just an academic exercise for me. It connects directly to something I genuinely care about and want to understand at a deeper level.

This project also matters to me because football is often discussed emotionally and based on intuition, but successful teams increasingly use analytics to support preparation, game planning, player evaluation, and decision-making. I want to understand that side of the sport better. Building a project around NFL game prediction would help me learn which factors actually matter when evaluating teams before a game, how to think critically about performance rather than relying only on record or reputation, and how data can support strategic football thinking. It would also help me practice turning a broad passion into a specific, structured problem that can be studied with data. In that sense, the project is both personally meaningful and professionally relevant.

Another reason I am motivated by this topic is that NFL analytics has become a major part of modern football culture. The NFL’s own football operations and analytics resources show how tracking data, advanced metrics, and data-driven evaluation are increasingly part of how the game is studied and understood. That makes this project feel current, useful, and connected to the real direction of the sport.

### Press Release Headline & Link

**Headline:** *New NFL Data Pipeline Turns Pregame Team Form Into Game-day Predictions*  
**Link to Press Release:** [press_release.md](press_release.md)

## Domain Exposition

### Terminology FIX

| Term | Meaning in this project | Why it matters |
|---|---|---|
| regular-season game | an nfl game that is part of the standard season schedule, excluding playoffs | keeps the prediction target consistent |
| home-team win | a binary target equal to 1 if the home team scores more points than the away team | this is the main prediction outcome |
| team reference table | a lookup table with team identifiers and metadata | supports relational joins and documentation |
| game table | one row per completed nfl game | core fact table for outcomes |
| team-game table | one row per team per game | lets us compute rolling prior features |
| matchup table | one row per game with home and away pregame features | final modeling table |
| pregame feature | a variable computed only from information available before kickoff | prevents leakage |
| leakage | using information from the current game or future games to predict the outcome | would make results misleading |
| pregame win percentage | team win rate before the current game | a simple measure of prior form |
| pregame point differential per game | prior average points scored minus prior average points allowed | summarizes overall prior strength |
| days of rest | days since the team’s previous game | helps capture scheduling effects |
| duckdb | an in-process analytical database used here for loading csv files and querying them with sql | satisfies the pipeline requirement and keeps the workflow reproducible |
| logistic regression | a machine-learning classification model for binary outcomes | the proof-of-concept model used in the pipeline |
| roc auc | a model-evaluation metric based on ranking predicted probabilities | helps assess classification quality beyond raw accuracy |

| Term | Meaning in this project | Why it matters |
|---|---|---|
| NFL | National Football League | The league this project focuses on |
| Outcome prediction | Predicting which team will win a game | This is the main goal of the project |
| Target variable | The value the model is trying to predict | In this case, the game winner or win probability |
| Feature | An input variable used by the model | Features may include points scored, turnovers, rest, or home field |
| Regular season | The main scheduled NFL games before the playoffs | Keeps the project scope specific and consistent |
| Home-field advantage | The advantage teams often have when playing at home | Home teams may perform differently than away teams |
| Strength of schedule | A measure of how difficult a team’s opponents have been | Helps prevent misleading comparisons between records |
| Point differential | Points scored minus points allowed | Often gives a stronger picture of team quality than record alone |
| Turnover differential | Takeaways minus giveaways | Turnovers often have a major effect on game outcomes |
| Offensive efficiency | How effectively a team scores or creates productive drives | Helps evaluate how strong an offense really is |
| Defensive efficiency | How effectively a team prevents scoring or productive drives | Helps evaluate how strong a defense really is |
| EPA | Expected Points Added | A play-based measure showing how much a play improves or hurts expected scoring |
| Win probability | The estimated chance a team wins from a certain game situation | Useful for understanding game state and prediction context |
| DVOA | Defense-adjusted Value Over Average | A play-based metric that adjusts performance for situation and opponent quality |
| SRS | Simple Rating System | A rating approach that accounts for scoring margin and strength of schedule; commonly referenced in football statistics resources |
| Predictive model | A statistical or machine learning system that estimates future outcomes from past data | This is the main analysis tool used in the project |
| Pregame information | Information known before kickoff | Important because the model should only use data available before the game starts |
| Advanced metrics | Statistics designed to capture performance more accurately than basic box-score totals | These often provide stronger insight than simple wins, yards, or points |

### Domain

This project lives in the domain of **sports analytics**, and more specifically in football analytics and predictive modeling. Sports analytics uses data to better understand team performance, player performance, strategy, and likely future outcomes. Within that larger domain, football analytics focuses on evaluating what leads to success in football games and how team strength can be measured more accurately than by basic record alone. Predicting NFL game outcomes is a natural problem in this domain because NFL games have clear results, well-documented statistics, and many measurable factors that may influence the final outcome.

The NFL is also especially relevant as a sports analytics domain because it now includes both traditional statistics and much more advanced forms of data collection. The league’s Next Gen Stats system uses tracking technology to capture player movement and support advanced analysis. Public football analytics also increasingly uses concepts such as EPA, win probability, and opponent-adjusted efficiency metrics like DVOA. These ideas reflect the fact that football performance is complicated: two teams may have the same record but be very different in actual strength, consistency, or efficiency. A project like this belongs in the football analytics domain because it uses measurable information to estimate future game outcomes in a way that could support preparation, evaluation, and decision-making.

### Background Reading FIX

https://drive.google.com/drive/folders/1DmNz53sf4Q_i9nTIDYbz9nB2Qzlk0tSc?usp=sharing  

| index | title | brief description | file in folder |
|---|---|---|---|
| 1 | predicting the outcome of nfl games using logistic regression | honors thesis focused directly on nfl game outcome prediction and model framing | `01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf` |
| 2 | modeling nfl football outcomes | paper discussing statistical models for nfl outcome prediction | `02_modeling_nfl_football_outcomes.pdf` |
| 3 | the effect of attendance on home field advantage in the nfl | study of home-field effects and how attendance relates to them | `03_nfl_home_field_advantage.pdf` |
| 4 | is the nfl betting market efficient? | economics paper on whether nfl betting prices are efficient | `04_nfl_betting_market_efficiency.pdf` |
| 5 | elo model convergence | paper comparing elo-style prediction ideas, including nfl-related forecasting context | `05_elo_model_convergence_nfl.pdf` |

| # | Title | Brief description |
|---|---|---|
| 1 | Footballonomics: The Anatomy of American Football; Evidence from 7 years of NFL game data | A research paper using several years of NFL data to study football outcomes and performance patterns. Useful for understanding how large-scale NFL data can support analytical conclusions. |
| 2 | nflWAR: A Reproducible Method for Offensive Player Evaluation in Football | A well-known football analytics paper that builds a reproducible framework for offensive player evaluation using expected points and win probability ideas. |
| 3 | Moving from Machine Learning to Statistics: the case of Expected Points in American football | A recent paper focused on expected points modeling in American football and the statistical issues involved in building these models well. |
| 4 | PEP: a tackle value measuring the prevention of expected points | A paper that develops a defensive football metric based on prevented expected points using tracking-style ideas and modeling. |
| 5 | NFL Play Prediction | A paper about predicting the outcome of an NFL play using multiple machine learning methods. Useful for understanding football prediction at the play level. |
| 6 | Deep Learning for In-Game NFL Predictions | A Stanford report that explores deep learning methods for predicting in-game NFL play outcomes. |
| 7 | Neural Network Models for Predicting NFL Play Outcomes | A Stanford paper that uses neural networks to predict play outcomes from pre-snap game state information. |
| 8 | NFL Score Difference Prediction with Markov Modeling | A Stanford report that models NFL score difference prediction using a Markov approach. Useful for thinking about game-level outcome prediction. |

## Data Creation

### Provenance FIX

the raw data for this project were obtained from the nflverse ecosystem using the `nflreadpy` python package. `nflreadpy` provides programmatic access to nflverse releases, including team metadata and schedules. i used `load_teams()` to obtain the team reference data and `load_schedules()` to obtain regular-season schedules and results across multiple seasons.

after loading the source data, i transformed them into a relational secondary dataset with four linked tables. first, i created a `teams` table to store team identifiers and descriptive metadata. second, i created a `games` table with one row per completed regular-season game. third, i created a `team_games` table with one row per team per game so that rolling prior features could be calculated separately for each team. fourth, i created a `matchups` table with one row per game containing the pregame home and away features and the binary target variable.

For the section, I continued my project on predicting NFL regular-season game outcomes. To create the data for this section, I used public NFL data available through Python and transformed those data into a relational structure suitable for database querying and predictive modeling.

The resulting project dataset contains four linked tables. First, I created a `teams` table to store team-level reference information. Second, I created a `games` table with one row per regular-season game. Third, I created a `team_games` table with one row per team per game so that rolling pregame summaries could be calculated. Fourth, I created a `matchups` table with one row per game containing pregame home and away features and the target variable. This follows the project rubric’s requirement that the dataset be constructed using the relational model with a minimum of four tables.

### Data Creation Code FIX

The raw NFL data were loaded from the `nflreadpy` package, but the relational project tables and proof-of-concept pipeline were created by my own code in this notebook. The package provided the source data; the cleaning, reshaping, feature engineering, table creation, SQL loading, querying, and modeling steps were my own work.

| file | what it does | link/path |
|---|---|---|
| `pipeline/project_1_pipeline.ipynb` | full notebook that loads the source data, creates the four relational tables, loads them into duckdb, runs queries, fits the model, and creates visualizations | `pipeline/project_1_pipeline.ipynb` |
| `pipeline/project_1_pipeline.md` | markdown export of the notebook pipeline | `pipeline/project_1_pipeline.md` |

### Bias Identification

Bias can enter this dataset through both the source data and the simplified feature choices. Team outcomes are affected by injuries, weather, quarterback changes, coaching, travel, and strength of opponent, but not all of these are represented in a schedule-based dataset. As a result, the model may assign too much importance to past scores or win percentage while missing important contextual factors.

Another major source of bias is temporal leakage. If any feature accidentally includes information from the current game or from future games, then the model would be misleadingly strong. There is also selection bias because the project only focuses on regular-season NFL games rather than all football competitions.

### Bias Mitigation

To reduce bias, I construct all rolling features using only information available before the game being predicted. This means that win percentage, prior scoring averages, and prior point differential are based only on earlier games for that team. I also exclude non-regular-season games to keep the target problem aligned with the stated project goal.

Bias is also reduced by using a relational design with saved intermediate tables, which makes the feature engineering process easier to inspect and debug. Finally, I evaluate the proof-of-concept model using later seasons as a test set rather than random shuffling, because that better reflects the real forecasting setting.

### Rationale for Critical Decisions

A major critical decision was to structure the data into four linked tables because the rubric requires a relational dataset with at least four tables. I created `teams`, `games`, `team_games`, and `matchups` to satisfy that requirement while also making the pipeline easier to understand and query.

Another important decision was to use transparent pregame schedule-derived features at this stage rather than richer but more complicated features. This reduces the risk of leakage, keeps the model explainable, and provides a reasonable proof of concept for the homework even though it is not yet the most advanced possible solution.

## Metadata

### Schema ER Diagram

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

### Data Table FIX/ADD LINK

| Table | Description | CSV file |
|---|---|---|
| teams | Team reference table | `teams.csv` |
| games | One row per NFL regular-season game | `games.csv` |
| team_games | One row per team per game with rolling pregame features | `team_games.csv` |
| matchups | One row per game for prediction modeling | `matchups.csv` |

### Data Dictionary

#### teams

| Feature | Data type | Description | Example |
|---|---|---|---|
| team_id | string | Team abbreviation used as key | BAL |
| team_name | string | Full team name | Baltimore Ravens |
| team_nick | string | Team nickname if available | Ravens |
| team_conf | string | Conference | AFC |
| team_division | string | Division | AFC North |

#### games

| Feature | Data type | Description | Example |
|---|---|---|---|
| game_id | string | Unique game identifier | 2023_01_BAL_HOU |
| season | integer | NFL season year | 2023 |
| week | integer | Regular-season week number | 1 |
| gameday | date | Date of game | 2023-09-10 |
| home_team | string | Home team ID | BAL |
| away_team | string | Away team ID | HOU |
| home_score | integer | Home final score | 25 |
| away_score | integer | Away final score | 9 |
| home_win | integer | 1 if home team won, else 0 | 1 |
| winner_team | string | Winner team ID | BAL |

#### team_games

| Feature | Data type | Description | Example |
|---|---|---|---|
| game_id | string | Foreign key to game | 2023_01_BAL_HOU |
| team_id | string | Team for the row | BAL |
| opponent_team | string | Opponent team ID | HOU |
| season | integer | Season year | 2023 |
| week | integer | Week number | 1 |
| gameday | date | Game date | 2023-09-10 |
| is_home | integer | 1 if home, 0 if away | 1 |
| points_for | integer | Points scored by team | 25 |
| points_against | integer | Points allowed by team | 9 |
| win | integer | 1 if team won | 1 |
| games_played_before | integer | Number of prior games | 0 |
| cum_wins_before | integer | Prior cumulative wins | 0 |
| cum_losses_before | integer | Prior cumulative losses | 0 |
| pregame_win_pct | float | Win percentage before this game | 0.500 |
| pregame_points_for_pg | float | Prior scoring average | 0.0 |
| pregame_points_against_pg | float | Prior allowed average | 0.0 |
| pregame_point_diff_pg | float | Prior point differential average | 0.0 |
| days_rest | float | Days since previous game | 7.0 |

#### matchups

| Feature | Data type | Description | Example |
|---|---|---|---|
| game_id | string | Unique game identifier | 2023_01_BAL_HOU |
| season | integer | Season year | 2023 |
| week | integer | Week number | 1 |
| gameday | date | Date | 2023-09-10 |
| home_team | string | Home team ID | BAL |
| away_team | string | Away team ID | HOU |
| home_pregame_win_pct | float | Home pregame win percentage | 0.500 |
| away_pregame_win_pct | float | Away pregame win percentage | 0.500 |
| home_pregame_points_for_pg | float | Home pregame scoring average | 0.0 |
| away_pregame_points_for_pg | float | Away pregame scoring average | 0.0 |
| home_pregame_points_against_pg | float | Home pregame allowed average | 0.0 |
| away_pregame_points_against_pg | float | Away pregame allowed average | 0.0 |
| home_pregame_point_diff_pg | float | Home pregame point differential average | 0.0 |
| away_pregame_point_diff_pg | float | Away pregame point differential average | 0.0 |
| home_days_rest | float | Home rest days | 7.0 |
| away_days_rest | float | Away rest days | 7.0 |
| pregame_win_pct_diff | float | Home minus away win percentage | 0.0 |
| pregame_points_for_pg_diff | float | Home minus away scoring average | 0.0 |
| pregame_points_against_pg_diff | float | Home minus away allowed average | 0.0 |
| pregame_point_diff_pg_diff | float | Home minus away point differential average | 0.0 |
| days_rest_diff | float | Home minus away rest days | 0.0 |
| target_home_win | integer | Prediction target | 1 |

### Quantification of Uncertainty *NOTE: quantitative/numeric estimate of uncertainty not given

| Numerical feature | Uncertainty discussion |
|---|---|
| home_score / away_score / points_for / points_against | These are official recorded scores, but their predictive meaning is still influenced by context not fully represented in the data. |
| pregame_win_pct | Deterministic from prior games, but unstable early in the season because it is based on few observations. |
| pregame_points_for_pg | Early-season averages are noisier because one unusual game can strongly affect the value. |
| pregame_points_against_pg | Same issue as above. |
| pregame_point_diff_pg | Inherits uncertainty from the underlying scoring averages. |
| days_rest | Exact from dates, but it does not capture travel quality, injury recovery, or hidden team circumstances. |
| difference features | Deterministic transformations of prior features, but their predictive usefulness may vary across seasons and situations. |
