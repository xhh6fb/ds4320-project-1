# ds 4320 project 1: pregame nfl win prediction

## executive summary

this repository contains my ds 4320 project 1 on predicting nfl regular-season game outcomes using a relational secondary dataset built from nflverse source data. the project refines the broad problem of predicting sports game outcomes into a specific pregame classification task: predicting whether the home team will win, using only information that would have been available before kickoff. the repository includes the project documentation, background readings, a separate press release, a linked data folder, and a reproducible pipeline in python, markdown, and sql.

**name:** Jolie Ng  
**netid:** xhh6fb 
**doi:** CREATE DOI
**press release:** [press_release.md](press_release.md)  
**data:** [uva onedrive data folder](REPLACE-onedrive-link)  
**pipeline:** [pipeline/project_1_pipeline.ipynb](pipeline/project_1_pipeline.ipynb)  
**license:** [mit license](LICENSE)

## problem definition

### initial general problem and refined specific problem statement

the initial general problem is **predicting sports game outcomes**.

the refined specific problem is: **predict whether the home team will win an nfl regular-season game using only pregame information available before kickoff, with features built from prior team performance and rest patterns.**

### rationale for the refinement

the original problem is too broad because sports outcomes vary widely across leagues, scoring systems, roster structures, and data availability. narrowing the problem to nfl regular-season games creates a more coherent domain with a consistent schedule structure, standardized team identities, and widely available historical data. it also makes the modeling task more realistic because the prediction target can be clearly defined as a binary outcome: home-team win or non-win.

this refined version is also a better fit for the relational-model requirement. nfl data naturally separates into multiple connected entities such as teams, games, team-game records, and matchup-level features. that makes it possible to build a dataset that is both analytically useful and structurally aligned with the project goal of creating a fully established secondary dataset using the relational model.

### motivation for the project

there are practical and analytical reasons to study pregame nfl prediction. in sports analytics, teams, media, and fans all care about what factors matter before a game starts, not only what explains the result afterward. a pregame model is useful because it forces careful attention to temporal ordering and leakage: the features must be based only on information known before the game. this creates a more honest predictive setting than many retrospective sports analyses.

this project is also motivating because it connects ideas from data engineering, database design, and machine learning. the final goal is not just to fit a model, but to show how raw source data can be transformed into a reusable relational dataset with explicit metadata, uncertainty discussion, and reproducible analytical steps.

### headline of press release and link

**headline:** *new nfl data pipeline turns pregame team form into game-day predictions*  
see [press_release.md](press_release.md)

## domain exposition

### terminology

| term | meaning in this project | why it matters |
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

### domain paragraph

this project lives in the domain of sports analytics, specifically predictive modeling for american football. nfl game outcomes are shaped by team strength, recent form, scheduling, and home-field effects, but prediction must be done carefully because postgame statistics can accidentally leak information into the model. a key challenge in this domain is designing features that are both predictive and genuinely pregame. that makes the problem a good match for a relational dataset with explicit time ordering, because the data must be organized so that each game can be joined to the correct prior information for both teams.

### background reading

save copies of these files into the `background_reading/` folder. if a source is a webpage instead of a downloadable pdf, save it as a pdf from your browser.

| index | title | brief description | file in folder |
|---|---|---|---|
| 1 | predicting the outcome of nfl games using logistic regression | honors thesis focused directly on nfl game outcome prediction and model framing | `01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf` |
| 2 | modeling nfl football outcomes | paper discussing statistical models for nfl outcome prediction | `02_modeling_nfl_football_outcomes.pdf` |
| 3 | the effect of attendance on home field advantage in the nfl | study of home-field effects and how attendance relates to them | `03_nfl_home_field_advantage.pdf` |
| 4 | is the nfl betting market efficient? | economics paper on whether nfl betting prices are efficient | `04_nfl_betting_market_efficiency.pdf` |
| 5 | elo model convergence | paper comparing elo-style prediction ideas, including nfl-related forecasting context | `05_elo_model_convergence_nfl.pdf` |

### background reading links used to collect files

1. [predicting the outcome of nfl games using logistic regression](https://scholars.unh.edu/cgi/viewcontent.cgi?article=1472&context=honors)  
2. [modeling nfl football outcomes](https://www.researchgate.net/publication/368564164_MODELING_NFL_FOOTBALL_OUTCOMES)  
3. [the effect of attendance on home field advantage in the nfl](https://surface.syr.edu/context/sportmanagement/article/1057/viewcontent/NFL_home_field_advantage_Preprint.pdf)  
4. [is the nfl betting market efficient?](https://econ.berkeley.edu/sites/default/files/Kuper.pdf)  
5. [elo model convergence](https://myweb.ecu.edu/robbinst/PDFs/Elo%20Model%20Convergence%20Final.pdf)

## data creation

### provenance

the raw data for this project were obtained from the nflverse ecosystem using the `nflreadpy` python package. `nflreadpy` provides programmatic access to nflverse releases, including team metadata and schedules. i used `load_teams()` to obtain the team reference data and `load_schedules()` to obtain regular-season schedules and results across multiple seasons.

after loading the source data, i transformed them into a relational secondary dataset with four linked tables. first, i created a `teams` table to store team identifiers and descriptive metadata. second, i created a `games` table with one row per completed regular-season game. third, i created a `team_games` table with one row per team per game so that rolling prior features could be calculated separately for each team. fourth, i created a `matchups` table with one row per game containing the pregame home and away features and the binary target variable.

### code table

| file | what it does | link/path |
|---|---|---|
| `pipeline/project_1_pipeline.ipynb` | full notebook that loads the source data, creates the four relational tables, loads them into duckdb, runs queries, fits the model, and creates visualizations | `pipeline/project_1_pipeline.ipynb` |
| `pipeline/project_1_pipeline.md` | markdown export of the notebook pipeline | `pipeline/project_1_pipeline.md` |

### bias identification

bias can enter this dataset through both source coverage and modeling simplification. the source data do not capture every pregame factor that may affect nfl outcomes, such as injuries, late roster changes, weather severity, travel fatigue, or quarterback uncertainty. because the model uses mainly prior results and rest patterns, it may over-credit teams whose past outcomes were shaped by unusually weak schedules or temporary conditions.

another important source of bias is temporal leakage. if features were built using information from the current game or future games, model performance would be falsely inflated. there is also selection bias because the project focuses only on completed regular-season games and excludes playoff games and other football contexts.

### bias mitigation

to reduce bias, all rolling features are constructed using only games that occurred earlier in the same season for the same team. this preserves temporal ordering and limits leakage. i also keep the modeling target narrow and explicit: completed nfl regular-season games only. that avoids blending multiple competition types with different incentives and structures.

i also make the dataset more auditable by saving intermediate relational tables rather than only a single final modeling table. this makes it easier to inspect how features were created and how uncertainty entered at each stage. finally, i evaluate the model using later seasons as the test set rather than random shuffling, which better reflects a real forecasting setting.

### rationale for critical decisions

the most important design choice was to use a four-table relational structure instead of one flattened file. the project goal is not just prediction; it is to create a fully established secondary dataset using the relational model. separating teams, games, team-games, and matchups keeps the data organized around meaningful entities and prevents the feature-engineering logic from being hidden inside one opaque table.

another important judgment call was feature scope. for this first project version, i deliberately limited the feature set to interpretable pregame summaries: prior win percentage, prior average points scored, prior average points allowed, prior average point differential, and rest days. richer feature sets could improve performance, but starting with transparent, leakage-resistant features gives a more defensible baseline and makes the uncertainty in the analysis easier to explain.

## metadata

### schema er diagram at the logical level

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
