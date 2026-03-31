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

this project lives in the domain of sports analytics, specifically predictive modeling for american football. nfl game outcomes are shaped by team strength, recent form, scheduling, and home-field effects, but prediction must be done carefully because postgame statistics can accidentally leak information into the model. a key challenge in this domain is designing features that are both predictive and genuinely pregame. that makes the problem a good match for a relational dataset with explicit time ordering, because the data must be organized so that each game can be joined to the correct prior information for both teams.

### Background Reading

save copies of these files into the `background_reading/` folder. if a source is a webpage instead of a downloadable pdf, save it as a pdf from your browser.

| index | title | brief description | file in folder |
|---|---|---|---|
| 1 | predicting the outcome of nfl games using logistic regression | honors thesis focused directly on nfl game outcome prediction and model framing | `01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf` |
| 2 | modeling nfl football outcomes | paper discussing statistical models for nfl outcome prediction | `02_modeling_nfl_football_outcomes.pdf` |
| 3 | the effect of attendance on home field advantage in the nfl | study of home-field effects and how attendance relates to them | `03_nfl_home_field_advantage.pdf` |
| 4 | is the nfl betting market efficient? | economics paper on whether nfl betting prices are efficient | `04_nfl_betting_market_efficiency.pdf` |
| 5 | elo model convergence | paper comparing elo-style prediction ideas, including nfl-related forecasting context | `05_elo_model_convergence_nfl.pdf` |

### Background Reading Links

1. [predicting the outcome of nfl games using logistic regression](https://scholars.unh.edu/cgi/viewcontent.cgi?article=1472&context=honors)  
2. [modeling nfl football outcomes](https://www.researchgate.net/publication/368564164_MODELING_NFL_FOOTBALL_OUTCOMES)  
3. [the effect of attendance on home field advantage in the nfl](https://surface.syr.edu/context/sportmanagement/article/1057/viewcontent/NFL_home_field_advantage_Preprint.pdf)  
4. [is the nfl betting market efficient?](https://econ.berkeley.edu/sites/default/files/Kuper.pdf)  
5. [elo model convergence](https://myweb.ecu.edu/robbinst/PDFs/Elo%20Model%20Convergence%20Final.pdf)

## Data Creation

### Provenance

the raw data for this project were obtained from the nflverse ecosystem using the `nflreadpy` python package. `nflreadpy` provides programmatic access to nflverse releases, including team metadata and schedules. i used `load_teams()` to obtain the team reference data and `load_schedules()` to obtain regular-season schedules and results across multiple seasons.

after loading the source data, i transformed them into a relational secondary dataset with four linked tables. first, i created a `teams` table to store team identifiers and descriptive metadata. second, i created a `games` table with one row per completed regular-season game. third, i created a `team_games` table with one row per team per game so that rolling prior features could be calculated separately for each team. fourth, i created a `matchups` table with one row per game containing the pregame home and away features and the binary target variable.

### Data Creation Code

| file | what it does | link/path |
|---|---|---|
| `pipeline/project_1_pipeline.ipynb` | full notebook that loads the source data, creates the four relational tables, loads them into duckdb, runs queries, fits the model, and creates visualizations | `pipeline/project_1_pipeline.ipynb` |
| `pipeline/project_1_pipeline.md` | markdown export of the notebook pipeline | `pipeline/project_1_pipeline.md` |

### Bias Identification

bias can enter this dataset through both source coverage and modeling simplification. the source data do not capture every pregame factor that may affect nfl outcomes, such as injuries, late roster changes, weather severity, travel fatigue, or quarterback uncertainty. because the model uses mainly prior results and rest patterns, it may over-credit teams whose past outcomes were shaped by unusually weak schedules or temporary conditions.

another important source of bias is temporal leakage. if features were built using information from the current game or future games, model performance would be falsely inflated. there is also selection bias because the project focuses only on completed regular-season games and excludes playoff games and other football contexts.

### Bias Mitigation

to reduce bias, all rolling features are constructed using only games that occurred earlier in the same season for the same team. this preserves temporal ordering and limits leakage. i also keep the modeling target narrow and explicit: completed nfl regular-season games only. that avoids blending multiple competition types with different incentives and structures.

i also make the dataset more auditable by saving intermediate relational tables rather than only a single final modeling table. this makes it easier to inspect how features were created and how uncertainty entered at each stage. finally, i evaluate the model using later seasons as the test set rather than random shuffling, which better reflects a real forecasting setting.

### Rationale for Critical Decisions

the most important design choice was to use a four-table relational structure instead of one flattened file. the project goal is not just prediction; it is to create a fully established secondary dataset using the relational model. separating teams, games, team-games, and matchups keeps the data organized around meaningful entities and prevents the feature-engineering logic from being hidden inside one opaque table.

another important judgment call was feature scope. for this first project version, i deliberately limited the feature set to interpretable pregame summaries: prior win percentage, prior average points scored, prior average points allowed, prior average point differential, and rest days. richer feature sets could improve performance, but starting with transparent, leakage-resistant features gives a more defensible baseline and makes the uncertainty in the analysis easier to explain.

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

### Data Dictionary: Table

ADD TABLE

### Data Dictionary: Quantification of Uncertainty

ADD TABLE
