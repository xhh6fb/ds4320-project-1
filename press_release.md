# new nfl data pipeline turns pregame team form into game-day predictions

## hook

nfl fans see predictions everywhere, but many of them rely on hidden assumptions or use information that would not actually be known before kickoff. this project builds a cleaner and more transparent alternative: a data pipeline that uses only pregame team form and scheduling information to estimate whether the home team will win.

## problem statement

predicting sports outcomes is harder than it looks. many postgame statistics explain a result only after the game is over, which makes them less useful for true forecasting. for nfl games, a fair prediction system has to rely only on information that would already exist before the teams take the field. that means carefully organizing historical data so that each game can be linked to the right prior information for both teams.

the specific problem in this project is predicting whether the home team will win an nfl regular-season game. this is a useful problem for sports analytics because it connects schedule effects, recent form, and home-field context to a concrete decision outcome.

## solution description

this project creates a relational dataset with four linked tables: teams, games, team-games, and matchups. those tables make it possible to compute pregame features such as prior win percentage, prior point differential per game, and days of rest without leaking information from the current game into the prediction.

the final pipeline then loads the data into duckdb, prepares a modeling table with sql, and uses logistic regression to estimate the probability of a home-team win. the result is a reproducible and explainable workflow that can be extended later with richer features such as betting lines, injury information, or travel adjustments.

## chart

![home win rate by pregame point differential gap](pipeline/figures/home_win_rate_by_gap.png)
