# New NFL Data Pipeline Turns Pregame Team Form Into Game-day Predictions

## Hook

Every NFL week, people try to answer the same question before kickoff: who is more likely to win? Coaches, fans, commentators, and analysts all make predictions, but those predictions are often influenced by reputation, headlines, and emotion as much as by evidence. This project takes a more structured approach. It builds a clean NFL dataset from historical game information and uses that dataset to estimate whether the home team is likely to win before the game begins.

## Problem Statement

NFL game outcomes can be difficult to predict because team strength is not captured perfectly by record alone. Two teams may both be 6–3, for example, but one may have consistently won by large margins while the other has survived several close games. A team may also look stronger or weaker depending on scheduling, rest, and defensive performance, not just wins and losses.

That creates a real sports-analytics problem: if we want to estimate the likely outcome of a game before kickoff, what information should we use, and how should we organize it so that the prediction is both fair and reproducible? A strong solution must avoid “cheating” by accidentally using information from the game being predicted. It must also structure the data in a way that makes the relationships between teams, games, and historical summaries easy to understand.

This project addresses that problem by focusing on a specific and realistic question: **can we estimate the probability that the home team will win an NFL regular-season game using only information available before kickoff?** That is a narrower and more meaningful question than simply asking who won after the fact, because it mirrors how prediction would actually be used in practice.

## Solution Description

To solve this problem, I created a relational secondary dataset from NFL schedule and team-reference data accessed through the `nflreadpy` package. Instead of working with one flat spreadsheet, I structured the project into four linked tables:

1. **`teams`** — a reference table containing team identifiers and metadata  
2. **`games`** — one row per completed regular-season NFL game  
3. **`team_games`** — one row per team per game, used to calculate rolling historical summaries  
4. **`matchups`** — one row per game containing the final pregame modeling features  

This structure matters because it allows the project to compute features safely and clearly. For example, the `team_games` table makes it possible to calculate each team’s prior win percentage, prior scoring average, prior points allowed average, prior point differential, and days of rest **using only earlier games**. Those pregame features are then merged into the `matchups` table so that each game has one row containing the home team’s pregame form, the away team’s pregame form, and difference features such as home-minus-away point differential.

Once those features are created, the project uses machine-learning classification models to estimate the probability of a home-team win. I included both a logistic-regression baseline and a random-forest comparison model. Logistic regression provides a clear, interpretable starting point for binary prediction, while random forest allows the project to capture more flexible relationships among the pregame variables. The final result is a reproducible forecasting pipeline that transforms historical team form into a pregame game-outcome prediction.

Just as importantly, the solution is designed to be understandable. The model is not presented as magic. Instead, it is grounded in football ideas that make intuitive sense: teams with stronger prior point differential, better recent win rates, and more favorable rest situations may be more likely to win. The project therefore shows not only how to build an NFL prediction system, but also how to do so in a way that is organized, transparent, and suitable for future improvement.

## Why This Matters

This project matters because sports analytics is increasingly part of how football is understood and discussed. Teams and analysts are always looking for better ways to evaluate performance before a matchup happens, not just after the score is final. A pipeline like this helps show how historical team form can be transformed into a structured prediction problem rather than a vague opinion.

It also matters as a data-science exercise because it demonstrates the full workflow: obtaining source data, designing a relational secondary dataset, documenting provenance, writing SQL queries, training machine-learning models, and interpreting the results. In that sense, the project is not only about football. It is also about building a clear and reproducible analytical system from raw source data to final prediction output.

## Chart

![ROC Curve for Final NFL Prediction Model](figures/model_roc_curve.png)

This chart shows the performance of the final classification model on held-out test data. The ROC curve summarizes how well the model distinguishes home-team wins from home-team losses across many possible probability thresholds. A stronger curve indicates that the model is learning meaningful pregame signals from the relational dataset rather than guessing randomly.
