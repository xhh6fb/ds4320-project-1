-- query 1: season-level home win rates and scoring summaries

SELECT
    season,
    COUNT(*) AS games_n,
    AVG(home_win) AS home_win_rate,
    AVG(home_score) AS avg_home_score,
    AVG(away_score) AS avg_away_score
FROM games
GROUP BY season
ORDER BY season;

-- query 2: weekly matchup summary from the final modeling table

SELECT
    season,
    week,
    AVG(pregame_point_diff_pg_diff) AS avg_point_diff_gap,
    AVG(target_home_win) AS avg_home_win_rate
FROM matchups
GROUP BY season, week
ORDER BY season, week;

-- query 3: relationship between recent-form advantage and home-team win rate

SELECT
    CASE
        WHEN last3_point_diff_pg_diff < 0 THEN 'home_recent_form_worse'
        WHEN last3_point_diff_pg_diff = 0 THEN 'recent_form_equal'
        WHEN last3_point_diff_pg_diff > 0 THEN 'home_recent_form_better'
    END AS recent_form_group,
    COUNT(*) AS games_n,
    AVG(target_home_win) AS home_win_rate
FROM matchups
GROUP BY recent_form_group
ORDER BY games_n DESC;

-- query 4: relationship between rest differential and home-team win rate

SELECT
    CASE
        WHEN days_rest_diff < 0 THEN 'home_less_rest'
        WHEN days_rest_diff = 0 THEN 'equal_rest'
        WHEN days_rest_diff > 0 THEN 'home_more_rest'
    END AS rest_diff_group,
    COUNT(*) AS games_n,
    AVG(target_home_win) AS home_win_rate
FROM matchups
GROUP BY rest_diff_group
ORDER BY games_n DESC;
