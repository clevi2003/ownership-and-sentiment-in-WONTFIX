# Sentiment Analysis Report

## Overview

This report summarizes an issue-level sentiment analysis of WONTFIX discussions versus comparison issues.
The primary inferential unit is the **issue**, while comment-level records are used mainly for temporal
trajectory analysis and descriptive summaries.

- Issues analyzed: **8114**
- Repositories represented: **22**
- Groups present: **comparison, wontfix**

## QA / coverage summary

| metric                                                   | value                     |
|:---------------------------------------------------------|:--------------------------|
| rq1_rows_raw                                             | 8116                      |
| comment_feature_rows_raw                                 | 22180                     |
| rq1_rows_normalized                                      | 8116                      |
| comment_feature_rows_normalized                          | 22180                     |
| rq1_duplicate_issue_keys                                 | 2                         |
| comment_duplicates                                       | 0                         |
| repos_represented                                        | 22                        |
| analysis_groups                                          | ['comparison', 'wontfix'] |
| rows_missing_core_sentiment_feature                      | 0                         |
| rows_missing_participation_covariates                    | 0                         |
| rows_not_marked_usable_for_rq1                           | -16228                    |
| issues_final                                             | 8114                      |
| repos_final                                              | 22                        |
| issues_analysis_set__comparison                          | 5888                      |
| issues_analysis_set__wontfix                             | 2226                      |
| issues_comparison_group__comparison                      | 5888                      |
| issues_comparison_group__wontfix                         | 2226                      |
| comment_count_median                                     | 2.0                       |
| comment_count_mean                                       | 2.733546955878728         |
| comments_with_text_count_median                          | 2.0                       |
| comments_with_text_count_mean                            | 2.733546955878728         |
| unique_commenter_count_median                            | 1.0                       |
| unique_commenter_count_mean                              | 1.8662805028346068        |
| zero_comment_issue_share                                 | 0.11400049297510476       |
| zero_text_comment_issue_share                            | 0.11400049297510476       |
| one_commenter_issue_share                                | 0.506531920138033         |
| missing_share__mean_comment_sentiment                    | 0.0                       |
| missing_share__median_comment_sentiment                  | 0.0                       |
| missing_share__min_comment_sentiment                     | 0.0                       |
| missing_share__max_comment_sentiment                     | 0.0                       |
| missing_share__std_comment_sentiment                     | 0.0                       |
| missing_share__comment_sentiment_change_late_minus_early | 0.0                       |
| missing_share__comment_sentiment_slope                   | 0.0                       |
| missing_share__negative_comment_share                    | 0.0                       |
| missing_share__positive_comment_share                    | 0.0                       |
| comments_final                                           | 22180                     |
| comment_missing_text_share                               | 0.0                       |

## Group descriptives

| analysis_set   | comparison_group   |   n_issues |   n_repos |   comment_count__mean |   comment_count__sd |   comment_count__median |   comment_count__q1 |   comment_count__q3 |   comments_with_text_count__mean |   comments_with_text_count__sd |   comments_with_text_count__median |   comments_with_text_count__q1 |   comments_with_text_count__q3 |   unique_commenter_count__mean |   unique_commenter_count__sd |   unique_commenter_count__median |   unique_commenter_count__q1 |   unique_commenter_count__q3 |   mean_comment_sentiment__mean |   mean_comment_sentiment__sd |   mean_comment_sentiment__median |   mean_comment_sentiment__q1 |   mean_comment_sentiment__q3 |   median_comment_sentiment__mean |   median_comment_sentiment__sd |   median_comment_sentiment__median |   median_comment_sentiment__q1 |   median_comment_sentiment__q3 |   min_comment_sentiment__mean |   min_comment_sentiment__sd |   min_comment_sentiment__median |   min_comment_sentiment__q1 |   min_comment_sentiment__q3 |   max_comment_sentiment__mean |   max_comment_sentiment__sd |   max_comment_sentiment__median |   max_comment_sentiment__q1 |   max_comment_sentiment__q3 |   std_comment_sentiment__mean |   std_comment_sentiment__sd |   std_comment_sentiment__median |   std_comment_sentiment__q1 |   std_comment_sentiment__q3 |   comment_sentiment_change_late_minus_early__mean |   comment_sentiment_change_late_minus_early__sd |   comment_sentiment_change_late_minus_early__median |   comment_sentiment_change_late_minus_early__q1 |   comment_sentiment_change_late_minus_early__q3 |   comment_sentiment_slope__mean |   comment_sentiment_slope__sd |   comment_sentiment_slope__median |   comment_sentiment_slope__q1 |   comment_sentiment_slope__q3 |   negative_comment_share__mean |   negative_comment_share__sd |   negative_comment_share__median |   negative_comment_share__q1 |   negative_comment_share__q3 |   positive_comment_share__mean |   positive_comment_share__sd |   positive_comment_share__median |   positive_comment_share__q1 |   positive_comment_share__q3 |
|:---------------|:-------------------|-----------:|----------:|----------------------:|--------------------:|------------------------:|--------------------:|--------------------:|---------------------------------:|-------------------------------:|-----------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|-------------------------------:|-----------------------------------:|-------------------------------:|-------------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|--------------------------------------------------:|------------------------------------------------:|----------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|--------------------------------:|------------------------------:|----------------------------------:|------------------------------:|------------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|
| comparison     | comparison         |       5888 |        22 |                2.7126 |              3.3145 |                       2 |                   1 |                   3 |                           2.7126 |                         3.3145 |                                  2 |                              1 |                              3 |                         1.8602 |                       1.577  |                                1 |                            1 |                            2 |                        -0.019  |                       0.217  |                                0 |                            0 |                            0 |                          -0.019  |                         0.2238 |                                  0 |                              0 |                              0 |                       -0.1416 |                      0.3037 |                               0 |                        -0.5 |                           0 |                        0.1033 |                      0.2954 |                               0 |                           0 |                        0.25 |                        0.1158 |                      0.1819 |                               0 |                           0 |                      0.2739 |                                            0.0166 |                                          0.2416 |                                                   0 |                                               0 |                                               0 |                          0.015  |                        0.1864 |                                 0 |                             0 |                             0 |                         0.1589 |                       0.2941 |                                0 |                            0 |                        0.25  |                         0.1251 |                       0.2592 |                                0 |                            0 |                       0.1067 |
| wontfix        | wontfix            |       2226 |        22 |                2.7889 |              3.8668 |                       2 |                   1 |                   4 |                           2.7889 |                         3.8668 |                                  2 |                              1 |                              4 |                         1.8823 |                       1.8371 |                                1 |                            1 |                            3 |                        -0.0209 |                       0.1693 |                                0 |                            0 |                            0 |                          -0.0214 |                         0.1743 |                                  0 |                              0 |                              0 |                       -0.1399 |                      0.2784 |                               0 |                        -0.4 |                           0 |                        0.0982 |                      0.2561 |                               0 |                           0 |                        0    |                        0.1067 |                      0.1679 |                               0 |                           0 |                      0.2524 |                                            0.0047 |                                          0.2157 |                                                   0 |                                               0 |                                               0 |                          0.0043 |                        0.1534 |                                 0 |                             0 |                             0 |                         0.1194 |                       0.2418 |                                0 |                            0 |                        0.125 |                         0.0849 |                       0.1994 |                                0 |                            0 |                       0      |

## Headline findings

- `repo_z_positive_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.2425, Hedges g=-0.2439, BH-adjusted p=1.297e-32).
- `positive_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.0403, Hedges g=-0.1648, BH-adjusted p=1.043e-12).
- `repo_z_negative_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.1614, Hedges g=-0.1618, BH-adjusted p=8.141e-12).
- `repo_z_std_comment_sentiment` was lower in WONTFIX than comparison issues (Δ=-0.1449, Hedges g=-0.1452, BH-adjusted p=1.876e-09).
- `negative_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.0395, Hedges g=-0.1406, BH-adjusted p=2.72e-09).

## Two-group tests: WONTFIX vs comparison

| feature                                          | test_family   |   wontfix_n |   comparison_n |   wontfix_mean |   comparison_mean |   mean_difference |   welch_t_stat |   p_value |   mann_whitney_u |   mann_whitney_p |   hedges_g |   cliffs_delta |   p_value_fdr_bh | reject_fdr_bh_05   |
|:-------------------------------------------------|:--------------|------------:|---------------:|---------------:|------------------:|------------------:|---------------:|----------:|-----------------:|-----------------:|-----------:|---------------:|-----------------:|:-------------------|
| mean_comment_sentiment                           | two_group     |        2226 |           5888 |        -0.0209 |           -0.019  |           -0.0019 |        -0.4198 |    0.6747 |      6.52535e+06 |           0.737  |    -0.0094 |        -0.0043 |           0.7143 | False              |
| median_comment_sentiment                         | two_group     |        2226 |           5888 |        -0.0214 |           -0.019  |           -0.0024 |        -0.5058 |    0.613  |      6.52401e+06 |           0.6806 |    -0.0113 |        -0.0045 |           0.6897 | False              |
| min_comment_sentiment                            | two_group     |        2226 |           5888 |        -0.1399 |           -0.1416 |            0.0017 |         0.2349 |    0.8143 |      6.61057e+06 |           0.4656 |     0.0056 |         0.0087 |           0.8143 | False              |
| max_comment_sentiment                            | two_group     |        2226 |           5888 |         0.0982 |            0.1033 |           -0.0051 |        -0.7638 |    0.445  |      6.48395e+06 |           0.3649 |    -0.0178 |        -0.0106 |           0.5341 | False              |
| std_comment_sentiment                            | two_group     |        2226 |           5888 |         0.1067 |            0.1158 |           -0.0091 |        -2.1252 |    0.0336 |      6.42696e+06 |           0.1075 |    -0.051  |        -0.0193 |           0.0504 | False              |
| comment_sentiment_change_late_minus_early        | two_group     |        2226 |           5888 |         0.0047 |            0.0166 |           -0.0119 |        -2.1436 |    0.0321 |      6.40615e+06 |           0.058  |    -0.0507 |        -0.0225 |           0.0504 | False              |
| comment_sentiment_slope                          | two_group     |        2226 |           5888 |         0.0043 |            0.015  |           -0.0106 |        -2.6233 |    0.0087 |      6.38315e+06 |           0.0271 |    -0.0598 |        -0.026  |           0.0175 | True               |
| negative_comment_share                           | two_group     |        2226 |           5888 |         0.1194 |            0.1589 |           -0.0395 |        -6.1664 |    0      |      6.26104e+06 |           0.0001 |    -0.1406 |        -0.0446 |           0      | True               |
| positive_comment_share                           | two_group     |        2226 |           5888 |         0.0849 |            0.1251 |           -0.0403 |        -7.4415 |    0      |      6.21921e+06 |           0      |    -0.1648 |        -0.051  |           0      | True               |
| repo_z_mean_comment_sentiment                    | two_group     |        2226 |           5888 |        -0.0351 |            0.0133 |           -0.0483 |        -2.2154 |    0.0268 |      6.14493e+06 |           0      |    -0.0483 |        -0.0623 |           0.0482 | True               |
| repo_z_median_comment_sentiment                  | two_group     |        2226 |           5888 |        -0.0331 |            0.0125 |           -0.0456 |        -2.0904 |    0.0366 |      5.99162e+06 |           0      |    -0.0456 |        -0.0857 |           0.0507 | False              |
| repo_z_min_comment_sentiment                     | two_group     |        2226 |           5888 |         0.0292 |           -0.0111 |            0.0403 |         1.7255 |    0.0845 |      6.94139e+06 |           0      |     0.0403 |         0.0592 |           0.1087 | False              |
| repo_z_max_comment_sentiment                     | two_group     |        2226 |           5888 |        -0.084  |            0.0317 |           -0.1157 |        -5.2036 |    0      |      5.91016e+06 |           0      |    -0.1158 |        -0.0981 |           0      | True               |
| repo_z_std_comment_sentiment                     | two_group     |        2226 |           5888 |        -0.1052 |            0.0398 |           -0.1449 |        -6.261  |    0      |      6.14818e+06 |           0      |    -0.1452 |        -0.0618 |           0      | True               |
| repo_z_comment_sentiment_change_late_minus_early | two_group     |        2226 |           5888 |        -0.0489 |            0.0185 |           -0.0673 |        -3.0009 |    0.0027 |      6.19894e+06 |           0.0001 |    -0.0674 |        -0.0541 |           0.0061 | True               |
| repo_z_comment_sentiment_slope                   | two_group     |        2226 |           5888 |        -0.0536 |            0.0203 |           -0.0739 |        -3.396  |    0.0007 |      6.13889e+06 |           0      |    -0.0739 |        -0.0632 |           0.0018 | True               |
| repo_z_negative_comment_share                    | two_group     |        2226 |           5888 |        -0.1171 |            0.0443 |           -0.1614 |        -7.1071 |    0      |      5.95357e+06 |           0      |    -0.1618 |        -0.0915 |           0      | True               |
| repo_z_positive_comment_share                    | two_group     |        2226 |           5888 |        -0.176  |            0.0665 |           -0.2425 |       -12.1993 |    0      |      5.74002e+06 |           0      |    -0.2439 |        -0.1241 |           0      | True               |

## Multi-group omnibus tests

_No rows available._

## Multi-group pairwise tests

_No rows available._

## Proportion / prevalence tests

| indicator                     | test_family   | test       |   statistic |   p_value |   wontfix_rate |   comparison_rate |   odds_ratio |   p_value_fdr_bh | reject_fdr_bh_05   |
|:------------------------------|:--------------|:-----------|------------:|----------:|---------------:|------------------:|-------------:|-----------------:|:-------------------|
| has_strongly_negative_comment | proportion    | chi_square |      8.1719 |    0.0043 |         0.2637 |            0.2962 |       0.851  |           0.0085 | True               |
| has_strongly_positive_comment | proportion    | chi_square |     13.7736 |    0.0002 |         0.2161 |            0.2561 |       0.8006 |           0.0008 | True               |
| late_more_negative_than_early | proportion    | chi_square |      0.5492 |    0.4586 |         0.1505 |            0.1437 |       1.0558 |           0.4586 | False              |
| high_sentiment_volatility     | proportion    | chi_square |      1.2435 |    0.2648 |         0.2412 |            0.2536 |       0.9359 |           0.3531 | False              |

## Early-vs-late within-group tests

| comparison_group   |   n_pairs |   early_mean |   late_mean |   late_minus_early_mean |   paired_t_p |   paired_t_stat |   wilcoxon_p |   wilcoxon_stat |
|:-------------------|----------:|-------------:|------------:|------------------------:|-------------:|----------------:|-------------:|----------------:|
| comparison         |      5888 |      -0.0273 |     -0.0107 |                  0.0166 |       0      |          5.2684 |        0     |          763782 |
| wontfix            |      2226 |      -0.024  |     -0.0193 |                  0.0047 |       0.3051 |          1.0258 |        0.506 |          114708 |

## Adjusted OLS models

_No rows available._

## Adjusted logistic models

_No rows available._

## Figures generated

- `01_issue_counts_by_group.png`
- `02_mean_comment_sentiment_distribution.png`
- `02b_mean_comment_sentiment_signed_distribution.png`
- `03_sentiment_volatility_distribution.png`
- `04_min_comment_sentiment_distribution.png`
- `04b_min_comment_sentiment_signed_distribution.png`
- `04c_max_comment_sentiment_distribution.png`
- `04d_max_comment_sentiment_signed_distribution.png`
- `04e_comment_sentiment_range_distribution.png`
- `04f_comment_sentiment_range_nonzero_distribution.png`
- `05_early_vs_late_sentiment.png`
- `06_mean_sentiment_vs_comment_count.png`
- `07_volatility_vs_unique_commenters.png`
- `08_repo_forest_effects_panel.png`
- `09_feature_correlation_heatmap.png`
- `10_comment_trajectory.png`


## Interpretation guardrails

- These analyses use sentiment features derived from issue and comment text. They are useful for comparative
  discussion-tone analysis, but they are not the same thing as intent, civility, or maintainer motivation.
- Repository baselines differ, so raw and within-repository-standardized analyses should be interpreted together.
- Comment-level records are not treated as independent observations in the main inferential tests.
- If certain optional upstream files were unavailable, subgroup or issue-type enrichment may be partial.
- Statistical significance should be read alongside effect sizes and confidence intervals.
