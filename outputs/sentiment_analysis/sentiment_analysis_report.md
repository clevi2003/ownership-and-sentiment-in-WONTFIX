# Sentiment Analysis Report

## Overview

This report summarizes an issue-level sentiment analysis of WONTFIX discussions versus comparison issues.
The primary inferential unit is the **issue**, while comment-level records are used mainly for temporal
trajectory analysis and descriptive summaries.

- Issues analyzed: **6319**
- Repositories represented: **8**
- Groups present: **comparison, wontfix**

## QA / coverage summary

| metric                                                   | value                     |
|:---------------------------------------------------------|:--------------------------|
| issue_feature_rows_raw                                   | 6319                      |
| comment_feature_rows_raw                                 | 14793                     |
| issue_feature_rows_normalized                            | 6319                      |
| comment_feature_rows_normalized                          | 14793                     |
| issue_duplicates                                         | 0                         |
| comment_duplicates                                       | 0                         |
| repos_represented                                        | 8                         |
| analysis_groups                                          | ['comparison', 'wontfix'] |
| issues_final                                             | 6319                      |
| repos_final                                              | 8                         |
| issues_analysis_set__comparison                          | 4495                      |
| issues_analysis_set__wontfix                             | 1824                      |
| issues_comparison_group__comparison                      | 4495                      |
| issues_comparison_group__wontfix                         | 1824                      |
| comment_count_median                                     | 1.0                       |
| comment_count_mean                                       | 2.3410349738882736        |
| comments_with_text_count_median                          | 1.0                       |
| comments_with_text_count_mean                            | 2.3410349738882736        |
| unique_commenter_count_median                            | 1.0                       |
| unique_commenter_count_mean                              | 1.7385662288336763        |
| zero_comment_issue_share                                 | 0.14131982908688084       |
| zero_text_comment_issue_share                            | 0.14131982908688084       |
| one_commenter_issue_share                                | 0.5730337078651685        |
| missing_share__mean_comment_sentiment                    | 0.0                       |
| missing_share__median_comment_sentiment                  | 0.0                       |
| missing_share__min_comment_sentiment                     | 0.0                       |
| missing_share__max_comment_sentiment                     | 0.0                       |
| missing_share__std_comment_sentiment                     | 0.0                       |
| missing_share__comment_sentiment_change_late_minus_early | 0.0                       |
| missing_share__comment_sentiment_slope                   | 0.0                       |
| missing_share__negative_comment_share                    | 0.0                       |
| missing_share__positive_comment_share                    | 0.0                       |
| comments_final                                           | 14793                     |
| comment_missing_text_share                               | 0.0                       |

## Group descriptives

| analysis_set   | comparison_group   |   n_issues |   n_repos |   comment_count__mean |   comment_count__sd |   comment_count__median |   comment_count__q1 |   comment_count__q3 |   comments_with_text_count__mean |   comments_with_text_count__sd |   comments_with_text_count__median |   comments_with_text_count__q1 |   comments_with_text_count__q3 |   unique_commenter_count__mean |   unique_commenter_count__sd |   unique_commenter_count__median |   unique_commenter_count__q1 |   unique_commenter_count__q3 |   mean_comment_sentiment__mean |   mean_comment_sentiment__sd |   mean_comment_sentiment__median |   mean_comment_sentiment__q1 |   mean_comment_sentiment__q3 |   median_comment_sentiment__mean |   median_comment_sentiment__sd |   median_comment_sentiment__median |   median_comment_sentiment__q1 |   median_comment_sentiment__q3 |   min_comment_sentiment__mean |   min_comment_sentiment__sd |   min_comment_sentiment__median |   min_comment_sentiment__q1 |   min_comment_sentiment__q3 |   max_comment_sentiment__mean |   max_comment_sentiment__sd |   max_comment_sentiment__median |   max_comment_sentiment__q1 |   max_comment_sentiment__q3 |   std_comment_sentiment__mean |   std_comment_sentiment__sd |   std_comment_sentiment__median |   std_comment_sentiment__q1 |   std_comment_sentiment__q3 |   comment_sentiment_change_late_minus_early__mean |   comment_sentiment_change_late_minus_early__sd |   comment_sentiment_change_late_minus_early__median |   comment_sentiment_change_late_minus_early__q1 |   comment_sentiment_change_late_minus_early__q3 |   comment_sentiment_slope__mean |   comment_sentiment_slope__sd |   comment_sentiment_slope__median |   comment_sentiment_slope__q1 |   comment_sentiment_slope__q3 |   negative_comment_share__mean |   negative_comment_share__sd |   negative_comment_share__median |   negative_comment_share__q1 |   negative_comment_share__q3 |   positive_comment_share__mean |   positive_comment_share__sd |   positive_comment_share__median |   positive_comment_share__q1 |   positive_comment_share__q3 |
|:---------------|:-------------------|-----------:|----------:|----------------------:|--------------------:|------------------------:|--------------------:|--------------------:|---------------------------------:|-------------------------------:|-----------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|-------------------------------:|-----------------------------------:|-------------------------------:|-------------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|------------------------------:|----------------------------:|--------------------------------:|----------------------------:|----------------------------:|--------------------------------------------------:|------------------------------------------------:|----------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|--------------------------------:|------------------------------:|----------------------------------:|------------------------------:|------------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|
| comparison     | comparison         |       4495 |         8 |                2.3241 |              3.1251 |                       1 |                   1 |                   3 |                           2.3241 |                         3.1251 |                                  1 |                              1 |                              3 |                         1.731  |                       1.6225 |                                1 |                            1 |                            2 |                        -0.0144 |                       0.1979 |                                0 |                            0 |                            0 |                          -0.0146 |                         0.2028 |                                  0 |                              0 |                              0 |                       -0.1013 |                      0.2738 |                               0 |                           0 |                           0 |                        0.0722 |                      0.261  |                               0 |                           0 |                           0 |                        0.0847 |                      0.1634 |                               0 |                           0 |                      0      |                                            0.019  |                                          0.212  |                                                   0 |                                               0 |                                               0 |                          0.0172 |                        0.1687 |                                 0 |                             0 |                             0 |                         0.123  |                       0.2705 |                                0 |                            0 |                            0 |                         0.0992 |                       0.2431 |                                0 |                            0 |                            0 |
| wontfix        | wontfix            |       1824 |         8 |                2.3827 |              3.4658 |                       1 |                   0 |                   3 |                           2.3827 |                         3.4658 |                                  1 |                              0 |                              3 |                         1.7571 |                       1.8844 |                                1 |                            0 |                            3 |                        -0.0167 |                       0.1462 |                                0 |                            0 |                            0 |                          -0.016  |                         0.1487 |                                  0 |                              0 |                              0 |                       -0.108  |                      0.2501 |                               0 |                           0 |                           0 |                        0.0729 |                      0.2221 |                               0 |                           0 |                           0 |                        0.0832 |                      0.1543 |                               0 |                           0 |                      0.0851 |                                            0.0021 |                                          0.1967 |                                                   0 |                                               0 |                                               0 |                          0.0017 |                        0.1432 |                                 0 |                             0 |                             0 |                         0.0878 |                       0.2094 |                                0 |                            0 |                            0 |                         0.0634 |                       0.173  |                                0 |                            0 |                            0 |

## Headline findings

- `repo_z_positive_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.2586, Hedges g=-0.2603, BH-adjusted p=2.353e-31).
- `repo_z_negative_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.1913, Hedges g=-0.1920, BH-adjusted p=1.987e-13).
- `positive_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.0358, Hedges g=-0.1588, BH-adjusted p=3.161e-10).
- `repo_z_std_comment_sentiment` was lower in WONTFIX than comparison issues (Δ=-0.1551, Hedges g=-0.1555, BH-adjusted p=8.056e-09).
- `negative_comment_share` was lower in WONTFIX than comparison issues (Δ=-0.0352, Hedges g=-0.1383, BH-adjusted p=1.156e-07).

## Two-group tests: WONTFIX vs comparison

| feature                                          | test_family   |   wontfix_n |   comparison_n |   wontfix_mean |   comparison_mean |   mean_difference |   welch_t_stat |   p_value |   mann_whitney_u |   mann_whitney_p |   hedges_g |   cliffs_delta |   p_value_fdr_bh | reject_fdr_bh_05   |
|:-------------------------------------------------|:--------------|------------:|---------------:|---------------:|------------------:|------------------:|---------------:|----------:|-----------------:|-----------------:|-----------:|---------------:|-----------------:|:-------------------|
| mean_comment_sentiment                           | two_group     |        1824 |           4495 |        -0.0167 |           -0.0144 |           -0.0023 |        -0.5123 |    0.6085 |      4.08933e+06 |           0.851  |    -0.0126 |        -0.0025 |           0.7302 | False              |
| median_comment_sentiment                         | two_group     |        1824 |           4495 |        -0.016  |           -0.0146 |           -0.0014 |        -0.3068 |    0.759  |      4.09142e+06 |           0.859  |    -0.0075 |        -0.002  |           0.8036 | False              |
| min_comment_sentiment                            | two_group     |        1824 |           4495 |        -0.108  |           -0.1013 |           -0.0067 |        -0.9361 |    0.3493 |      4.07619e+06 |           0.6423 |    -0.025  |        -0.0057 |           0.4491 | False              |
| max_comment_sentiment                            | two_group     |        1824 |           4495 |         0.0729 |            0.0722 |            0.0007 |         0.1085 |    0.9136 |      4.10425e+06 |           0.921  |     0.0028 |         0.0012 |           0.9136 | False              |
| std_comment_sentiment                            | two_group     |        1824 |           4495 |         0.0832 |            0.0847 |           -0.0015 |        -0.3516 |    0.7251 |      4.11132e+06 |           0.8109 |    -0.0095 |         0.0029 |           0.8036 | False              |
| comment_sentiment_change_late_minus_early        | two_group     |        1824 |           4495 |         0.0021 |            0.019  |           -0.0169 |        -3.0169 |    0.0026 |      3.97257e+06 |           0.0097 |    -0.0811 |        -0.0309 |           0.0046 | True               |
| comment_sentiment_slope                          | two_group     |        1824 |           4495 |         0.0017 |            0.0172 |           -0.0155 |        -3.694  |    0.0002 |      3.96388e+06 |           0.0052 |    -0.0957 |        -0.0331 |           0.0004 | True               |
| negative_comment_share                           | two_group     |        1824 |           4495 |         0.0878 |            0.123  |           -0.0352 |        -5.5397 |    0      |      3.9684e+06  |           0.0056 |    -0.1383 |        -0.032  |           0      | True               |
| positive_comment_share                           | two_group     |        1824 |           4495 |         0.0634 |            0.0992 |           -0.0358 |        -6.5786 |    0      |      3.9541e+06  |           0.0011 |    -0.1588 |        -0.0355 |           0      | True               |
| repo_z_mean_comment_sentiment                    | two_group     |        1824 |           4495 |        -0.0289 |            0.0117 |           -0.0406 |        -1.6958 |    0.09   |      3.80706e+06 |           0      |    -0.0406 |        -0.0713 |           0.135  | False              |
| repo_z_median_comment_sentiment                  | two_group     |        1824 |           4495 |        -0.0243 |            0.0099 |           -0.0342 |        -1.4292 |    0.153  |      3.66065e+06 |           0      |    -0.0342 |        -0.107  |           0.2119 | False              |
| repo_z_min_comment_sentiment                     | two_group     |        1824 |           4495 |         0.0355 |           -0.0144 |            0.0499 |         1.9265 |    0.0541 |      4.48097e+06 |           0      |     0.0499 |         0.0931 |           0.0885 | False              |
| repo_z_max_comment_sentiment                     | two_group     |        1824 |           4495 |        -0.0894 |            0.0363 |           -0.1257 |        -5.1492 |    0      |      3.68012e+06 |           0      |    -0.1258 |        -0.1023 |           0      | True               |
| repo_z_std_comment_sentiment                     | two_group     |        1824 |           4495 |        -0.1104 |            0.0448 |           -0.1551 |        -6.0298 |    0      |      3.79008e+06 |           0      |    -0.1555 |        -0.0755 |           0      | True               |
| repo_z_comment_sentiment_change_late_minus_early | two_group     |        1824 |           4495 |        -0.0699 |            0.0284 |           -0.0983 |        -3.945  |    0.0001 |      3.76043e+06 |           0      |    -0.0984 |        -0.0827 |           0.0002 | True               |
| repo_z_comment_sentiment_slope                   | two_group     |        1824 |           4495 |        -0.0798 |            0.0324 |           -0.1121 |        -4.6157 |    0      |      3.70996e+06 |           0      |    -0.1122 |        -0.095  |           0      | True               |
| repo_z_negative_comment_share                    | two_group     |        1824 |           4495 |        -0.1361 |            0.0552 |           -0.1913 |        -7.6644 |    0      |      3.59341e+06 |           0      |    -0.192  |        -0.1234 |           0      | True               |
| repo_z_positive_comment_share                    | two_group     |        1824 |           4495 |        -0.184  |            0.0747 |           -0.2586 |       -11.9636 |    0      |      3.55567e+06 |           0      |    -0.2603 |        -0.1326 |           0      | True               |

## Multi-group omnibus tests

_No rows available._

## Multi-group pairwise tests

_No rows available._

## Proportion / prevalence tests

| indicator                     | test_family   | test          |   statistic |   p_value |   wontfix_rate |   comparison_rate |   odds_ratio |   p_value_fdr_bh | reject_fdr_bh_05   |
|:------------------------------|:--------------|:--------------|------------:|----------:|---------------:|------------------:|-------------:|-----------------:|:-------------------|
| has_strongly_negative_comment | proportion    | chi_square    |      3.8141 |    0.0508 |         0.1996 |            0.2222 |       0.8725 |           0.0678 | False              |
| has_strongly_positive_comment | proportion    | chi_square    |      6.4216 |    0.0113 |         0.1639 |            0.1915 |       0.8275 |           0.023  | True               |
| late_more_negative_than_early | proportion    | chi_square    |      6.3855 |    0.0115 |         0.1212 |            0.0992 |       1.2516 |           0.023  | True               |
| high_sentiment_volatility     | proportion    | fishers_exact |    nan      |    1      |         1      |            1      |     nan      |           1      | False              |

## Early-vs-late within-group tests

| comparison_group   |   n_pairs |   early_mean |   late_mean |   late_minus_early_mean |   paired_t_p |   paired_t_stat |   wilcoxon_p |   wilcoxon_stat |
|:-------------------|----------:|-------------:|------------:|------------------------:|-------------:|----------------:|-------------:|----------------:|
| comparison         |      4495 |      -0.024  |     -0.0051 |                  0.019  |       0      |          5.9962 |        0     |          220263 |
| wontfix            |      1824 |      -0.0184 |     -0.0163 |                  0.0021 |       0.6482 |          0.4563 |        0.881 |           49878 |

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
- `06b_mean_sentiment_vs_comment_count_binned_trend.png`
- `07_volatility_vs_unique_commenters.png`
- `07b_volatility_vs_unique_commenters_binned_trend.png`
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
