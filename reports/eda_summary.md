# FlowCast - EDA summary (auto-generated)

* Records: 181,200 over 25 segments, 01 Jan 2025 - 31 May 2025.
* Volume peaks twice a day (AM ~08:00-09:00, PM ~17:00-18:00); weekday vs weekend profiles differ -> weekend/peak flags and hour encodings are justified.
* Volume skewness = 1.00: right-skewed, so RMSE (not accuracy) is used and tree models are expected to beat linear ones.
* Correlation volume vs occupancy = 0.94; speed vs occupancy = -0.93 (strong, redundancy noted).
* Congestion classes are imbalanced: Free-flow 61.4%, Moderate 23.8%, Heavy 9.3%, Severe 5.5% -> macro-F1 is the headline metric.
* Incident rate: 0.91% of windows. Rain/Fog raise it to 3.4%/3.4% vs 0.8% in clear weather.
* Heavy congestion ratio (V/C >= 0.8) share: 14.7%.
