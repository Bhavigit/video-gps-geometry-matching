# GPS Curvature Calculation

This repository implements a civil-engineering based curvature
calculation from GPS points using a 3-point circumcircle method.

## Scripts
- compute_gps_curvature.py
  Computes signed curvature (1/m) from GPS latitude and longitude,
  grouped by networkID and lane.

- compare_gps_vs_geometry_curvature.py
  Compares GPS-derived curvature against geometry curvature using
  MAE, RMSE, and correlation.

## Notes
Results show low correlation with geometry curvature, confirming
that geometry curvature should remain the ground-truth source.
