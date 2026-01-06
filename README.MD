# GPS to Geometry Matching

This repository contains a script to match video frame GPS data
with road geometry data.

## Method
- GPS latitude/longitude from video frames is used to compute
  cumulative distance (loc) along the road.
- Geometry segments are matched using:
  LocFrom ≤ loc ≤ LocTo
- Gradient, crossfall, and curvature are attached per frame.

## Notes
- Curvature is taken directly from the geometry table
  (not computed from GPS).
- Raw and aggregated outputs are included.
