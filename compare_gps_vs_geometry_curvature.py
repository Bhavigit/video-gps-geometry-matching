import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# CONFIG
GPS_CURV_CSV = "output/video_gps_curvature_with_loc.csv"
GEOM_CSV = "data/table_geometry.csv"
OUT_CSV = "output/curvature_comparison.csv"


def main():
    print("Loading data...")

    gps_df = pd.read_csv(GPS_CURV_CSV)
    geom_df = pd.read_csv(GEOM_CSV)

    gps_df.columns = gps_df.columns.str.strip()
    geom_df.columns = geom_df.columns.str.strip()

    matched_rows = []

    print("Matching GPS curvature with geometry curvature...")

    for _, r in gps_df.iterrows():

        # Skip invalid curvature
        if pd.isna(r["gps_curvature"]):
            continue

        g = geom_df[
            (geom_df["networkID"] == r["networkID_y"]) &
            (geom_df["lane"] == r["lane_y"]) &
            (geom_df["locFrom"] <= r["loc"]) &
            (r["loc"] <= geom_df["locTo"])
        ]

        if not g.empty:
            g = g.iloc[0]

            matched_rows.append({
                "gps_curvature": r["gps_curvature"],
                "geometry_curvature": g["curvature"],
                "loc": r["loc"],
                "networkID": r["networkID_y"],
                "lane": r["lane_y"]
            })

    comp_df = pd.DataFrame(matched_rows)
    print(f"Matched points: {len(comp_df)}")

    if len(comp_df) == 0:
        print(" No matched data — check loc ranges")
        return

    # ---- Error metrics ----
    mae = mean_absolute_error(
       (comp_df["geometry_curvature"]),
       (comp_df["gps_curvature"])
    )

    rmse = np.sqrt(
        mean_squared_error(
            (comp_df["geometry_curvature"]),
            (comp_df["gps_curvature"])
        )
    )

    corr = np.corrcoef(
        (comp_df["geometry_curvature"]),
        (comp_df["gps_curvature"])
    )[0, 1]

    comp_df.to_csv(OUT_CSV, index=False)

    print("\n==== Curvature Comparison Results ====")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"Corr : {corr:.3f}")
    print(f"\nSaved comparison to: {OUT_CSV}")


if __name__ == "__main__":
    main()
