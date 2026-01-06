
import pandas as pd
import numpy as np
import os


VIDEO_GPS_CSV = "data/table_video_gps_data.csv"
GEOMETRY_CSV = "data/table_geometry.csv"
OUTPUT_CSV = "data/video_frames_with_geometry_1.csv"

EARTH_RADIUS_M = 6371000.0
MAX_GPS_JUMP_METERS = 300


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_cumulative_distance(df):
    """
    Compute loc with reset on GPS jumps.
    """
    df = df.sort_values(["networkID", "lane", "FrameNumber"]).reset_index(drop=True)
    df["loc"] = 0.0

    for (net, lane), idxs in df.groupby(["networkID", "lane"]).groups.items():
        idxs = list(idxs)

        for i in range(1, len(idxs)):
            prev_i = idxs[i - 1]
            curr_i = idxs[i]

            lat1, lon1 = df.loc[prev_i, ["latitude", "longitude"]]
            lat2, lon2 = df.loc[curr_i, ["latitude", "longitude"]]

            dist = haversine(lat1, lon1, lat2, lon2)

            if dist > MAX_GPS_JUMP_METERS:

                df.loc[curr_i, "loc"] = 0.0
            else:
                df.loc[curr_i, "loc"] = df.loc[prev_i, "loc"] + dist

    return df


def match_geometry(video_df, geom_df):
    rows = []

    for _, r in video_df.iterrows():
        g = geom_df[
            (geom_df["networkID"] == r["networkID"]) &
            (geom_df["lane"] == r["lane"]) &
            (geom_df["locFrom"] <= r["loc"]) &
            (r["loc"] <= geom_df["locTo"])
        ]

        if not g.empty:
            g = g.iloc[0]
            rows.append({
                "FrameNumber": r["FrameNumber"],
                "networkID": r["networkID"],
                "lane": r["lane"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "loc": r["loc"],
                "gradient": g["gradient"],
                "crossfall": g["crossfall"],
                "curvature": g["curvature"]
            })
        else:
            rows.append({
                "FrameNumber": r["FrameNumber"],
                "networkID": r["networkID"],
                "lane": r["lane"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "loc": r["loc"],
                "gradient": np.nan,
                "crossfall": np.nan,
                "curvature": np.nan
            })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    video_df = pd.read_csv(VIDEO_GPS_CSV)
    geom_df = pd.read_csv(GEOMETRY_CSV)


    video_df.columns = video_df.columns.str.strip()
    geom_df.columns = geom_df.columns.str.strip()

    # --- Compute cumulative distance---
    print("Computing cumulative distance...")
    video_df = compute_cumulative_distance(video_df)

    # --- Match geometry ---
    print("Matching geometry...")
    matched_df = match_geometry(video_df, geom_df)

    # --- Save RAW matched output ---
    raw_output = "data/video_frames_with_geometry.csv"
    matched_df.to_csv(raw_output, index=False)
    print(f"Raw matched file saved: {raw_output}")


    # --- Aggregate to one row per frame ---
    final_df = (
        matched_df
        .groupby(
            ["FrameNumber", "networkID", "lane"],
            as_index=False
        )
        .agg({
            "latitude": "mean",
            "longitude": "mean",
            "loc": "mean",
            "gradient": "mean",
            "crossfall": "mean",
            "curvature": "mean"
        })
    )

    final_output = "data/video_frames_with_geometry_final.csv"
    final_df.to_csv(final_output, index=False)
    print(f"Final aggregated file saved: {final_output}")

if __name__ == "__main__":
    main()