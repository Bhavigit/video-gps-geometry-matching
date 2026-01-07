import pandas as pd
import numpy as np

# CONFIG
INPUT_CSV = "data/table_video_gps_data.csv"
OUTPUT_CSV = "output/video_gps_curvature_1.csv"

EARTH_RADIUS = 6371000.0
SMOOTH_WINDOW = 5


def curvature_from_gps(lat, lon):
    """
    Civil-engineering curvature using 3 GPS points.
    Returns signed curvature (1/m).
    """
    lat = np.radians(lat)
    lon = np.radians(lon)

    lat0 = lat[1]

    x = EARTH_RADIUS * np.cos(lat0) * (lon - lon[1])
    y = EARTH_RADIUS * (lat - lat[1])

    x1, x2, x3 = x
    y1, y2, y3 = y

    a = np.hypot(x2 - x1, y2 - y1)
    b = np.hypot(x3 - x2, y3 - y2)
    c = np.hypot(x3 - x1, y3 - y1)

    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)

    if area_sq <= 0:
        return 0.0

    area = np.sqrt(area_sq)
    R = (a * b * c) / (4 * area)

    sign = np.sign((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    return sign / R


def main():
    print("Loading GPS data...")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()


    df = df.sort_values(
        ["networkID", "lane", "FrameNumber"]
    ).reset_index(drop=True)

    # Smooth GPS to reduce noise
    df["latitude"] = (
        df.groupby(["networkID", "lane"])["latitude"]
        .rolling(SMOOTH_WINDOW, center=True)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["longitude"] = (
        df.groupby(["networkID", "lane"])["longitude"]
        .rolling(SMOOTH_WINDOW, center=True)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["gps_curvature"] = np.nan

    print("Computing GPS curvature (lane-wise)...")
    for (net, lane), g in df.groupby(["networkID", "lane"]):
        idxs = g.index.tolist()
        for j in range(1, len(idxs) - 1):
            i = idxs[j]
            lat = df.loc[i - 1:i + 1, "latitude"].values
            lon = df.loc[i - 1:i + 1, "longitude"].values
            if not np.any(np.isnan(lat)) and not np.any(np.isnan(lon)):
                df.loc[i, "gps_curvature"] = curvature_from_gps(lat, lon)


    df.to_csv(OUTPUT_CSV, index=False)
    print("Curvature computation completed.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
