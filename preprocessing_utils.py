import numpy as np
import pandas as pd
import os
from math import radians


R_EARTH = 6373.0 * 1000
MAX_DISTANCE = 60 * 1000
SINGAPORE_LAT_LON = (1.3521, 103.8198)
LATLONG_MAP_PATH = os.path.join(os.getcwd(), "processed_data/map_latlong.csv")


def get_distance_to_list_of_pts(pt, list_of_pts):

    lats = np.array(map(lambda x: x[0], list_of_pts))
    lons = np.array(map(lambda x: x[1], list_of_pts))
    lat1 = radians(pt[0])
    lon1 = radians(pt[1])
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R_EARTH * c  # distance in m
    return distance


def filter_distance(df):
    ''' filter from dataframe of radar's reading with lat long that has been estimated '''
    return df[get_distance_to_list_of_pts(SINGAPORE_LAT_LON, df[["map_lat", "map_long"]].values) <= MAX_DISTANCE]


def valid_lat_long():
    latlong_map = filter_distance(pd.read_csv(LATLONG_MAP_PATH))
    return set(map(lambda x, y: (x, y), latlong_map.lat, latlong_map.long))


if __name__ == '__main__':
    valid_lat_long = valid_lat_long()
    import os
    from tqdm import tqdm
    dir_path = os.path.join(os.getcwd(), "datasets/")
    radar = os.path.join(dir_path, "radar")
    all_df = []
    for date_csv in tqdm(os.listdir(radar)):
        fpath = os.path.join(radar, date_csv)
        df = pd.read_csv(fpath)
        df = df[df.apply(lambda x: (x["lat"], x["long"]) in valid_lat_long, axis=1)]
        week = date_csv[-6:-4]
        df["week"] = np.repeat(week, len(df))
        all_df.append(df)
    df = pd.concat(all_df)
    df.to_csv("full_training_data.csv", index=False)
