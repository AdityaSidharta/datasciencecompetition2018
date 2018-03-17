import os
import pandas as pd
import numpy as np
from tqdm import tqdm
'''determine probability distribution of the gauge / weather station'''

FIELD_CORR_TABLE = os.path.join(os.getcwd(), "corr.csv")
NUM_STATIONS = 50
FIELD_CORR_STATIONS = ["corr_station_" + str(i) for i in range(NUM_STATIONS)]


def compute_correlation(train_df, gauge_df):
    corr_df = {}
    training_weeks = np.unique(train_df.week)
    contraband_weeks = {18, 19, 20, 21, 22, 23, 29, 31, 35, 38, 42, 45, 46, 49, 50, 51}
    train_df = train_df[train_df.week.apply(lambda x: x not in contraband_weeks)]
    selected_weeks = [x for x in training_weeks if x not in contraband_weeks]
    gauge_df = gauge_df[gauge_df.apply(lambda x: x.Week in selected_weeks, axis=1)].sort_values("Week")
    for latlon, df in tqdm(train_df.groupby(["lat", "long"])):
        lat = latlon[0]
        lon = latlon[1]
        df_mean = np.matmul(df.values[:, 3:-2], np.arange(34)) / 2016.
        for i in range(NUM_STATIONS):
            if i not in corr_df:
                corr_df[i] = {}
            corr_df[i][(lat, lon)] = np.corrcoef(df_mean, gauge_df[str(i)].values)[0, 1]
    return corr_df


def add_lat_long(df_corr):
    """ add latlong back """
    if 'lat' not in df_corr.columns:
        df_corr['lat'] = (df_corr['latlong'] // 1000.).astype(int)
        df_corr['long'] = (df_corr['latlong'] % 1000.).astype(int)
    return df_corr


def filter_negative_correlation(df_corr):
    for i in range(NUM_STATIONS):
        df_corr[FIELD_CORR_STATIONS[i]] = df_corr[FIELD_CORR_STATIONS[i]].apply(lambda x: x if x > 0 else 0)
    return df_corr


def normalise_correlation(df_corr):
    '''df_corr must be rid of negative entries'''
    col_sum = np.sum(df_corr[FIELD_CORR_STATIONS].values, axis=0)
    df_corr[FIELD_CORR_STATIONS] = df_corr[FIELD_CORR_STATIONS].values / col_sum
    return df_corr


def exponential_correlation(df_corr):
    return np.exp(df_corr)


def add_coords(df_corr, latlong_map):
    x, y = latlong_map.set_index(["map_lat", "map_long"]).loc[list(map(lambda x, y: (x, y), df_corr["lat"],
                                                                       df_corr["long"]))].values.T
    df_corr["x"] = x
    df_corr["y"] = y
    return df_corr


if __name__ == '__main__':
    # df_corr = pd.read_csv(FIELD_CORR_TABLE)
    # linear_normalised_df = normalise_correlation(filter_negative_correlation(add_lat_long(df_corr)))
    train_df = pd.read_csv("data_singapore_only.csv").sort_values("week")
    gauge_df = pd.read_csv("datasets/gauge.csv")
    corr = compute_correlation(train_df, gauge_df)
    station_dict = {i: "corr_station_" + str(i) for i in range(NUM_STATIONS)}
    station_dict.update({"level_0": "lat", "level_1": "long"})
    LATLONG_MAP_PATH = os.path.join(os.getcwd(), "processed_data/map_latlong.csv")
    latlong_map = pd.read_csv(LATLONG_MAP_PATH)
    add_coords(pd.DataFrame(corr).reset_index().rename(columns=station_dict), latlong_map)\
        .to_csv("toto_corr_3.csv", index=False)
