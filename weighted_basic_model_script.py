import pandas as pd
from tqdm import tqdm
from WeightedBasicModel import WeightedBasicModel


def match_week(train_df, gauge_df):
    weeks = train_df.week
    grouped_gauge_df = gauge_df.groupby("Week")
    reformatted_y = []
    for week in tqdm(weeks):
        reformatted_y.append(grouped_gauge_df.get_group(week))
    return pd.concat(reformatted_y)[[str(i) for i in range(50)]]


train_path = "data_singapore_only.csv"
gauge_path = "to_toto_3.csv"
prob_table_path = "linear_correlation_prob_table.csv"

train_df = pd.read_csv(train_path).head(100)[["lat", "long", "week"] + [str(i) for i in range(34)]]
gauge_df = pd.read_csv(gauge_path)
prob_table = pd.read_csv(prob_table_path)
reformatted_y = match_week(train_df, gauge_df)
print reformatted_y.head()
print train_df.head()
clf = WeightedBasicModel().fit(train_df, reformatted_y, prob_table=prob_table)
clf.plot_lost()
