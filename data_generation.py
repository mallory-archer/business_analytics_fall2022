import os
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pd.options.display.max_columns = 25


def make_row(input_row_dict, feature_cols, buy_not_buy):
    def draw_values(draw_dict):
        if draw_dict['dist'] == 'uniform':
            return int(np.random.uniform(low=draw_dict['min'], high=draw_dict['max'], size=1))
        elif draw_dict['dist'] == 'normal':
            return int(np.random.normal(loc=draw_dict['mean'], scale=draw_dict['std'], size=1))
        elif draw_dict['dist'] == 'binary':
            return int(np.random.uniform(low=0, high=1, size=1) < draw_dict['prob'])
        else:
            print('No method for drawing attribute from %s distribution. Check make_row function.' % draw_dict['dist'])
            return None
    return np.array([draw_values(v) for k, v in input_row_dict.items() if k in feature_cols] + [buy_not_buy])


def make_type_df(input_row_dict, feature_cols, buy_TF, scale_factor=1):
    # if buy_TF:
    #     # return pd.DataFrame(np.tile(make_row(input_row_dict, feature_cols, buy_not_buy='1'), (r[select_class_col_name][buy_TF], 1)), columns=feature_cols + [select_class_col_name])
    #     return pd.DataFrame([make_row(input_row_dict, feature_cols, buy_not_buy='1') for n in range(r[select_class_col_name][buy_TF])], columns=feature_cols + [select_class_col_name])
    # else:
    #     return pd.DataFrame([make_row(input_row_dict, feature_cols, buy_not_buy='0') for n in range(r[select_class_col_name][buy_TF])], columns=feature_cols + [select_class_col_name])
    return pd.DataFrame([make_row(input_row_dict, feature_cols, buy_not_buy=buy_TF) for n in range(r[select_class_col_name][buy_TF] * scale_factor)], columns=feature_cols + [select_class_col_name])


# --- general params, see specific sections for additional params ----
save_TF = True
fp = 'orig_data'
fp_output = 'student_data'
fn_nlp_csv = 'DisneylandReviews.csv'
fn_nlp_txt = 'disney_tweets.txt'

fn_future_data = 'projected_customer_base.csv'
fn_historical_data = 'historical_customer_data.csv'

select_feature_cols = ['Number of family members', 'Income', 'Has child under 6 years old']
select_class_col_name = 'Buy fast pass'

# ------------ Decision tree ------------
# decision_tree_customer_types = {'1': {'Number of family members': 2, 'Income': 100000, 'Has child under 6 years old': 0, 'Buy fast pass': {False: 58, True: 146}},
#                                 '2': {'Number of family members': 2, 'Income': 50000, 'Has child under 6 years old': 0, 'Buy fast pass': {False: 53, True: 3}},
#                                 '3': {'Number of family members': 6, 'Income': 67000, 'Has child under 6 years old': 0.5, 'Buy fast pass': {False: 96, True: 4}},
#                                 '4': {'Number of family members': 4, 'Income': 67000, 'Has child under 6 years old': 0, 'Buy fast pass': {False: 1, True: 12}},
#                                 '5': {'Number of family members': 4, 'Income': 67000, 'Has child under 6 years old': 1, 'Buy fast pass': {False: 54, True: 13}}}

decision_tree_customer_types_draw_info = {'1': {'Number of family members': {'dist': 'uniform', 'min': 1, 'max': 3}, 'Income': {'dist': 'normal', 'mean': 100000, 'std': 7000}, 'Has child under 6 years old': {'dist': 'binary', 'prob': 1-0.2}, 'Buy fast pass': {False: 58, True: 146}},
                                          '2': {'Number of family members': {'dist': 'uniform', 'min': 1, 'max': 3}, 'Income': {'dist': 'normal', 'mean': 50000, 'std': 4000}, 'Has child under 6 years old': {'dist': 'binary', 'prob': 1-0.2}, 'Buy fast pass': {False: 53, True: 3}},
                                          '3': {'Number of family members': {'dist': 'normal', 'mean': 5.5, 'std': 0.5}, 'Income': {'dist': 'normal', 'mean': 67000, 'std': 6000}, 'Has child under 6 years old': {'dist': 'binary', 'prob': 0.5}, 'Buy fast pass': {False: 96, True: 4}},
                                          '4': {'Number of family members': {'dist': 'uniform', 'min': 3, 'max': 5}, 'Income': {'dist': 'normal', 'mean': 67000, 'std': 6000}, 'Has child under 6 years old': {'dist': 'binary', 'prob': 1-0.2}, 'Buy fast pass': {False: 1, True: 12}},
                                          '5': {'Number of family members': {'dist': 'uniform', 'min': 3, 'max': 5}, 'Income': {'dist': 'normal', 'mean': 67000, 'std': 6000}, 'Has child under 6 years old': {'dist': 'binary', 'prob': 0.8}, 'Buy fast pass': {False: 54, True: 13}}}



base_df = pd.DataFrame(columns=select_feature_cols + [select_class_col_name])
# for r in decision_tree_customer_types.values():
for r in decision_tree_customer_types_draw_info.values():
    for buy_choice in [True, False]:
        base_df = pd.concat([base_df, make_type_df(input_row_dict=r, feature_cols=select_feature_cols, buy_TF=buy_choice, scale_factor=50)], axis=0, ignore_index=True)

# ------------- NLP ------------------
# open orig Kaggle .csv in sublime and conver to UTF-8

df_reviews = pd.read_csv(os.path.join(fp, fn_nlp_csv), encoding='utf-8')
if save_TF:
    df_reviews['Review_Text'].to_csv('disney_tweets.csv', index=False)

# use sublime to save disney_tweets.csv --> disney_tweets.txt

with open(os.path.join(fp, fn_nlp_txt), 'r') as f:
    orig_text = f.read()
split_text = orig_text.split("\n")  # break into separate tweets by line

sia_model = SentimentIntensityAnalyzer()
sia_scores = [sia_model.polarity_scores(obs)['compound'] >= 0 for obs in split_text]

# add NLP to base datadrame (not all are needed)
base_df['review'] = [split_text[i] for i in random.sample(range(len(sia_scores)), base_df.shape[0])]    # takes random sample of tweets

# for selected comments added to dataframe, use model to classify as positive/negative
base_df['sentiment_positive_TF'] = [sia_model.polarity_scores(obs['review'])['compound'] >= 0 for i, obs in base_df.iterrows()]

# tie number of visits to positive / negative experience
mean_num_visits_based_on_positive_review = {True: {'mean': 3.5, 'std': 0.6}, False: {'mean': 1.8, 'std': 0.2}}
base_df['Number of annual visits'] = [max(1, int(np.random.normal(mean_num_visits_based_on_positive_review[r['sentiment_positive_TF']]['mean'],
                                                       mean_num_visits_based_on_positive_review[r['sentiment_positive_TF']]['std']))) for i, r in base_df.iterrows()]

# ----------- Clustering for ad response -------------
sse = list()
max_num_clusters = 5
for n in range(1, max_num_clusters):
    kmeans_model = KMeans(n_clusters=n)
    kmeans_model.fit(X=base_df[select_feature_cols])
    sse.append(kmeans_model.inertia_)
plt.plot(list(range(1, max_num_clusters)), sse)

select_num_clusters = 2
kmeans_model = KMeans(n_clusters=select_num_clusters)
kmeans_model.fit(X=base_df[select_feature_cols])
kmeans_predictions = kmeans_model.predict(X=base_df[select_feature_cols])
base_df['Customer Type'] = kmeans_predictions

# associate level of merchandise spend with response to ad exposure (cluster)
cluster_spend_mean = {0: 30, 1: 50}
base_df['Merchandise Spend'] = [max(0, round(np.random.normal(cluster_spend_mean[r['Customer Type']], 4), 2)) for i, r in base_df.iterrows()]


# ------------------------------------------------
# shuffle base_df
base_df = base_df.sample(frac=1)

# split test/train, remove labels from test data set, split into sub model data sets
frac_of_data_for_hist = 0.6
historical_data_index = base_df.sample(frac=0.6).index
future_data_index = [i for i in base_df.index if i not in historical_data_index]

df_historical = base_df.loc[historical_data_index]
df_future = base_df.loc[future_data_index]

drop_cols_historical = ['sentiment_positive_TF', 'Customer Type']
drop_cols_future = drop_cols_historical + ['Buy fast pass', 'Number of annual visits', 'Merchandise Spend']

# ---- SAVE -----
if save_TF:
    df_historical.drop(columns=drop_cols_historical).to_csv(os.path.join(fp_output, fn_historical_data), index=False)
    df_future.drop(columns=drop_cols_future).to_csv(os.path.join(fp_output, fn_future_data), index=False)

