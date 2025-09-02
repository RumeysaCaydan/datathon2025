import pandas as pnd
import matplotlib.pyplot as plt


df = pnd.read_csv("train_processed.csv")
print("Min:", df['session_value'].min())
print("Max:", df['session_value'].max())
print("avg:", df['session_value'].mean())
print("Median:", df['session_value'].median())
print("Standart Deviation:", df['session_value'].std())
plt.hist(df['session_value'], bins=30, edgecolor='black'); plt.title("Session Value Distribution"); plt.show()

# --------- Feature engineering----------
data = pnd.read_csv('trainset/train.csv')
data['event_time'] = pnd.to_datetime(data['event_time'])


span = (data.groupby('user_session')['event_time']
          .agg(start='min', end='max').reset_index())
span['session_duration_sec'] = (span['end'] - span['start']).dt.total_seconds()
span['start_hour'] = span['start'].dt.hour
span['start_weekday'] = span['start'].dt.weekday          # 0=Mon, 6=Sun
span['is_weekend'] = span['start_weekday'].isin([5,6]).astype(int)
span['is_night'] = ((span['start_hour'] >= 22) | (span['start_hour'] <= 6)).astype(int)
span = span[['user_session','session_duration_sec','start_hour','start_weekday','is_weekend','is_night']]


uniq = (data.groupby('user_session')
        .agg(unique_products=('product_id','nunique'),
             unique_categories=('category_id','nunique'))
        .reset_index())


session_event_count = data.groupby('user_session').size().reset_index(name='event_count')


event_type_count = (data.groupby(['user_session','event_type'])
                      .size().unstack(fill_value=0).reset_index())
for col in ['VIEW','ADD_CART','REMOVE_CART','PURCHASE']:
    if col not in event_type_count.columns:
        event_type_count[col] = 0


fe = (event_type_count
      .merge(session_event_count, on='user_session', how='left')
      .merge(uniq,                on='user_session', how='left')
      .merge(span,                on='user_session', how='left')
     ).fillna(0)

fe['cart_ratio'] = fe['ADD_CART'] / (fe['event_count'] + 1)
fe['view_ratio'] = fe['VIEW'] / (fe['event_count'] + 1)
fe['product_diversity']  = fe['unique_products']  / (fe['event_count'] + 1)
fe['category_diversity'] = fe['unique_categories'] / (fe['event_count'] + 1)
fe['actions_per_minute'] = fe['event_count'] / (fe['session_duration_sec']/60 + 1)
fe['has_purchase'] = (fe['PURCHASE'] > 0).astype(int)
fe['avg_gap_sec'] = fe['session_duration_sec'] / fe['event_count'].clip(lower=1)

session_value = data[['user_session','session_value']].drop_duplicates()
result_train = fe.merge(session_value, on='user_session', how='left')
result_train.to_csv('train_processed.csv', index=False)
