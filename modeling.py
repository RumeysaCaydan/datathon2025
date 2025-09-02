from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pandas as pd

#  read data
train_data = pd.read_csv("train_processed.csv")

# aim and properties  
y = train_data['session_value']
X = train_data.drop(columns=['session_value', 'user_session'], errors='ignore')
X = X.select_dtypes(include=['number'])

# 3) Train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

# 4) Modeling
model = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=7,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="rmse"   # fit içine vermeyen sürümler için burada
)

# 5) train
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 6) evaluation
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("MSE:", mse)
# ===== SUBMISSION OLUŞTURMA (modeling.py'nin en altına ekle) =====
import pandas as pd

#  read test to row data 
test_raw = pd.read_csv("test/test.csv")
test_raw['event_time'] = pd.to_datetime(test_raw['event_time'])

# produce propertys of session- level with same in train
span_t = (test_raw.groupby('user_session')['event_time']
            .agg(start='min', end='max').reset_index())
span_t['session_duration_sec'] = (span_t['end'] - span_t['start']).dt.total_seconds()
span_t['start_hour'] = span_t['start'].dt.hour
span_t['start_weekday'] = span_t['start'].dt.weekday
span_t['is_weekend'] = span_t['start_weekday'].isin([5,6]).astype(int)
span_t['is_night'] = ((span_t['start_hour'] >= 22) | (span_t['start_hour'] <= 6)).astype(int)
span_t = span_t[['user_session','session_duration_sec','start_hour','start_weekday','is_weekend','is_night']]

uniq_t = (test_raw.groupby('user_session')
          .agg(unique_products=('product_id','nunique'),
               unique_categories=('category_id','nunique'))
          .reset_index())

cnt_t = test_raw.groupby('user_session').size().reset_index(name='event_count')

evt_t = (test_raw.groupby(['user_session','event_type'])
           .size().unstack(fill_value=0).reset_index())
for col in ['VIEW','ADD_CART','REMOVE_CART','PURCHASE']:
    if col not in evt_t.columns:
        evt_t[col] = 0

fe_test = (evt_t
           .merge(cnt_t,   on='user_session', how='left')
           .merge(uniq_t,  on='user_session', how='left')
           .merge(span_t,  on='user_session', how='left')
          ).fillna(0)

# derivative features (exactly the same with train)
fe_test['cart_ratio'] = fe_test['ADD_CART'] / (fe_test['event_count'] + 1)
fe_test['view_ratio'] = fe_test['VIEW'] / (fe_test['event_count'] + 1)
fe_test['product_diversity']  = fe_test['unique_products']  / (fe_test['event_count'] + 1)
fe_test['category_diversity'] = fe_test['unique_categories'] / (fe_test['event_count'] + 1)
fe_test['actions_per_minute'] = fe_test['event_count'] / (fe_test['session_duration_sec']/60 + 1)
fe_test['has_purchase'] = (fe_test['PURCHASE'] > 0).astype(int)
fe_test['avg_gap_sec'] = fe_test['session_duration_sec'] / fe_test['event_count'].clip(lower=1)

# align exactly with train feature set 
train_processed = pd.read_csv("train_processed.csv")
feature_cols = [c for c in train_processed.columns if c not in ['session_value','user_session']]

X_test = fe_test.copy()
for c in feature_cols:
    if c not in X_test.columns:
        X_test[c] = 0
X_test = X_test[feature_cols]

#  estimate and submission file 
preds = model.predict(X_test)

submission = pd.DataFrame({
    "user_session": fe_test["user_session"],
    "session_value": preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv oluşturuldu")
