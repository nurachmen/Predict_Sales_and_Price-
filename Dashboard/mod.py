import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


 

df = pd.read_csv(r'C:\Users\Nurachmen\Documents\Data_Analisis\Dashboard Units Sold\sales.csv')

# data_fix = df[['price','retail_price','units_sold','uses_ad_boosts','rating','rating_count',
#            'rating_five_count','rating_four_count','rating_three_count','badges_count',
#            'badge_local_product','badge_product_quality','badge_fast_shipping','shipping_option_price',
#            'merchant_rating_count','merchant_rating']]

data_fix = df[['price','uses_ad_boosts','retail_price','rating_count',
           'rating','badge_local_product','badge_product_quality','badge_fast_shipping',
           'merchant_rating_count','merchant_rating','units_sold']]

data = data_fix.drop('units_sold', axis = 1)
target = data_fix['units_sold']

X_train, X_test, y_train, y_test = train_test_split(
    data, 
    target, 
    random_state=101, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# model_fix1 = XGBRegressor(colsample_bytree=1,
#                  gamma=0,                 
#                  learning_rate=0.300000012,
#                  base_score=0.5,
#                  max_depth=6,
#                  min_child_weight=1,
#                  n_estimators=10000,
#                  n_jobs=0,
#                  reg_alpha=0.75,
#                  reg_lambda=0.45,
#                  subsample=0.6,
#                  seed=42,
#                  booster='gbtree',
#                  scale_pos_weight=1,
#                  random_state = 101,
#                  verbosity=None)

rf_model =  RandomForestRegressor(bootstrap =  True,
                       max_depth = 80,
                       max_features = 3,
                       min_samples_leaf = 4,
                       min_samples_split = 8,
                       n_estimators = 300,
                       random_state = 101)

RFFmodel = rf_model.fit(X_train, y_train)
# XGBmodel = model_fix1.fit(X_train, y_train)

print('Random Forest Regressor model score',round((RFFmodel.score(X_train, y_train)*100),2),'%')
# print('XGB Regressor model score',round((XGBmodel.score(X_train, y_train)*100),2),'%')
import joblib
# joblib.dump(XGBmodel,'modeljoblib')
joblib.dump(RFFmodel,'modeljoblib')