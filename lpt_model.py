import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

def pred(df):

    X = df

    # Load the trained model
    model = joblib.load('LPT_1hour_gradient_boosting_model_V2_220124.joblib')
    # print(df)
    # for i in df.columns:
    #     print(df[i], df[i].dtype)

    # Make predictions
    prediction = model.predict(X)

    result = prediction[0]



    return result # 0: down, 1:UP, 2:no change on price



