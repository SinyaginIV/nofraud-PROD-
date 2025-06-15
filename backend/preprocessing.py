
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

def preprocess_file(df):
    le_region = joblib.load('model/le_region.pkl')
    le_device = joblib.load('model/le_device.pkl')
    df['Region'] = le_region.transform(df['Region'])
    df['DeviceType'] = le_device.transform(df['DeviceType'])
    df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
    df['IsNight'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight']
    df[features] = RobustScaler().fit_transform(df[features])
    return df[features]
