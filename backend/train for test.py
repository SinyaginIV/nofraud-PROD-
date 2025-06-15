import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Пути
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Чтение датасета
df = pd.read_csv('fraud_dataset_real.csv')

# Проверка на наличие нужных колонок
required_columns = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight', 'Fraud']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing_columns}")

# Обучение энкодеров
le_region = LabelEncoder()
le_device = LabelEncoder()
df['Region'] = le_region.fit_transform(df['Region'])
df['DeviceType'] = le_device.fit_transform(df['DeviceType'])

# Сохраняем энкодеры
joblib.dump(le_region, os.path.join(MODEL_DIR, 'le_region.pkl'))
joblib.dump(le_device, os.path.join(MODEL_DIR, 'le_device.pkl'))

# Разделяем данные
features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight']
X = df[features]
y = df['Fraud']

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Сохраняем модель
joblib.dump(model, os.path.join(MODEL_DIR, 'real_model.pkl'))

print("✅ Модель и энкодеры успешно обучены и сохранены.")