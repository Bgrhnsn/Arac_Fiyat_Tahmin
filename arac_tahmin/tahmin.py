import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('car_price_prediction.csv')

# id ve doors sütunları bizim için gereksiz
data = data.drop(['ID', 'Doors'], axis=1)

# araç yaşını hesaplama
dtime = datetime.datetime.now()
data['Age'] = dtime.year - data['Prod. year']
data = data.drop(['Prod. year'], axis=1)

# vergi kısmındaki boş yerlere 0 yazdırma
data['Levy'].replace({'-': 0}, inplace=True)
data['Levy'] = data['Levy'].astype('int')

# mileage sütununda km ifadesini kaldırma
data['Mileage'] = data['Mileage'].str.replace('km', '').astype('int')

# engine volumde 1.2 turbo yazan değeri kaldırma 
data['Engine volume'] = data['Engine volume'].str.replace('Turbo', '').astype('float')

# sadece sayısal sütunlarda uç değer analizi yapmak.
data_numeric = data.select_dtypes(exclude='object')

# her bir sayısal sütun için uç değerlerin dışındaki satırları veri setinden çıkarmak ve veri setini temizlemek.
for col in data_numeric.columns:
    q1 = data_numeric[col].quantile(0.25)
    q3 = data_numeric[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - (1.5 * iqr)
    high = q3 + (1.5 * iqr)
    data = data.loc[(data[col] <= high) & (data[col] >= low)]

# kategorik verileri Label Encoding ile dönüştürme
label_encoders = {}
data_object = data.select_dtypes(include='object')
for col in data_object.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# bağımsız ve bağımlı değişkenleri ayırma
x = data.drop('Price', axis=1)
y = data['Price']

# eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# modelleri tanımlama
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR()
}


results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
    score = model.score (x_test,y_test)
    print(f"{name} - R2: {r2}, RMSE: {rmse}")

# modellerin tablo haline getirilmesi
results_df = pd.DataFrame(results)
print(results_df)


# en iyi modelin seçimi
best_model = RandomForestRegressor(random_state=42)
best_model.fit(x_train, y_train)

# örnek araba oluşturma
new_car = {
    'Manufacturer': 'FIAT',
    'Model': 'Punto',
    'Age': dtime.year - 2010,
    'Category': 'Sedan',
    'Fuel type': 'Diesel',
    'Engine volume': 1.6,
    'Mileage': 2000,
    'Gear box type': 'Manual',
    'Drive wheels': 'Front',
    'Wheel': 'Left wheel',
    'Color': 'Black',
    'Registration_year': 2010,
    'Registration_month': 5,
    'Registration_day': 4
}

# yeni araç verisini uygun formata dönüştürme
new_car_df = pd.DataFrame([new_car])

# kategorik sütunları Label Encoding ile dönüştürme
for col, le in label_encoders.items():
    if col in new_car_df.columns:
        new_car_df[col] = new_car_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# yeni araç verisinin eksik sütunlarını doldurma
new_car_df = new_car_df.reindex(columns=x.columns, fill_value=0)

# dolar üzerinden fiyat tahmini
predicted_price_usd = best_model.predict(new_car_df)[0]
print(f"USD Olarak Tahmin Edilen Fiyat: {predicted_price_usd}")

# döviz kuru
exchange_rate = 32

# ÖTV ve KDV oranları
otv_rate = 0.45  # %45
kdv_rate = 0.18  # %18

# Fiyatı Türk Lirasına çevirme
predicted_price_try = predicted_price_usd * exchange_rate
otv = predicted_price_try * otv_rate
kdv = (predicted_price_try + otv) * kdv_rate

total_price_try = predicted_price_try + otv + kdv
print(f"TRY Olarak Tahmin Edilen Fiyat : {total_price_try}")



# Görselleştirme 
data['Manufacturer'].unique()
import pandas as pd
manufacturer_counts = data['Manufacturer'].value_counts()
for manufacturer, count in manufacturer_counts.items():
    print(f"{manufacturer}: {count}")
top10=data['Manufacturer'].value_counts().sort_values(ascending=False) [:10]
top10.plot(figsize= (15,3))
plt.show

# R2 değerlerini görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title('R² Scores of Different Models')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)  # R² skoru 0 ile 1 arasında olduğu için
plt.show()


# RMSE değerlerini görselleştirme
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results_df, palette='viridis')
plt.title('RMSE of Different Models')
plt.xlabel('Model')
plt.ylabel('RMSE')



# R2 ve RMSE değerlerini görselleştirme
plt.figure(figsize=(14, 6))

# R2 değerlerini görselleştirme
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title('R² Scores of Different Models')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)

# RMSE değerlerini görselleştirme
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results_df, palette='viridis')
plt.title('RMSE of Different Models')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# parametrelerin değerlerinin görselleştirmesi
importances = best_model.feature_importances_
features = x.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances in Random Forest')
plt.show()

# farklı kilometrelerin fiyat üzerindeki etkisi
mileages = [1000, 50000, 100000, 150000, 200000]
predicted_prices = []

for mileage in mileages:
    new_car_df['Mileage'] = mileage
    predicted_price_usd = best_model.predict(new_car_df)[0]
    predicted_prices.append(predicted_price_usd)
    
plt.figure(figsize=(10, 6))
plt.plot(mileages, predicted_prices, marker='o')
plt.title('Predicted Price vs Mileage')
plt.xlabel('Mileage')
plt.ylabel('Predicted Price (USD)')
plt.show()


# arabanın yaşı ve fiyat arasındaki ilişkinin görselleştirilmesi
ages = [1, 5, 10, 15, 20]  # Aracın yaşları
predicted_prices_age = []

for age in ages:
    new_car_df['Age'] = age
    predicted_price_usd = best_model.predict(new_car_df)[0]
    predicted_prices_age.append(predicted_price_usd)

plt.figure(figsize=(10, 6))
plt.plot(ages, predicted_prices_age, marker='o')
plt.title('Predicted Price vs Age')
plt.xlabel('Age (years)')
plt.ylabel('Predicted Price (USD)')
plt.show()


# Orijinal yakıt türlerini geri dönüştürüyoruz
data['Fuel type'] = label_encoders['Fuel type'].inverse_transform(data['Fuel type'])

# Mevcut yakıt türlerini çekiyoruz
fuel_types = data['Fuel type'].unique()

predicted_prices_fuel = []

for fuel_type in fuel_types:
    # Yeni araç verisinin uygun formata dönüştürülmesi
    new_car_df['Fuel type'] = fuel_type
    
    # Eğitim setinde olmayan değerler için kontrol
    if fuel_type in label_encoders['Fuel type'].classes_:
        new_car_df['Fuel type'] = label_encoders['Fuel type'].transform(new_car_df['Fuel type'])
    else:
        # Eğitim setinde olmayan değerler için -1 veya uygun bir değer atıyoruz
        new_car_df['Fuel type'] = -1
    
    predicted_price_usd = best_model.predict(new_car_df)[0]
    predicted_prices_fuel.append(predicted_price_usd)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(fuel_types, predicted_prices_fuel, color=['blue', 'green', 'red', 'purple'])
plt.title('Predicted Price vs Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Predicted Price (USD)')
plt.show()