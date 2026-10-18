# model.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 60)
print("🚢 TITANIC KURTULMA TAHMİNİ - MODEL EĞİTİMİ")
print("=" * 60)

# ============================================
# ADIM 1: VERİYİ OKU
# ============================================
print("\n📂 Veri okunuyor...")
df = pd.read_csv('titanic.csv')
print(f"   {len(df)} yolcu verisi yüklendi.")
print(f"   Sütunlar: {list(df.columns)}")

# ============================================
# ADIM 2: VERİ ÖNİŞLEME
# ============================================
print("\n🔧 Veri önişleme başlıyor...")

# 1. Gereksiz sütunları çıkar
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 2. Eksik verileri doldur
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# 3. Kategorik verileri sayısallaştır
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])  # female=0, male=1

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])  # S=0, C=1, Q=2

print("   ✅ Veri temizlendi!")

# ============================================
# ADIM 3: FEATURE ENGINEERING
# ============================================
print("\n⚙️ Yeni özellikler oluşturuluyor...")

# 1. Aile büyüklüğü
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 2. Yaş grupları
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=[0, 1, 2, 3, 4])

print("   ✅ Yeni özellikler eklendi!")
print(f"   Sütunlar: {list(df.columns)}")

# ============================================
# ADIM 4: VERİYİ BÖL
# ============================================
print("\n📊 Veri bölünüyor...")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Eğitim: {len(X_train)} örnek")
print(f"   Test: {len(X_test)} örnek")

# ============================================
# ADIM 5: ÖLÇEKLENDİR
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Veri ölçeklendirildi!")

# ============================================
# ADIM 6: MODEL EĞİT
# ============================================
print("\n🌲 Random Forest modeli eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ============================================
# ADIM 7: MODEL DEĞERLENDİR
# ============================================
print("\n📈 Model değerlendiriliyor...")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"   Test Doğruluğu: {accuracy*100:.2f}%")

print("\n📋 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Öldü', 'Hayatta']))

# Konfüzyon matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Öldü', 'Hayatta'],
            yticklabels=['Öldü', 'Hayatta'])
plt.title('Konfüzyon Matrisi')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("   📊 Konfüzyon matrisi kaydedildi: confusion_matrix.png")

# ============================================
# ADIM 8: ÖZELLİK ÖNEMİ
# ============================================
print("\n📊 Özellik önemi:")

feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': model.feature_importances_
}).sort_values('Önem', ascending=False)

for i, row in feature_importance.iterrows():
    print(f"   {row['Özellik']:15s}: {row['Önem']*100:.2f}%")

# ============================================
# ADIM 9: MODELİ KAYDET
# ============================================
print("\n💾 Model kaydediliyor...")

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_sex, 'le_sex.pkl')
joblib.dump(le_embarked, 'le_embarked.pkl')

print("   ✅ model.pkl kaydedildi!")
print("   ✅ scaler.pkl kaydedildi!")
print("   ✅ le_sex.pkl kaydedildi!")
print("   ✅ le_embarked.pkl kaydedildi!")

print("\n" + "=" * 60)
print("✅ Model eğitimi tamamlandı!")
print("=" * 60)