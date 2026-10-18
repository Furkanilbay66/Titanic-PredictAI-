# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ============================================
# MODELİ YÜKLE
# ============================================
print("📂 Model yükleniyor...")

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_sex = joblib.load('le_sex.pkl')
le_embarked = joblib.load('le_embarked.pkl')

print("✅ Model başarıyla yüklendi!")

# ============================================
# ANA SAYFA
# ============================================
@app.route('/')
def home():
    return render_template('index.html')

# ============================================
# TAHMİN ENDPOINT'İ
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Veriyi al
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Veri gönderilmedi!'}), 400
        
        # 2. Gerekli alanları kontrol et
        required = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for field in required:
            if field not in data:
                return jsonify({'error': f'{field} alanı zorunlu!'}), 400
        
        # 3. Veriyi dönüştür
        sex = data['Sex']
        embarked = data['Embarked']
        
        # Sex'i label encoding ile dönüştür
        sex_encoded = le_sex.transform([sex])[0]
        
        # Embarked'ı label encoding ile dönüştür
        embarked_encoded = le_embarked.transform([embarked])[0]
        
        # 4. DataFrame oluştur
        input_data = pd.DataFrame([{
            'Pclass': int(data['Pclass']),
            'Sex': sex_encoded,
            'Age': float(data['Age']),
            'SibSp': int(data['SibSp']),
            'Parch': int(data['Parch']),
            'Fare': float(data['Fare']),
            'Embarked': embarked_encoded,
            'FamilySize': int(data['SibSp']) + int(data['Parch']) + 1,
            'AgeGroup': int(pd.cut([float(data['Age'])], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=[0, 1, 2, 3, 4])[0])
        }])
        
        # 5. Ölçeklendir
        input_scaled = scaler.transform(input_data)
        
        # 6. Tahmin yap
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        
        # 7. Sonucu hazırla
        result = {
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'survival': 'Hayatta Kaldı' if prediction == 1 else 'Hayatını Kaybetti',
            'message': '✅ Tahmin başarıyla yapıldı!'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# SAĞLIK KONTROLÜ
# ============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    }), 200

# ============================================
# API DOKÜMANTASYONU
# ============================================
@app.route('/docs', methods=['GET'])
def docs():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic API Dokümantasyonu</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .endpoint { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
            pre { background: #2d2d2d; color: #f8f8f8; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>🚢 Titanic API Dokümantasyonu</h1>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Tahmin yapar</p>
            <pre>
{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 100,
    "Embarked": "S"
}</pre>
            <p><strong>Sex:</strong> "male" veya "female"</p>
            <p><strong>Embarked:</strong> "S", "C" veya "Q"</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>API sağlık kontrolü</p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("=" * 60)
    print("🚢 TITANIC API BAŞLATILIYOR...")
    print("=" * 60)
    print(f"📁 Ana Sayfa: http://localhost:5000/")
    print(f"📁 Dokümantasyon: http://localhost:5000/docs")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)