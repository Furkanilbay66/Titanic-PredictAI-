# test_api.py
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("\n" + "=" * 50)
    print("🏥 Health Check Testi")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_predict():
    print("\n" + "=" * 50)
    print("🔮 Tahmin Testi")
    print("=" * 50)
    
    test_data = [
        {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 1, "Parch": 0, "Fare": 100, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
        {"Pclass": 1, "Sex": "male", "Age": 35, "SibSp": 0, "Parch": 0, "Fare": 50, "Embarked": "C"},
        {"Pclass": 3, "Sex": "female", "Age": 18, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "Q"}
    ]
    
    for i, data in enumerate(test_data, 1):
        print(f"\n📝 Test {i}: {data['Sex']}, {data['Age']} yaş, {data['Pclass']}. sınıf")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result['survival']} (%{result['probability']})")
        else:
            print(f"   ❌ Hata: {response.json()}")

def test_error():
    print("\n" + "=" * 50)
    print("❌ Hata Yönetimi Testi")
    print("=" * 50)
    
    test_cases = [
        {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 1, "Parch": 0, "Fare": 100},
        {"Pclass": 1, "Sex": "invalid", "Age": 25, "SibSp": 1, "Parch": 0, "Fare": 100, "Embarked": "S"}
    ]
    
    for i, data in enumerate(test_cases, 1):
        print(f"\n🔴 Test {i}: {data}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

if __name__ == "__main__":
    print("🚀 API Test Başlatılıyor...")
    print("⚠️  Önce API'yi başlatın: python app.py")
    print("=" * 50)
    
    try:
        test_health()
        test_predict()
        test_error()
        print("\n✅ Tüm testler tamamlandı!")
    except requests.exceptions.ConnectionError:
        print("\n❌ API'ye bağlanılamadı! app.py'yi çalıştırın.")