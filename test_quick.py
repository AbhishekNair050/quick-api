"""
Quick test of the Quick-API functionality
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

print("🔄 Testing Quick-API...")

# Create and train a simple model
print("1. Creating sample data...")
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

print("2. Training model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print("3. Saving model...")
model_path = "test_model.pkl"
joblib.dump(model, model_path)

print("4. Testing Quick-API import...")
try:
    from quick_api import create_api
    print("   ✅ Import successful!")
    
    print("5. Creating API...")
    api = create_api(model_path, title="Test API")
    print("   ✅ API created successfully!")
    
    print("6. Testing model info...")
    app = api.get_app()
    print(f"   ✅ FastAPI app created: {type(app)}")
    
    print("\n🎉 All tests passed! Quick-API is working correctly.")
    print(f"📄 Model saved as: {model_path}")
    print("\n💡 To start the API server, run:")
    print(f"   quick-api serve {model_path}")
    print("   or")
    print("   python -c \"from quick_api import create_api; api = create_api('test_model.pkl'); api.run()\"")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
