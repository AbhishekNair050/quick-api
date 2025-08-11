"""
Complete demonstration of Quick-API functionality
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import requests
import time
import threading
from quick_api import create_api


def train_and_save_model():
    """Train and save a sample model"""
    print("🔄 Creating and training a sample model...")
    
    # Create a realistic dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("📊 Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = "demo_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    return model_path, X_test[:5]  # Return model path and sample test data


def start_api_server(model_path):
    """Start the API server in a separate thread"""
    def run_server():
        api = create_api(
            model_path=model_path,
            title="Quick-API Demo",
            description="A demonstration of Quick-API with a RandomForest classifier",
            version="1.0.0",
            host="localhost",
            port=8000
        )
        api.run()
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("🚀 Starting API server...")
    time.sleep(3)  # Give server time to start
    
    return server_thread


def test_api_endpoints(sample_data):
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("\n🧪 Testing API endpoints...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"   ✅ Health check: {response.json()['status']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
        
        # Test info endpoint
        print("2. Testing info endpoint...")
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Model type: {info['model_type']}")
            print(f"   ✅ Available endpoints: {info['endpoints']}")
        else:
            print(f"   ❌ Info endpoint failed: {response.status_code}")
        
        # Test prediction endpoint
        print("3. Testing prediction endpoint...")
        prediction_data = {
            "data": sample_data.tolist()
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Predictions: {result['predictions']}")
            print(f"   ✅ Model type: {result['model_type']}")
            print(f"   ✅ Input shape: {result['input_shape']}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # Test single prediction
        print("4. Testing single sample prediction...")
        single_prediction_data = {
            "data": sample_data[0].tolist()
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=single_prediction_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Single prediction: {result['predictions']}")
        else:
            print(f"   ❌ Single prediction failed: {response.status_code}")
        
        print(f"\n📝 API Documentation available at: {base_url}/docs")
        print(f"🌐 Try the interactive API at: {base_url}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running.")
    except Exception as e:
        print(f"❌ Error testing API: {e}")


def demonstrate_cli():
    """Demonstrate CLI usage"""
    print("\n💻 CLI Usage Examples:")
    print("After installing quick-api, you can use these commands:")
    print()
    print("1. Get model information:")
    print("   quick-api info demo_classifier.pkl")
    print()
    print("2. Serve the model:")
    print("   quick-api serve demo_classifier.pkl")
    print()
    print("3. Serve with custom settings:")
    print("   quick-api serve demo_classifier.pkl --host 0.0.0.0 --port 8080 --title 'My API'")
    print()
    print("4. Serve in development mode:")
    print("   quick-api serve demo_classifier.pkl --reload")


def main():
    """Main demonstration function"""
    print("🎯 Quick-API Complete Demonstration")
    print("=" * 50)
    
    # Train and save model
    model_path, sample_data = train_and_save_model()
    
    # Start API server
    server_thread = start_api_server(model_path)
    
    # Test API endpoints
    test_api_endpoints(sample_data)
    
    # Demonstrate CLI
    demonstrate_cli()
    
    print("\n" + "=" * 50)
    print("🎉 Demonstration complete!")
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ One-line API creation")
    print("   ✅ Automatic model loading")
    print("   ✅ FastAPI integration")
    print("   ✅ JSON input/output handling")
    print("   ✅ Health and info endpoints")
    print("   ✅ Interactive documentation")
    print("   ✅ CLI interface")
    
    print(f"\n🔧 Next Steps:")
    print(f"   1. Visit http://localhost:8000/docs for interactive API docs")
    print(f"   2. Try making predictions with curl or your favorite HTTP client")
    print(f"   3. Check out the examples/ directory for more use cases")
    print(f"   4. Install with: pip install quick-api")
    
    # Keep the server running
    print(f"\n⏳ Server will keep running. Press Ctrl+C to stop.")
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n👋 Shutting down demo...")


if __name__ == "__main__":
    main()
