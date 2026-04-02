import pytest
from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io
import os
import json

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "database" in data["services"]
    assert "disk" in data["services"]
    assert "llm" in data["services"]

def test_analyze_skin_endpoint():
    """Test the unified skin analysis streaming endpoint."""
    # Create a dummy image for testing
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Form data required by the updated API
    payload = {
        "user_id": "TEST-PATIENT-001",
        "patient_name": "Test User",
        "age": 30
    }

    # Use stream=True to handle the StreamingResponse
    response = client.post(
        "/api/v1/analyze_skin",
        data=payload,
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 200
    
    # The response is a stream. The first part is JSON metadata followed by ||METADATA_END||
    content = response.text
    assert "||METADATA_END||" in content
    
    parts = content.split("||METADATA_END||")
    metadata = json.loads(parts[0])
    
    assert metadata["user_id"] == payload["user_id"]
    assert metadata["patient_name"] == payload["patient_name"]
    assert "prediction" in metadata
    assert "accuracy" in metadata
    
    # Ensure there is some LLM recommendation text after the metadata
    assert len(parts[1]) > 0

def test_history_endpoint():
    """Test retrieving history for a patient."""
    user_id = "TEST-PATIENT-001"
    response = client.get(f"/api/v1/history/{user_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_list_models():
    """Test listing available models."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "active_model" in data

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup test uploads after tests."""
    def remove_test_uploads():
        upload_dir = "data/uploads"
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                # Only remove files that look like UUIDs generated during tests
                if len(f.split('.')[0]) == 36: 
                    try:
                        os.remove(os.path.join(upload_dir, f))
                    except:
                        pass
    request.addfinalizer(remove_test_uploads)
