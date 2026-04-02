import pytest
from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io
import os

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "services" in response.json()

def test_analyze_skin_endpoint():
    # Create a dummy image for testing
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/api/v1/analyze_skin",
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "accuracy" in data
    assert "llm_recommendation" in data
    assert "id" in data

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup test uploads after tests."""
    def remove_test_uploads():
        upload_dir = "data/uploads"
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if f.endswith(".jpg"):
                    os.remove(os.path.join(upload_dir, f))
    request.addfinalizer(remove_test_uploads)
