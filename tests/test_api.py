import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from PIL import Image
import io
import os
import json
import anyio
from datetime import datetime, timezone

# We use anyio_backend to tell pytest-anyio to use asyncio
@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_endpoint(async_client):
    """Test the health check endpoint for all services."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "services" in data
    assert "database" in data["services"]
    assert "llm" in data["services"]
    assert "skin_model" in data["services"]


@pytest.mark.anyio
async def test_analyze_skin_endpoint(async_client):
    """Test the unified skin analysis streaming endpoint with a valid image."""
    # Create a dummy image for testing
    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_data = img_byte_arr.getvalue()

    # Form data required by the updated API
    payload = {
        "user_id": "1011",
        "patient_name": "Test User",
        "age": "30" # Form data values are strings
    }

    response = await async_client.post(
        "/api/v1/analyze_skin",
        data=payload,
        files={"file": ("test_image.jpg", img_data, "image/jpeg")},
    )

    assert response.status_code == 200
    
    # Read the stream
    content = ""
    async for chunk in response.aiter_text():
        content += chunk
        
    assert "||METADATA_END||" in content

    parts = content.split("||METADATA_END||")
    metadata = json.loads(parts[0])

    assert metadata["user_id"] == payload["user_id"]
    assert metadata["patient_name"] == payload["patient_name"]
    assert "prediction" in metadata
    assert "accuracy" in metadata
    
    # Check if a heatmap was created
    assert "heatmap_path" in metadata
    assert os.path.exists(metadata["heatmap_path"])


@pytest.mark.anyio
async def test_analyze_skin_invalid_file(async_client):
    """Test the analysis endpoint with an invalid file type."""
    payload = {"user_id": "TEST", "patient_name": "Test", "age": "20"}
    response = await async_client.post(
        "/api/v1/analyze_skin",
        data=payload,
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    # Our file_validator should catch this and return 400
    assert response.status_code == 400


@pytest.mark.anyio
async def test_history_endpoint(async_client):
    """Test retrieving history for a patient."""
    user_id = "1001"
    response = await async_client.get(f"/api/v1/history/{user_id}")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
    # If test_analyze_skin_endpoint ran, we should have at least one record
    if len(history) > 0:
        assert history[0]["user_id"] == user_id


@pytest.mark.anyio
async def test_stats_endpoint(async_client):
    """Test the disease statistics endpoint."""
    response = await async_client.get("/api/v1/stats")
    assert response.status_code == 200
    stats = response.json()
    assert isinstance(stats, dict)
    # Keys should be disease names, values should be integers
    for disease, count in stats.items():
        assert isinstance(disease, str)
        assert isinstance(count, int)


@pytest.mark.anyio
async def test_list_models(async_client):
    """Test listing available models."""
    response = await async_client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "active_model" in data
    assert "efficientnet" in data["active_model"].lower()


@pytest.fixture(scope="session", autouse=True)
def cleanup_uploads():
    """Cleanup test uploads after the test session."""
    yield
    upload_dir = "data/uploads"
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            # Remove UUID-named files (36 chars + extension) and heatmaps
            name_part = f.split(".")[0]
            if len(name_part) == 36 or name_part.startswith("heatmap_"):
                try:
                    os.remove(os.path.join(upload_dir, f))
                except:
                    pass
