from main import app
from sanic_testing.testing import SanicTestClient

test_client = SanicTestClient(app)


def test_get_vroot():
    _, response = test_client.get("/")
    assert response.json == {
            "statusText": "Root Endpoint of CV-API-V3"
        }
    assert response.status_code == 200


def test_get_version():
    _, response = test_client.get("/version")
    assert response.json == {
            "statusText": "Version Fetch Successful",
            "version" : "3.0.0"
        }
    assert response.status_code == 200