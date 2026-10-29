from fastapi.testclient import TestClient

from app.main import create_app


def test_liveness_probe_returns_service_details() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_readiness_probe_is_ready() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_dashboard_and_assets_are_served() -> None:
    with TestClient(create_app()) as client:
        dashboard = client.get("/")
        stylesheet = client.get("/assets/styles.css")
    assert dashboard.status_code == 200
    assert "Make sense of the evidence" in dashboard.text
    assert stylesheet.status_code == 200


def test_agent_tools_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/agent/tools")
    assert response.status_code == 200
    tools = response.json()
    names = {t["name"] for t in tools}
    assert "research_documents" in names
    assert "search_corpus" in names
    assert "extract_metrics_table" in names
    assert "compare_documents" in names


def test_list_documents_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

