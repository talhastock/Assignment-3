from pathlib import Path

def test_dockerfile_exists():
    assert Path("Dockerfile").exists()
    assert "FROM python" in Path("Dockerfile").read_text()