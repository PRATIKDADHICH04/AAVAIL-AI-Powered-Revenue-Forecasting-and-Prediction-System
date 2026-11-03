import logging
from app.logging_config import get_logger

def test_logger_writes_to_test_file(tmp_path):
    logger_name = f"test_logger_{tmp_path.name}"
    path = tmp_path / "test.log"
    logger = get_logger(logger_name, str(path))
    logger.info("unit-test-message")
    for handler in logger.handlers:
        handler.flush()
    assert path.exists()
    assert "unit-test-message" in path.read_text()
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
