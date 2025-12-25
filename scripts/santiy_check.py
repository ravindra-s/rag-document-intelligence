from rag.config import settings
from rag.logging_config import configure_logging
import logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Project root: %s", settings.project_root)
    logger.info("Raw data dir: %s", settings.data_raw_dir)
    logger.info("Processed data dir: %s", settings.data_processed_dir)


if __name__ == "__main__":
    main()