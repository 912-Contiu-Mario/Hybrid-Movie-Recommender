#!/usr/bin/env python


import sys
import logging
from app.training.lightgcn_training_service import main
from app.config.logging import setup_logging

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting LightGCN training service runner")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
        sys.exit(1) 