import os
import time
import logging
import requests
import argparse
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from app.training.train_model import train_and_savelightgcn_model
from app.db.database import SessionLocal
from app.repositories.sqlite import SQLiteRatingRepository
from app.config import get_training_config, paths
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LightGCNTrainingService:
    def __init__(self, config=None, api_url=None):
        self.config = config or get_training_config()
        self.api_url = api_url or self.config.get('api_url', 'http://localhost:8000')
        self.scheduler = BlockingScheduler()
        
    def train_and_notify(self):
        db = SessionLocal()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"model_{timestamp}"
            
            logger.info(f"Starting training for version {version}")
            
            rating_repo = SQLiteRatingRepository(db)
            
            result = train_and_savelightgcn_model(
                rating_repo=rating_repo,
                version=version,
                config=self.config,
                save_model=True
            )
            
            test_results = result['test_results']
            logger.info(f"Training completed. Test NDCG@{self.config.get('k', 20)}: {test_results.get(f'ndcg@{self.config.get('k', 20)}', 0):.4f}")
            
            self._notify_api(version)
            
            return result
        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
        finally:
            db.close()
    
    def _notify_api(self, version):
        try:
            reload_url = f"{self.api_url}/recommendations/reload"
            response = requests.post(
                reload_url,
                json={"version": version},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"API successfully notified about new model {version}")
            else:
                logger.error(f"Failed to notify API: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error notifying API: {str(e)}")
    
    def schedule_training(self, interval_hours=48, start_hour=2):
        self.scheduler.add_job(
            self.train_and_notify,
            'cron',
            hour=start_hour,
            id='lightgcn_training'
        )
        logger.info(f"Scheduled LightGCN training to run daily at {start_hour}:00")
        
        # self.scheduler.add_job(
        #     self.train_and_notify,
        #     'interval',
        #     hours=interval_hours,
        #     id='lightgcn_training'
        # )
        # logger.info(f"Scheduled LightGCN training to run every {interval_hours} hours")
    
    def run_once(self):
        logger.info("Running one-time training process")
        return self.train_and_notify()
    
    def start(self):
        logger.info("Starting LightGCN training service")
        
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Training service stopped")
        
def main():
    parser = argparse.ArgumentParser(description='LightGCN Training Service')
    parser.add_argument('--run-once', action='store_true', help='Run training once without scheduling')
    parser.add_argument('--interval', type=int, default=24, help='Training interval in hours (default: 24)')
    parser.add_argument('--start-hour', type=int, default=2, help='Hour to run scheduled training (default: 2)')
    parser.add_argument('--api-url', type=str, help='API service URL (default: from config)')
    
    args = parser.parse_args()
    
    service = LightGCNTrainingService(api_url=args.api_url)
    
    if args.run_once:
        service.run_once()
    else:
        service.schedule_training(interval_hours=args.interval, start_hour=args.start_hour)
        service.start()

if __name__ == "__main__":
    main() 