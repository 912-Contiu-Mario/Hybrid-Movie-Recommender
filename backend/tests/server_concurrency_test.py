import asyncio
import httpx
import time
import logging
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("server_test")

class ServerConcurrencyTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.reload_complete = False
    
    async def trigger_model_reload(self, version="latest"):
        """Trigger a model reload via the reload endpoint"""
        logger.info(f"Triggering LightGCN model reload at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Start timing the reload
                reload_start = time.time()
                
                # Fire the reload request
                response = await client.post(
                    f"{self.base_url}/recommendations/reload?version={version}"
                )
                
                reload_end = time.time()
                reload_duration = reload_end - reload_start
                
                if response.status_code != 200:
                    logger.error(f"Failed to trigger model reload: {response.status_code} - {response.text}")
                    return False, 0
                
                logger.info(f"Model reload completed in {reload_duration:.2f}s. Response: {response.json()}")
                self.reload_complete = True
                return True, reload_duration
        except Exception as e:
            logger.error(f"Error triggering model reload: {e}")
            return False, 0
    
    async def make_recommendation_request(self, user_id, req_id, reload_in_progress_event):
        """Make a recommendation request and check if it runs during reload"""
        start_time = time.time()
        concurrent_with_reload = False
        
        logger.info(f"Request {req_id}: Starting at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/recommendations/lightgcn?user_id={user_id}&num_recs=10"
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Check if this request overlapped with the reload
                concurrent_with_reload = not self.reload_complete
                
                if response.status_code == 200:
                    status = "CONCURRENT with model reload" if concurrent_with_reload else "AFTER model reload"
                    results = response.json()
                    rec_count = len(results.get("recommendations", []))
                    logger.info(f"Request {req_id}: Got {rec_count} recommendations in {duration:.2f}s - {status}")
                    return {
                        "success": True, 
                        "concurrent_with_reload": concurrent_with_reload, 
                        "duration": duration,
                        "recommendations": rec_count
                    }
                else:
                    logger.error(f"Request {req_id}: Failed with status {response.status_code} in {duration:.2f}s")
                    return {
                        "success": False, 
                        "concurrent_with_reload": concurrent_with_reload, 
                        "duration": duration,
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"Request {req_id}: Exception: {str(e)} in {duration:.2f}s")
                return {
                    "success": False, 
                    "concurrent_with_reload": concurrent_with_reload, 
                    "duration": duration,
                    "error": str(e)
                }
    
    async def run_test(self, user_id=1, request_count=10, model_version="latest"):
        """Run the concurrency test with true overlap between reload and requests"""
        print("\n==== SERVER CONCURRENCY TEST (MODEL RELOAD) ====")
        print(f"Testing server at {self.base_url}")
        print("Make sure your server is running first!\n")
        
        # Event to track reload progress
        reload_in_progress = asyncio.Event()
        reload_in_progress.set()  # Mark as in progress initially
        
        # Create tasks for all recommendation requests
        recommendation_tasks = []
        for i in range(request_count):
            task = asyncio.create_task(
                self.make_recommendation_request(user_id, i, reload_in_progress)
            )
            recommendation_tasks.append(task)
            
            # Small stagger to spread requests over time
            if i < request_count - 1:  # Don't wait after the last one
                await asyncio.sleep(0.05)
        
        # Fire the model reload in parallel with the recommendation requests
        reload_task = asyncio.create_task(self.trigger_model_reload(model_version))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*recommendation_tasks)
        reload_success, reload_duration = await reload_task
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        during_reload = [r for r in successful if r["concurrent_with_reload"]]
        after_reload = [r for r in successful if not r["concurrent_with_reload"]]
        
        print("\n==== TEST RESULTS ====")
        print(f"Model reload duration: {reload_duration:.2f}s")
        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {len(successful)}")
        print(f"Requests during model reload: {len(during_reload)}")
        print(f"Requests after model reload: {len(after_reload)}")
        
        if len(during_reload) > 0:
            print("\n✅ SUCCESS: Recommendations ran concurrently with model reload")
            avg_time = sum(r["duration"] for r in during_reload) / len(during_reload)
            print(f"   Average response time during model reload: {avg_time:.2f}s")
            
            if len(after_reload) > 0:
                avg_time_after = sum(r["duration"] for r in after_reload) / len(after_reload)
                print(f"   Average response time after model reload: {avg_time_after:.2f}s")
        else:
            print("\n❌ FAILURE: No recommendations completed during model reload")
            print("   Possible reasons:")
            print("   - Model reload is too fast")
            print("   - Recommendations all waited for model reload to complete")
            print("   - Server is handling requests sequentially")

async def run_test(base_url="http://localhost:8000", model_version="latest"):
    tester = ServerConcurrencyTest(base_url)
    await tester.run_test(model_version=model_version)

if __name__ == "__main__":
    # If a model version is provided as a command-line argument, use it
    if len(sys.argv) > 1:
        model_version = sys.argv[1]
        print(f"Using model version: {model_version}")
        asyncio.run(run_test(model_version=model_version))
    else:
        asyncio.run(run_test()) 