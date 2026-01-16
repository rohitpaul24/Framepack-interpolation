"""
Background worker for async video processing.
"""

import threading
import queue
import time
from datetime import datetime
from .jobs import Job, JobStatus
import traceback

# Global job queue
job_queue = queue.Queue()
worker_thread = None
is_running = False


def process_job(job: Job, process_func):
    """
    Process a single job.
    
    Args:
        job: Job instance
        process_func: Function to call for processing (process_video_multi_loops_api)
    """
    try:
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now().isoformat()
        job.progress = 0
        job.save()
        
        print(f"[Worker] Processing job {job.job_id}: {job.video_name}")
        
        # Call the processing function
        result = process_func(
            input_video_path=job.video_path,
            duration=job.params.get('duration'),
            include_loops=job.params.get('include_loops', False),
            include_trimmed=job.params.get('include_trimmed', False),
            include_interpolations=job.params.get('include_interpolations', True),
            job_id=job.job_id  # Pass job_id for progress updates
        )
        
        # Mark as completed
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        job.result = result
        job.progress = 100
        job.save()
        
        print(f"[Worker] Completed job {job.job_id}")
        
    except Exception as e:
        # Mark as failed
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now().isoformat()
        job.save()
        
        print(f"[Worker] Failed job {job.job_id}: {e}")
        traceback.print_exc()


def worker_loop(process_func):
    """
    Background worker that processes jobs from queue.
    
    Args:
        process_func: Function to call for processing
    """
    global is_running
    is_running = True
    
    print("[Worker] Background worker started")
    
    while is_running:
        try:
            # Get job from queue (timeout 1 second)
            job = job_queue.get(timeout=1)
            
            # Process the job
            process_job(job, process_func)
            
            # Mark task as done
            job_queue.task_done()
            
        except queue.Empty:
            # No jobs in queue, continue waiting
            continue
        except Exception as e:
            print(f"[Worker] Error in worker loop: {e}")
            traceback.print_exc()


def start_worker(process_func):
    """
    Start background worker thread.
    
    Args:
        process_func: Function to call for processing
    """
    global worker_thread, is_running
    
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(
            target=worker_loop,
            args=(process_func,),
            daemon=True,
            name="VideoProcessingWorker"
        )
        worker_thread.start()
        print("[Worker] Worker thread started")
    else:
        print("[Worker] Worker thread already running")


def stop_worker():
    """Stop background worker thread"""
    global is_running
    is_running = False
    print("[Worker] Worker thread stopping...")


def queue_job(job: Job):
    """Add job to processing queue"""
    job_queue.put(job)
    print(f"[Worker] Job {job.job_id} queued (queue size: {job_queue.qsize()})")


def get_queue_size() -> int:
    """Get current queue size"""
    return job_queue.qsize()
