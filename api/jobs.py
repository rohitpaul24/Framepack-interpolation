"""
Job management system for async video processing.
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from pydantic import BaseModel

# Job storage directory
JOBS_DIR = Path("./jobs")
JOBS_DIR.mkdir(exist_ok=True)


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobData(BaseModel):
    """Job data model"""
    job_id: str
    video_name: str
    video_path: str
    params: dict
    status: str = JobStatus.QUEUED
    progress: int = 0
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class Job:
    """Job management class"""
    
    def __init__(self, job_id: str, video_name: str, video_path: str, params: dict):
        self.job_id = job_id
        self.video_name = video_name
        self.video_path = video_path
        self.params = params
        self.status = JobStatus.QUEUED
        self.progress = 0
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
    
    def to_dict(self) -> dict:
        """Convert job to dictionary"""
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "params": self.params,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error
        }
    
    def save(self):
        """Save job to disk"""
        job_file = JOBS_DIR / f"{self.job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(job_id: str) -> Optional['Job']:
        """Load job from disk"""
        job_file = JOBS_DIR / f"{job_id}.json"
        if not job_file.exists():
            return None
        
        with open(job_file, 'r') as f:
            data = json.load(f)
        
        job = Job(
            job_id=data['job_id'],
            video_name=data['video_name'],
            video_path=data['video_path'],
            params=data['params']
        )
        job.status = data['status']
        job.progress = data['progress']
        job.created_at = data['created_at']
        job.started_at = data.get('started_at')
        job.completed_at = data.get('completed_at')
        job.result = data.get('result')
        job.error = data.get('error')
        
        return job
    
    @staticmethod
    def list_all(status: Optional[str] = None, limit: int = 100) -> List[dict]:
        """List all jobs, optionally filtered by status"""
        jobs = []
        
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                job = Job.load(job_file.stem)
                if job and (status is None or job.status == status):
                    jobs.append({
                        "job_id": job.job_id,
                        "video_name": job.video_name,
                        "status": job.status,
                        "progress": job.progress,
                        "created_at": job.created_at,
                        "started_at": job.started_at,
                        "completed_at": job.completed_at
                    })
            except Exception as e:
                print(f"Error loading job {job_file}: {e}")
                continue
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jobs[:limit]
    
    @staticmethod
    def delete(job_id: str) -> bool:
        """Delete a job"""
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
            return True
        return False
    
    @staticmethod
    def create_job(video_name: str, video_path: str, params: dict) -> 'Job':
        """Create a new job"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, video_name, video_path, params)
        job.save()
        return job
