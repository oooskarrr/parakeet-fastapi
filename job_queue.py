
import uuid
import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobQueue:
    def __init__(self, storage_file: str = "jobs.json"):
        self.storage_file = storage_file
        self.jobs: Dict[str, Dict[str, Any]] = {}  # All jobs by ID
        self.queue: List[str] = []                 # List of job IDs in pending order
        self.batches: Dict[str, Dict[str, Any]] = {} # Batch metadata
        self.lock = threading.RLock()              # Thread safety
        self.load_jobs()

    def load_jobs(self):
        """Load jobs from disk persistence"""
        with self.lock:
            if os.path.exists(self.storage_file):
                try:
                    with open(self.storage_file, 'r') as f:
                        data = json.load(f)
                        self.jobs = data.get("jobs", {})
                        self.batches = data.get("batches", {})
                        self.queue = data.get("queue", [])
                        
                        # Reset any specific "processing" jobs to "pending" on restart 
                        # (unless we have resume logic, which comes later)
                        for job_id, job in self.jobs.items():
                            if job["status"] == JobStatus.PROCESSING:
                                job["status"] = JobStatus.PENDING
                                if job_id not in self.queue:
                                    self.queue.insert(0, job_id)
                        
                    logger.info(f"Loaded {len(self.jobs)} jobs from storage.")
                except Exception as e:
                    logger.error(f"Failed to load jobs: {e}")
                    self.jobs = {}
                    self.queue = []

    def save_jobs(self):
        """Save jobs to disk persistence"""
        with self.lock:
            try:
                data = {
                    "jobs": self.jobs,
                    "batches": self.batches,
                    "queue": self.queue
                }
                # Atomic write ideally, but simple overwrite for now
                with open(self.storage_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save jobs: {e}")

    def create_batch(self, file_list: List[Dict[str, str]]) -> str:
        """
        Create a new batch of jobs.
        Args:
            file_list: List of dicts with keys 'filename', 'temp_path', etc.
        Returns:
            batch_id
        """
        with self.lock:
            batch_id = str(uuid.uuid4())
            batch_jobs = []
            
            for file_info in file_list:
                job_id = str(uuid.uuid4())
                job = {
                    "job_id": job_id,
                    "batch_id": batch_id,
                    "filename": file_info["filename"],
                    "temp_path": file_info["temp_path"],
                    "duration": file_info.get("duration", 0), # Store pre-calculated duration
                    "status": JobStatus.PENDING,
                    "created_at": time.time(),
                    "progress": {
                        "percent": 0,
                        "current_chunk": 0,
                        "total_chunks": 0,
                        "status": "queued"
                    },
                    "result": None,
                    "error": None
                }
                self.jobs[job_id] = job
                self.queue.append(job_id)
                batch_jobs.append(job_id)
            
            self.batches[batch_id] = {
                "batch_id": batch_id,
                "created_at": time.time(),
                "job_ids": batch_jobs,
                "status": "active"
            }
            
            self.save_jobs()
            return batch_id

    def get_next_job(self) -> Optional[str]:
        """Get the next pending job ID"""
        with self.lock:
            if not self.queue:
                return None
            return self.queue[0]  # Peek, don't remove yet (remove when done or processing starts?)
            # Strategy: Keep in queue until claimed? 
            # Better: Remove from queue, set status to processing.

    def claim_next_job(self) -> Optional[Dict[str, Any]]:
        """Atomically claim the next pending job for processing"""
        with self.lock:
            if not self.queue:
                return None
            
            job_id = self.queue.pop(0)
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = JobStatus.PROCESSING
                self.save_jobs()
                return self.jobs[job_id]
            return None

    def update_job_progress(self, job_id: str, progress_data: Dict[str, Any]):
        """
        Update progress for a specific job.
        NOTE: This does NOT persist to disk to avoid performance penalties during rapid updates.
        Persistence happens on status changes (claim, complete, fail).
        """
        with self.lock:
            if job_id in self.jobs:
                # Merge progress data
                self.jobs[job_id]["progress"].update(progress_data)
                # self.save_jobs()  <-- DEACTIVATED for performance

    def complete_job(self, job_id: str, result: Any):
        """Mark job as completed"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = JobStatus.COMPLETED
                self.jobs[job_id]["result"] = result
                self.jobs[job_id]["progress"]["percent"] = 100
                self.jobs[job_id]["completed_at"] = time.time()
                self.save_jobs()

    def fail_job(self, job_id: str, error_msg: str):
        """Mark job as failed"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = JobStatus.FAILED
                self.jobs[job_id]["error"] = error_msg
                self.save_jobs()

    def set_job_duration(self, job_id: str, duration: float):
        """Set the audio duration for a job (needed for batch ETA)"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["duration"] = duration
                # We don't save to disk yet to avoid I/O, will be saved on status change

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get full status of a batch"""
        with self.lock:
            if batch_id not in self.batches:
                return None
            
            batch = self.batches[batch_id]
            jobs_status = []
            for jid in batch["job_ids"]:
                if jid in self.jobs:
                    jobs_status.append(self.jobs[jid])
            
            return {
                "batch_id": batch_id,
                "created_at": batch["created_at"],
                "jobs": jobs_status
            }

    def delete_batch(self, batch_id: str):
        """Permanently delete a batch and its associated jobs from storage"""
        with self.lock:
            if batch_id in self.batches:
                logger.info(f"Cleaning up batch {batch_id} from storage...")
                job_ids = self.batches[batch_id].get("job_ids", [])
                for jid in job_ids:
                    if jid in self.jobs:
                        del self.jobs[jid]
                    if jid in self.queue:
                        self.queue.remove(jid)
                
                del self.batches[batch_id]
                self.save_jobs()
                logger.info(f"Batch {batch_id} and its {len(job_ids)} jobs deleted.")


