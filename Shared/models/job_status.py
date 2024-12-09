from enum import Enum
class JobStatus(str,Enum):
    queued="Queued"
    processing="Processing"
    completed="Completed"
    failed="Failed"