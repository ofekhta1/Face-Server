from datetime import datetime
from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator,ConfigDict
from Shared.models.job_type import JobType
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from .job_status import JobStatus
import json

class JobMetadata(BaseModel):
    """
    Metadata model for tracking job processing details.
    
    Provides robust serialization and deserialization with comprehensive validation.
    """
    status: JobStatus
    type: Optional[JobType] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    valid_images: Optional[List[str]] = Field(default_factory=list)
    invalid_images: Optional[List[str]] = Field(default_factory=list)
    return_detector: DetectorName = DetectorName.retinaface_buffalo
    return_embedder: EmbedderName = EmbedderName.resnet100
    save_invalid: bool = False
    error_message: Optional[str] = None

    @field_validator('valid_images', 'invalid_images', mode='before')
    @classmethod
    def parse_image_list(cls, v):
        """
        Ensure image lists are always lists, handling string representations.
        
        :param v: Input value (potentially a string or list)
        :return: Parsed list of images
        """
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the original string as a single-item list
                return [v]
        return v or []

    @model_validator(mode='before')
    @classmethod
    def validate_and_normalize_input(cls, values):
        """
        Comprehensive input validation and normalization.
        
        :param values: Raw input dictionary
        :return: Normalized values
        """
        # Convert empty strings to None
        for field in ['error_message', 'type']:
            if values.get(field) == '':
                values[field] = None

        return values

    def to_dict(self) -> dict:
        """
        Convert JobMetadata to a dictionary representation.
        
        :return: Serialized dictionary of job metadata
        """
        return {
            "status": self.status.value,
            "type": self.type.value if self.type else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() 
                if self.completed_at 
                else datetime.max.isoformat()
            ),
            "valid_images": self.valid_images or [],
            "invalid_images": self.invalid_images or [],
            "error_message": self.error_message or None,
            "return_detector": self.return_detector,
            "return_embedder": self.return_embedder,
            "save_invalid": self.save_invalid
        }

    @classmethod
    def from_dict(cls, data: Union[dict, str]) -> 'JobMetadata':
        """
        Create a JobMetadata instance from a dictionary or JSON string.
        
        :param data: Input data for creating JobMetadata
        :return: JobMetadata instance
        """
        # Handle JSON string input
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {e}")

        # Use Pydantic's built-in parsing to handle validation
        return cls(**data)

    model_config = ConfigDict(
        # Allow population by alias (if needed in future)
        populate_by_name = True,
        
        # Validate default values
        validate_assignment = True,
        
        # Arbitrary types are allowed (for enum and custom types)
        arbitrary_types_allowed = True
    )