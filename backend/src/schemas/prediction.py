from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    education: str
    work_experience: int = Field(..., ge=0)
    occupation: str
    hours_per_week: int = Field(..., ge=10, le=80)
    # ... другие фичи

class PredictionOutput(BaseModel):
    prediction: float
    probability: float
    explanation: Dict[str, Any]
    recommendations: List[str]