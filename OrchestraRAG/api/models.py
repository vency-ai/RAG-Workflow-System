# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

# Pydantic Model for API Input
class QuestionList(BaseModel):
    """
    Represents the input payload for the /run-sequence API endpoint.
    Contains a list of questions and optional flags for cache usage and evaluation.
    """
    questions: List[str] = Field(..., description="A list of questions to process.")
    use_cache: Optional[bool] = Field(default=False, description="Whether to use the RAG service's internal cache for this session.")
    eval: Optional[bool] = Field(default=True, description="Whether to trigger evaluation for this session.")

# You might add more Pydantic models here for API responses if they become complex.
# For example, if you wanted a strictly typed response for /run-sequence:
# class StructuredAnswer(BaseModel):
#     question: str
#     answer: str
#     sources: List[Dict]
#     model_used: str
#     evaluation: Optional[Dict] = None

# class RunSequenceResponse(BaseModel):
#     answers: List[str]
#     structured_answers: List[StructuredAnswer]
#     summary: Optional[str]
#     node_timings: Dict[str, float]
#     pipeline_time: float
#     active_chain_sessions: Optional[int]
#     evaluation_results: Optional[Dict]
