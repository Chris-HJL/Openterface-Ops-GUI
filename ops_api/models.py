"""
Pydantic模型定义
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Request models
class CreateSessionRequest(BaseModel):
    language: Optional[str] = "en"
    api_url: Optional[str] = None
    model: Optional[str] = None
    ui_model_api_url: Optional[str] = None
    ui_model: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    image: Optional[str] = None  # Base64 encoded image
    model: Optional[str] = None
    rag_enabled: Optional[bool] = None
    get_image_from_server: Optional[bool] = False

class BuildIndexRequest(BaseModel):
    session_id: str
    docs_dir: Optional[str] = "./docs"
    index_dir: Optional[str] = "./index"

class SwitchLangRequest(BaseModel):
    session_id: str
    lang_code: str

class ToggleRequest(BaseModel):
    session_id: str
    mode: Optional[bool] = None

class ClearHistoryRequest(BaseModel):
    session_id: str

class GetImageRequest(BaseModel):
    session_id: str

class ReactRequest(BaseModel):
    session_id: str
    task: str
    max_iterations: Optional[int] = 20
    rag_enabled: Optional[bool] = None
    model: Optional[str] = None

class StopReactRequest(BaseModel):
    session_id: str

# Response models
class SessionResponse(BaseModel):
    session_id: str
    message: str
    success: bool

class GetImageResponse(BaseModel):
    image: str  # Base64 encoded image
    image_id: Optional[str] = None  # Unique identifier for the image
    message: str
    success: bool

class ChatResponse(BaseModel):
    response: str
    image: Optional[str] = None  # Base64 encoded processed image
    history: List[Dict[str, Any]]
    success: bool

class IndexResponse(BaseModel):
    message: str
    success: bool

class StatusResponse(BaseModel):
    api_status: str
    ui_model_status: str
    success: bool

class LangResponse(BaseModel):
    message: str
    current_lang: str
    success: bool

class ToggleResponse(BaseModel):
    message: str
    current_state: bool
    success: bool

class ReactResponse(BaseModel):
    message: str
    iterations_completed: int
    final_status: str
    success: bool
    images: Optional[List[str]] = None  # List of base64 images from each iteration
