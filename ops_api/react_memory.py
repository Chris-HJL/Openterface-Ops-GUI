from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from datetime import datetime
import json
import os

@dataclass
class IterationRecord:
    iteration_number: int
    timestamp: str
    image_path: str
    prompt: str
    llm_response: str
    execution_success: bool
    task_status: str
    
    image_base64: Optional[str] = None
    context: Optional[str] = None
    parsed_action: Optional[str] = None
    parsed_element: Optional[str] = None
    parsed_input: Optional[str] = None
    parsed_key: Optional[str] = None
    reasoning: Optional[str] = None
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None
    screen_changed: bool = False
    new_elements_detected: List[str] = field(default_factory=list)
    helpful: bool = False
    confidence: float = 0.0

@dataclass
class ReActMemory:
    session_id: str
    task_description: str
    start_time: str
    end_time: Optional[str] = None
    
    iterations: List[IterationRecord] = field(default_factory=list)
    successful_actions: Dict[str, List[int]] = field(default_factory=dict)
    failed_actions: Dict[str, List[int]] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)
    strategy_adjustments: List[str] = field(default_factory=list)
    
    final_status: Optional[str] = None
    final_reasoning: Optional[str] = None

class ReActMemoryStore:
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.memories: Dict[str, ReActMemory] = {}
        self.storage_dir = storage_dir
    
    def create_memory(self, session_id: str, task: str) -> ReActMemory:
        memory = ReActMemory(
            session_id=session_id,
            task_description=task,
            start_time=datetime.now().isoformat()
        )
        self.memories[session_id] = memory
        return memory
    
    def add_iteration(self, session_id: str, record: IterationRecord):
        if session_id in self.memories:
            self.memories[session_id].iterations.append(record)
    
    def update_patterns(self, session_id: str, record: IterationRecord):
        memory = self.memories.get(session_id)
        if not memory:
            return
        
        action_key = f"{record.parsed_action}:{record.parsed_element}"
        
        if record.execution_success and record.helpful:
            if action_key not in memory.successful_actions:
                memory.successful_actions[action_key] = []
            memory.successful_actions[action_key].append(record.iteration_number)
        elif not record.execution_success:
            if action_key not in memory.failed_actions:
                memory.failed_actions[action_key] = []
            memory.failed_actions[action_key].append(record.iteration_number)
    
    def finalize_memory(self, session_id: str, final_status: str, final_reasoning: Optional[str] = None):
        memory = self.memories.get(session_id)
        if memory:
            memory.end_time = datetime.now().isoformat()
            memory.final_status = final_status
            memory.final_reasoning = final_reasoning
            
            if self.storage_dir:
                self._save_to_file(memory)
    
    def _save_to_file(self, memory: ReActMemory):
        os.makedirs(self.storage_dir, exist_ok=True)
        filename = f"{memory.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(memory), f, ensure_ascii=False, indent=2)
