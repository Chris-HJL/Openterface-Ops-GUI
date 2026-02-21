from typing import Optional
from .react_memory import ReActMemory

class ReActContextBuilder:
    
    def __init__(self, memory: ReActMemory, max_context_iterations: int = 5):
        self.memory = memory
        self.max_context_iterations = max_context_iterations
    
    def build_context(self, current_iteration: int) -> str:
        context_parts = []
        
        context_parts.append(f"Task: {self.memory.task_description}")
        context_parts.append(f"Current iteration: {current_iteration}")
        
        if self.memory.iterations:
            context_parts.append("\n## Previous Iterations Summary:")
            
            recent_iterations = self.memory.iterations[-self.max_context_iterations:]
            
            for iteration in recent_iterations:
                context_parts.append(
                    f"\nIteration {iteration.iteration_number}:"
                    f"\n  Action: {iteration.parsed_action} on {iteration.parsed_element}"
                    f"\n  Reasoning: {iteration.reasoning}"
                    f"\n  Result: {'Success' if iteration.execution_success else 'Failed'}"
                )
                if iteration.execution_error:
                    context_parts.append(f"  Error: {iteration.execution_error}")
        
        if self.memory.successful_actions:
            context_parts.append("\n## Successful Actions:")
            for action, iterations in self.memory.successful_actions.items():
                context_parts.append(f"  - {action} (iterations: {iterations})")
        
        if self.memory.failed_actions:
            context_parts.append("\n## Failed Actions (to avoid):")
            for action, iterations in self.memory.failed_actions.items():
                context_parts.append(f"  - {action} (iterations: {iterations})")
        
        if self.memory.key_findings:
            context_parts.append("\n## Key Findings:")
            for finding in self.memory.key_findings:
                context_parts.append(f"  - {finding}")
        
        return "\n".join(context_parts)
    
    def build_enhanced_prompt(self, base_prompt: str, current_iteration: int) -> str:
        context = self.build_context(current_iteration)
        
        enhanced_prompt = f"""
{context}

## Current Task:
{base_prompt}

Please analyze the current screen and determine the next action based on:
1. The task description
2. Previous iterations and their outcomes
3. Successful and failed actions patterns
4. Current screen state

CRITICAL RULES - MUST FOLLOW:

## Status Determination:
You must determine if the previous action was successful or if the task is completed based on BOTH:
- The 'Result' field in the previous iterations summary
- The current screen state

IMPORTANT: It happens that the previous action summary may show 'Success' but the current screen state may not reflect the expected change. In such cases, you must:
1. Acknowledge the discrepancy
2. Determine if the screen is still loading (look for loading indicators, spinners, partial content)
3. If loading, wait and do not take action
4. If not loading but state is incorrect, take corrective action to reach the target state

Respond with one of the following:
- <task_status>completed</task_status> if the task is done
- <task_status>in_progress</task_status> if the task is not yet completed or the screen is still loading

If the screen is still loading, NO action or additional information should be provided,
Else if not completed, also provide:
- the next action to perform
Else if completed, also provide:
- <final_reasoning>brief explanation of the task completion</final_reasoning>
"""
        return enhanced_prompt
