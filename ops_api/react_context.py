from typing import Optional
from .react_memory import ReActMemory

class ReActContextBuilder:

    def __init__(self, memory: ReActMemory, max_context_iterations: int = 10):
        self.memory = memory
        self.max_context_iterations = max_context_iterations

    def build_context(self, current_iteration: int) -> str:
        context_parts = []

        context_parts.append(f"Task: {self.memory.task_description}")
        context_parts.append(f"Current iteration: {current_iteration}")

        # Task Plan section
        if self.memory.task_plan and self.memory.task_plan.subtasks:
            plan = self.memory.task_plan
            context_parts.append(f"\n## Task Plan ({plan.completed_count}/{plan.total_count} completed):")
            context_parts.append(f"  Overview: {plan.overview}")
            context_parts.append("  Sub-tasks:")
            for st in plan.subtasks:
                status_icon = {
                    "pending": "[ ]",
                    "in_progress": "[>]",
                    "completed": "[x]",
                    "skipped": "[-]"
                }.get(st.status, "[ ]")
                line = f"    {status_icon} {st.id}: {st.description} ({st.status})"
                if st.notes:
                    line += f" - {st.notes}"
                if st.completed_iteration:
                    line += f" (completed at iteration {st.completed_iteration})"
                context_parts.append(line)

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
        return context