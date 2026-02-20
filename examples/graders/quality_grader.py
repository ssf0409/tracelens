"""Example LLMGrader template for subjective quality evaluation.

This is a template showing how to build an LLM-as-judge grader.
You must implement _call_llm() with your preferred LLM client
(OpenAI, Anthropic, LiteLLM, etc.).
"""

import json

from eval_kit import LLMGrader
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript


class QualityGrader(LLMGrader):
    """Evaluates output quality using an LLM as judge.

    Rubric dimensions:
        - completeness: Does the output address all parts of the task?
        - clarity: Is the output clear and well-organized?
        - accuracy: Is the output factually correct?

    Each dimension is scored 1-10. The overall score is the average
    normalized to 0-1. Passing threshold is 0.7 (7/10 average).

    To use this grader, subclass it and implement _call_llm():

        class MyQualityGrader(QualityGrader):
            async def _call_llm(self, prompt: str) -> str:
                response = await openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content
    """

    def __init__(self, model: str = "gpt-4") -> None:
        super().__init__(grader_id="quality", model=model)

    def build_grading_prompt(
        self,
        transcript: Transcript,
        task: Task,
    ) -> str:
        return f"""You are an expert evaluator. Score the following output on three dimensions.

## Task
Name: {task.name}
Description: {task.description or 'N/A'}
Input: {json.dumps(task.input_data)}

## Agent Output
{json.dumps(transcript.final_output, indent=2)}

## Rubric
Score each dimension from 1 to 10:
- **completeness**: Does the output address all parts of the task?
- **clarity**: Is the output clear and well-organized?
- **accuracy**: Is the output factually correct?

## Response Format
Return ONLY valid JSON:
{{
    "completeness": <1-10>,
    "clarity": <1-10>,
    "accuracy": <1-10>,
    "feedback": "<brief explanation>"
}}"""

    def parse_llm_response(
        self,
        response: str,
        task: Task,
    ) -> tuple[bool, float, dict[str, float], str]:
        data = json.loads(response)

        metrics = {
            "completeness": float(data["completeness"]) / 10.0,
            "clarity": float(data["clarity"]) / 10.0,
            "accuracy": float(data["accuracy"]) / 10.0,
        }

        score = sum(metrics.values()) / len(metrics)
        passed = score >= 0.7
        feedback = data.get("feedback", "")

        return passed, score, metrics, feedback
