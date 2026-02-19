"""
LLM-based skill updater that generates new skills from failed trajectories.
Uses Azure OpenAI o3 model for analysis.

Required environment variables:
    AZURE_OPENAI_API_KEY      – Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT     – Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION  – API version (default: 2025-01-01-preview)
"""
import json
import os
import re
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI


class SkillUpdater:
    def __init__(
        self,
        max_new_skills_per_update: int = 3,
        max_completion_tokens: int = 2048,
    ):
        # Read credentials from environment variables — never hardcode secrets.
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        if not api_key or not endpoint:
            raise EnvironmentError(
                "SkillUpdater requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "environment variables to be set."
            )

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = "o3"
        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
    ) -> List[Dict]:
        """
        Analyse failed trajectories and generate new skills to address the gaps.

        Args:
            failed_trajectories: List of dicts with keys:
                ``task``       – task description string
                ``trajectory`` – list of ``{action, observation}`` step dicts
                ``task_type``  – detected task category string
            current_skills: The current skill bank dict (with keys
                ``general_skills``, ``task_specific_skills``, etc.)

        Returns:
            List of new skill dicts ready to be passed to
            ``SkillsOnlyMemory.add_skills()``.
        """
        if not failed_trajectories:
            return []

        # Compute the next available dyn_ index BEFORE calling the LLM so we
        # can tell it which IDs to use, avoiding duplicate-ID collisions.
        next_dyn_idx = self._next_dyn_index(current_skills)

        prompt = self._build_analysis_prompt(
            failed_trajectories, current_skills, next_dyn_idx
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            raw_skills = self._parse_skills_response(response.choices[0].message.content)

            # Reassign dyn_ IDs on our side to guarantee no collisions,
            # regardless of what the LLM returned.
            reassigned = self._reassign_dyn_ids(raw_skills, next_dyn_idx)

            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_skills_generated': len(reassigned),
                'skill_ids': [s.get('skill_id') for s in reassigned],
            })

            return reassigned[:self.max_new_skills_per_update]

        except Exception as e:
            print(f"[SkillUpdater] Error calling o3: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_dyn_index(self, current_skills: Dict) -> int:
        """
        Scan the current skill bank for existing ``dyn_NNN`` IDs and return
        the next unused integer index (1-based).
        """
        max_idx = 0
        pattern = re.compile(r'^dyn_(\d+)$')

        for skill in current_skills.get('general_skills', []):
            m = pattern.match(skill.get('skill_id', ''))
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        for skills in current_skills.get('task_specific_skills', {}).values():
            for skill in skills:
                m = pattern.match(skill.get('skill_id', ''))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    def _reassign_dyn_ids(self, skills: List[Dict], start_idx: int) -> List[Dict]:
        """
        Replace whatever skill_id values the LLM returned with guaranteed-unique
        ``dyn_NNN`` IDs starting from ``start_idx``.
        """
        reassigned = []
        for i, skill in enumerate(skills):
            updated = dict(skill)
            updated['skill_id'] = f"dyn_{start_idx + i:03d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
    ) -> str:
        # Format failure examples
        failure_examples = []
        for i, traj in enumerate(failed_trajectories[:5]):
            failure_examples.append(
                f"\nExample {i + 1}:\n"
                f"Task: {traj['task']}\n"
                f"Task Type: {traj['task_type']}\n"
                f"Trajectory (last 5 steps):\n"
                f"{self._format_trajectory(traj['trajectory'][-5:])}\n"
            )

        # Collect all existing skill titles (for deduplication hint to the LLM)
        existing_titles = [s['title'] for s in current_skills.get('general_skills', [])]
        for task_type, skills in current_skills.get('task_specific_skills', {}).items():
            for s in skills:
                existing_titles.append(f"[{task_type}] {s.get('title', '')}")

        # Show the LLM what IDs to use (we'll reassign them anyway, but
        # providing the range avoids confusion in the returned JSON)
        example_ids = ", ".join(
            f'"dyn_{next_dyn_idx + j:03d}"'
            for j in range(self.max_new_skills_per_update)
        )

        return f"""Analyze these failed agent trajectories and suggest NEW skills to add to the skill bank.

FAILED TRAJECTORIES:
{''.join(failure_examples)}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-{self.max_new_skills_per_update} NEW actionable skills that would help avoid these failures.
Each skill must have: skill_id, title (3-5 words), principle (1-2 sentences), when_to_apply.

Use skill_ids: {example_ids}

Return ONLY a JSON array of skills, no other text.
Example format:
[{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location by examining the environment.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""

    def _format_trajectory(self, steps: List[Dict]) -> str:
        lines = []
        for step in steps:
            action = step.get('action', 'unknown')
            obs = step.get('observation', '')[:200]
            lines.append(f"  Action: {action}\n  Observation: {obs}")
        return '\n'.join(lines)

    def _parse_skills_response(self, response: str) -> List[Dict]:
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                skills = json.loads(response[json_start:json_end])
                return [
                    s for s in skills
                    if all(k in s for k in ['skill_id', 'title', 'principle'])
                ]
        except json.JSONDecodeError as e:
            print(f"[SkillUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_skills_generated': 0}
        return {
            'total_updates': len(self.update_history),
            'total_skills_generated': sum(h['num_skills_generated'] for h in self.update_history),
            'all_skill_ids': [sid for h in self.update_history for sid in h['skill_ids']],
        }
