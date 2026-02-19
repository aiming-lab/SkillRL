# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    raise ImportError(
        "sentence-transformers and faiss-cpu are required for retrieval memory. "
        "Install with: pip install sentence-transformers faiss-cpu"
    )

from .base import BaseMemory


# Global cache for embedding models to avoid loading multiple times
_EMBEDDING_MODEL_CACHE = {}


class RetrievalMemory(BaseMemory):
    """
    Retrieval-augmented memory system for ALFWorld agent training.

    Uses semantic similarity search to retrieve relevant past experiences (both successful
    and failed trajectories) based on current task description. Helps agents learn from
    previous episodes by providing:
    - General skills (Claude-style concise principles)
    - Task-specific skills for the detected task type
    - Common mistakes to avoid
    - (Optional) Similar trajectory examples for context
    """

    def __init__(
        self,
        memory_json_path: str,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        skills_json_path: str = None
    ):
        """
        Initialize retrieval memory system.

        Args:
            memory_json_path: Path to JSON file containing pre-generated memories
            embedding_model_name: HuggingFace model name for text embeddings
            device: Device to run embedding model ('cuda' or 'cpu')
            skills_json_path: Optional path to Claude-style skills JSON
        """
        import torch

        self.embedding_model_name = embedding_model_name
        self.requested_device = device
        self.device = device
        self.embedding_model = None  # Lazy load
        self.memory_json_path = memory_json_path  # Save for cache lookup

        # Load memories from JSON
        with open(memory_json_path, 'r') as f:
            self.memories = json.load(f)

        # Separate success and failure memories
        self.success_memories = [
            m for m in self.memories
            if m['tags']['outcome'] == 'Success'
        ]
        self.failure_memories = [
            m for m in self.memories
            if m['tags']['outcome'] in ['Failure', 'Failed']
        ]

        print(f"[RetrievalMemory] Loaded {len(self.success_memories)} success and {len(self.failure_memories)} failure memories")

        # Load Claude-style skills if available
        self.skills = None
        if skills_json_path is None:
            # Try to find skills file in same directory as memory JSON
            skills_path = os.path.join(
                os.path.dirname(memory_json_path),
                'claude_style_skills.json'
            )
            if os.path.exists(skills_path):
                skills_json_path = skills_path

        if skills_json_path and os.path.exists(skills_json_path):
            with open(skills_json_path, 'r') as f:
                self.skills = json.load(f)
            print(f"[RetrievalMemory] Loaded Claude-style skills: "
                  f"{len(self.skills.get('general_skills', []))} general, "
                  f"{sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())} task-specific, "
                  f"{len(self.skills.get('common_mistakes', []))} mistakes")
        else:
            print("[RetrievalMemory] No Claude-style skills found, using legacy trajectory-based retrieval")

        # Lazy initialization - build indices only when needed
        self.success_embeddings = None
        self.failure_embeddings = None
        self.success_index = None
        self.failure_index = None
        self._initialized = False

        print("[RetrievalMemory] Initialized (embedding model will be loaded on first retrieval)")

    def _ensure_initialized(self):
        """Lazy initialization of embedding model and indices."""
        if self._initialized:
            return

        import torch

        # Auto-detect device availability
        device = self.requested_device
        if device == "cuda" and not torch.cuda.is_available():
            print(f"[RetrievalMemory] CUDA requested but not available, falling back to CPU")
            device = "cpu"

        self.device = device

        # Use cached model if available (memory optimization)
        cache_key = f"{self.embedding_model_name}_{device}"
        if cache_key in _EMBEDDING_MODEL_CACHE:
            print(f"[RetrievalMemory] Reusing cached embedding model on {device}")
            self.embedding_model = _EMBEDDING_MODEL_CACHE[cache_key]
        else:
            try:
                print(f"[RetrievalMemory] Loading embedding model {self.embedding_model_name} on {device}...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                _EMBEDDING_MODEL_CACHE[cache_key] = self.embedding_model
                print(f"[RetrievalMemory] Embedding model loaded successfully and cached")
            except Exception as e:
                print(f"[RetrievalMemory] Failed to load model on {device}, trying CPU: {e}")
                device = "cpu"
                self.device = device
                cache_key = f"{self.embedding_model_name}_{device}"
                if cache_key in _EMBEDDING_MODEL_CACHE:
                    self.embedding_model = _EMBEDDING_MODEL_CACHE[cache_key]
                else:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                    _EMBEDDING_MODEL_CACHE[cache_key] = self.embedding_model

        # Try to load cached embeddings first
        import time
        import pickle
        cache_path = os.path.join(
            os.path.dirname(self.memory_json_path),
            'embeddings_cache.pkl'
        )

        if os.path.exists(cache_path):
            try:
                print(f"[RetrievalMemory] Loading cached embeddings from {cache_path}...")
                start = time.time()
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                # Verify cache is for the same model
                if cache_data['embedding_model_name'] == self.embedding_model_name:
                    self.success_embeddings = cache_data['success_embeddings']
                    self.failure_embeddings = cache_data['failure_embeddings']
                    print(f"[RetrievalMemory] ✓ Loaded cached embeddings in {time.time()-start:.2f}s")
                else:
                    print(f"[RetrievalMemory] Cache model mismatch, recomputing...")
                    raise ValueError("Model mismatch")
            except Exception as e:
                print(f"[RetrievalMemory] Failed to load cache: {e}, computing embeddings...")
                self.success_embeddings = None
                self.failure_embeddings = None

        # Compute embeddings if not cached
        if self.success_embeddings is None:
            start = time.time()
            print(f"[RetrievalMemory] Building embeddings for {len(self.success_memories)} success memories...")
            self.success_embeddings = self._build_embeddings(self.success_memories)
            print(f"[RetrievalMemory] Success embeddings done in {time.time()-start:.2f}s")

        if self.failure_embeddings is None:
            start = time.time()
            print(f"[RetrievalMemory] Building embeddings for {len(self.failure_memories)} failure memories...")
            self.failure_embeddings = self._build_embeddings(self.failure_memories)
            print(f"[RetrievalMemory] Failure embeddings done in {time.time()-start:.2f}s")

        # Build FAISS index for fast retrieval
        print("[RetrievalMemory] Building FAISS indices...")
        self.success_index = self._build_faiss_index(self.success_embeddings)
        self.failure_index = self._build_faiss_index(self.failure_embeddings)

        self._initialized = True
        print(f"[RetrievalMemory] ✓ Initialization complete on {self.device}")

    def _build_embeddings(self, memories: List[Dict]) -> np.ndarray:
        """Build embeddings for task descriptions in memories."""
        texts = [m['content']['task_meta']['original_goal'] for m in memories]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        index.add(embeddings)
        return index

    def _detect_task_type(self, task_description: str) -> str:
        """Detect ALFWorld task type from task description."""
        goal = task_description.lower()

        if 'look at' in goal and 'under' in goal:
            return 'look_at_obj_in_light'
        elif 'clean' in goal:
            return 'clean'
        elif 'heat' in goal:
            return 'heat'
        elif 'cool' in goal:
            return 'cool'
        elif 'examine' in goal or 'find' in goal:
            return 'examine'
        elif 'put' in goal:
            return 'pick_and_place'
        else:
            return 'pick_and_place'  # Default

    def retrieve(
        self,
        task_description: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        max_tokens: int = 2000,
        include_examples: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for a given task description.

        If Claude-style skills are available, returns concise skills.
        Otherwise, falls back to trajectory-based retrieval.

        Args:
            task_description: Current task description to retrieve memories for
            top_k: Maximum number of memories to retrieve (for trajectory examples)
            similarity_threshold: Minimum similarity score to include memory
            max_tokens: Maximum token budget for all retrieved memories
            include_examples: Whether to include trajectory examples (increases context)

        Returns:
            Dictionary containing:
                - general_skills: List of general skills/principles
                - task_specific_skills: List of task-type specific skills
                - mistakes_to_avoid: List of common mistakes
                - task_specific_examples: (optional) List of similar trajectories
        """
        # Use Claude-style skills if available
        if self.skills:
            return self._retrieve_with_skills(task_description, max_tokens, include_examples)

        # Fallback to legacy trajectory-based retrieval
        return self._retrieve_legacy(task_description, top_k, similarity_threshold, max_tokens)

    def _retrieve_with_skills(
        self,
        task_description: str,
        max_tokens: int,
        include_examples: bool
    ) -> Dict[str, Any]:
        """Retrieve using Claude-style skills (concise, actionable)."""
        task_type = self._detect_task_type(task_description)

        # Get general skills (most important ones)
        general_skills = self.skills.get('general_skills', [])[:6]

        # Get task-specific skills
        task_skills = self.skills.get('task_specific_skills', {}).get(task_type, [])

        # Get common mistakes
        common_mistakes = self.skills.get('common_mistakes', [])[:5]

        result = {
            'general_skills': general_skills,
            'task_specific_skills': task_skills,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'task_specific_examples': []
        }

        # Optionally include similar trajectory examples
        if include_examples:
            self._ensure_initialized()
            query_embedding = self.embedding_model.encode(
                [task_description],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            scores, indices = self.success_index.search(query_embedding, 3)
            examples = [
                (self.success_memories[idx], float(score))
                for idx, score in zip(indices[0], scores[0])
                if score >= 0.6
            ]
            result['task_specific_examples'] = examples[:2]  # Limit to 2 examples

        return result

    def _retrieve_legacy(
        self,
        task_description: str,
        top_k: int,
        similarity_threshold: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Legacy trajectory-based retrieval (fallback)."""
        # Ensure embedding model and indices are loaded
        self._ensure_initialized()

        # Embed query
        query_embedding = self.embedding_model.encode(
            [task_description],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Retrieve from success memories
        scores, indices = self.success_index.search(query_embedding, top_k)
        success_results = [
            (self.success_memories[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if score >= similarity_threshold
        ]

        # Retrieve from failure memories (for mistakes to avoid)
        fail_k = min(top_k // 2, 5)
        fail_scores, fail_indices = self.failure_index.search(query_embedding, fail_k)
        failure_results = [
            (self.failure_memories[idx], float(score))
            for idx, score in zip(fail_indices[0], fail_scores[0])
            if score >= similarity_threshold
        ]

        # Categorize and budget
        return self._categorize_and_budget(success_results, failure_results, max_tokens)

    def _categorize_and_budget(
        self,
        success_results: List[Tuple[Dict, float]],
        failure_results: List[Tuple[Dict, float]],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Categorize memories into general skills and task-specific examples,
        applying token budget constraints.
        """
        # Extract general planning patterns (unique patterns across memories)
        general_skills = self._extract_general_patterns(success_results)

        # Task-specific examples are already sorted by similarity
        task_specific = success_results

        # Budget allocation
        general_tokens = 500
        mistakes_tokens = 300
        examples_tokens = max_tokens - general_tokens - mistakes_tokens

        # Truncate to fit budget
        selected_general = self._truncate_by_tokens(general_skills, general_tokens)
        selected_examples = self._truncate_by_tokens(task_specific, examples_tokens)
        selected_mistakes = self._truncate_by_tokens(failure_results, mistakes_tokens)

        return {
            'general_skills': selected_general,
            'task_specific_examples': selected_examples,
            'mistakes_to_avoid': selected_mistakes
        }

    def _extract_general_patterns(
        self,
        success_results: List[Tuple[Dict, float]]
    ) -> List[Dict[str, Any]]:
        """
        Extract unique general planning patterns from successful memories.
        Returns list of (pattern, example_task, score) tuples.
        """
        patterns = {}

        for memory, score in success_results:
            content = memory['content']
            guidelines = content.get('strategic_guidelines', {})
            pattern = guidelines.get('planning_pattern')

            if pattern and pattern not in patterns:
                patterns[pattern] = {
                    'pattern': pattern,
                    'example_task': content['task_meta']['original_goal'],
                    'score': score
                }

        # Return list of pattern dicts sorted by score
        return sorted(patterns.values(), key=lambda x: x['score'], reverse=True)

    def _truncate_by_tokens(
        self,
        items: List,
        max_tokens: int
    ) -> List:
        """
        Truncate list of items to fit within token budget.
        Rough estimation: 1 token ≈ 4 characters
        """
        result = []
        current_tokens = 0

        for item in items:
            # Estimate tokens for this item
            if isinstance(item, dict):
                item_text = json.dumps(item)
            elif isinstance(item, tuple):
                item_text = json.dumps(item[0])  # Memory dict is first element
            else:
                item_text = str(item)

            estimated_tokens = len(item_text) // 4

            if current_tokens + estimated_tokens <= max_tokens:
                result.append(item)
                current_tokens += estimated_tokens
            else:
                break

        return result

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved memories into a string suitable for prompt injection.

        Args:
            retrieved_memories: Dict from retrieve() containing categorized memories

        Returns:
            Formatted string to insert into agent prompt
        """
        sections = []

        # Check if using Claude-style skills (has 'task_specific_skills' key)
        if 'task_specific_skills' in retrieved_memories:
            return self._format_claude_style(retrieved_memories)
        else:
            return self._format_legacy_style(retrieved_memories)

    def _format_claude_style(self, retrieved_memories: Dict[str, Any]) -> str:
        """Format Claude-style skills (concise, actionable)."""
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')

        # Format general skills
        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        # Format task-specific skills
        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            task_name = task_type.replace('_', ' ').title()
            lines = [f"### {task_name} Skills"]
            for skill in task_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        # Format common mistakes
        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes[:5]:  # Limit to 5
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        # Format trajectory examples (if any)
        examples = retrieved_memories.get('task_specific_examples', [])
        if examples:
            lines = ["### Reference Examples"]
            for i, (memory, score) in enumerate(examples, 1):
                content = memory['content']
                task = content['task_meta']['original_goal']
                traj = content.get('refined_trajectory') or {}
                trajectory = traj.get('refined_trajectory', []) if traj else []

                lines.append(f"{i}. \"{task}\"")
                if trajectory:
                    steps = [s.get('action', '') for s in trajectory[:4]]
                    lines.append(f"   Steps: {' → '.join(steps)}")
            sections.append("\n".join(lines))

        if sections:
            return "\n\n".join(sections)
        else:
            return "No relevant skills found for this task."

    def _format_legacy_style(self, retrieved_memories: Dict[str, Any]) -> str:
        """Format legacy trajectory-based memories."""
        legacy_sections = []

        # Format general planning patterns
        if retrieved_memories.get('general_skills'):
            general_lines = ["### General Planning Patterns:"]
            for i, skill in enumerate(retrieved_memories['general_skills'], 1):
                general_lines.append(
                    f"{i}. [Pattern]: {skill['pattern']}\n"
                    f"   - Example: {skill['example_task']}"
                )
            legacy_sections.append("\n".join(general_lines))

        # Format task-specific successful examples
        if retrieved_memories.get('task_specific_examples'):
            example_lines = ["### Similar Successful Examples:"]
            for i, (memory, score) in enumerate(retrieved_memories['task_specific_examples'], 1):
                content = memory['content']
                task = content['task_meta']['original_goal']
                traj = content.get('refined_trajectory') or {}
                trajectory = traj.get('refined_trajectory', []) if traj else []

                example_lines.append(f"{i}. Task: \"{task}\" (similarity: {score:.2f})")
                example_lines.append("   Steps:")

                # Show first few steps of trajectory
                for step in trajectory[:5]:  # Limit to 5 steps to save tokens
                    action = step.get('action', '')
                    obs = step.get('critical_observation', '')
                    example_lines.append(f"   - {action}")
                    if obs:
                        example_lines.append(f"     Observation: {obs}")

                if len(trajectory) > 5:
                    example_lines.append(f"   ... ({len(trajectory) - 5} more steps)")

                example_lines.append("")  # Empty line between examples

            legacy_sections.append("\n".join(example_lines))

        # Format mistakes to avoid from failures
        if retrieved_memories.get('mistakes_to_avoid'):
            mistake_lines = ["### Common Mistakes to Avoid:"]
            for memory, score in retrieved_memories['mistakes_to_avoid']:
                content = memory['content']
                guidelines = content.get('strategic_guidelines') or {}
                mistakes = guidelines.get('mistakes_to_avoid', []) if guidelines else []

                for mistake in mistakes:
                    if isinstance(mistake, dict):
                        trigger = mistake.get('trigger_condition', '')
                        bad_action = mistake.get('bad_action', '')
                        if trigger and bad_action:
                            mistake_lines.append(
                                f"- When: {trigger}\n"
                                f"  Don't: {bad_action}"
                            )
                    elif isinstance(mistake, str):
                        mistake_lines.append(f"- {mistake}")

            legacy_sections.append("\n".join(mistake_lines))

        # Combine all sections
        if legacy_sections:
            return "\n\n".join(legacy_sections)
        else:
            return "No relevant past experience found for this task."

    def save_trajectory(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        success: bool,
        save_path: str
    ):
        """
        Save a completed trajectory to the memory pool.

        Args:
            task_description: Task description for this trajectory
            trajectory: List of step dicts with keys like {step_index, action, observation}
            success: Whether the trajectory was successful
            save_path: Path to JSON file to save to
        """
        memory_id = f"mem_alfworld_{uuid.uuid4().hex[:8]}"

        if success:
            # Extract planning pattern from trajectory
            planning_pattern = self._extract_planning_pattern(trajectory)

            memory_obj = {
                "memory_id": memory_id,
                "contextual_description": f"AlfWorld task: {task_description}. Solved successfully.",
                "tags": {
                    "environment": "Alfworld",
                    "outcome": "Success",
                    "training_generated": True,
                    "timestamp": datetime.now().isoformat()
                },
                "content": {
                    "task_meta": {
                        "original_goal": task_description
                    },
                    "refined_trajectory": {
                        "refined_trajectory": trajectory
                    },
                    "strategic_guidelines": {
                        "planning_pattern": planning_pattern,
                        "mistakes_to_avoid": []
                    }
                },
                "origin_env_id": "training"
            }
        else:
            # For failed trajectories, save mistakes to avoid
            mistakes = self._extract_mistakes(trajectory, task_description)

            memory_obj = {
                "memory_id": memory_id,
                "contextual_description": f"AlfWorld task: {task_description}. Failed.",
                "tags": {
                    "environment": "Alfworld",
                    "outcome": "Failure",
                    "training_generated": True,
                    "timestamp": datetime.now().isoformat()
                },
                "content": {
                    "task_meta": {
                        "original_goal": task_description
                    },
                    "refined_trajectory": None,
                    "strategic_guidelines": {
                        "planning_pattern": None,
                        "mistakes_to_avoid": mistakes
                    }
                },
                "origin_env_id": "training"
            }

        # Append to JSON file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Load existing memories if file exists
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        else:
            existing = []

        existing.append(memory_obj)

        with open(save_path, 'w') as f:
            json.dump(existing, f, indent=2)

    def _extract_planning_pattern(self, trajectory: List[Dict]) -> str:
        """
        Extract a high-level planning pattern from a successful trajectory.
        Returns a string like "Search [Location] → Acquire [Object] → Navigate → Place"
        """
        # Simple heuristic: extract action types in sequence
        action_types = []
        for step in trajectory:
            action = step.get('action', '').lower()

            # Categorize action types
            if 'go to' in action or 'navigate' in action:
                if 'Navigate' not in action_types:
                    action_types.append('Navigate')
            elif 'take' in action or 'pick' in action or 'grab' in action:
                if 'Acquire' not in action_types:
                    action_types.append('Acquire')
            elif 'put' in action or 'place' in action or 'move' in action:
                if 'Place' not in action_types:
                    action_types.append('Place')
            elif 'open' in action:
                if 'Open' not in action_types:
                    action_types.append('Open')
            elif 'use' in action or 'turn on' in action:
                if 'Use' not in action_types:
                    action_types.append('Use')
            elif 'search' in action or 'look' in action or 'examine' in action:
                if 'Search' not in action_types:
                    action_types.append('Search')

        # Create pattern string
        if action_types:
            pattern = " → ".join(action_types)
            return pattern
        else:
            return "Unknown pattern"

    def _extract_mistakes(
        self,
        trajectory: List[Dict],
        task_description: str
    ) -> List[str]:
        """
        Extract common mistakes from a failed trajectory.
        Returns a list of mistake descriptions.
        """
        mistakes = []

        # Heuristic: detect common failure patterns
        actions = [step.get('action', '') for step in trajectory]

        # Check for repetitive actions
        if len(actions) != len(set(actions)):
            mistakes.append(
                "Repeated the same action multiple times without making progress"
            )

        # Check for wandering (many 'go to' actions)
        go_to_count = sum(1 for a in actions if 'go to' in a.lower())
        if go_to_count > len(actions) * 0.5:
            mistakes.append(
                "Spent too much time wandering between locations without taking productive actions"
            )

        # Check if task requires multiple objects but agent gave up early
        if 'two' in task_description.lower() and len(trajectory) < 5:
            mistakes.append(
                f"Task requires finding two objects but stopped searching too early after {len(trajectory)} steps"
            )

        # Generic failure message if no specific mistakes detected
        if not mistakes:
            mistakes.append(
                f"Failed to complete task: {task_description}. Consider different approach."
            )

        return mistakes

    # Required BaseMemory interface methods (not used in this implementation)
    def reset(self, batch_size: int):
        """Not used in retrieval memory - retrieval is stateless."""
        pass

    def store(self, record: Dict[str, List[Any]]):
        """Not used in retrieval memory - use save_trajectory() instead."""
        pass

    def fetch(self, step: int):
        """Not used in retrieval memory - use retrieve() instead."""
        pass

    def __len__(self):
        """Return total number of memories."""
        return len(self.memories)

    def __getitem__(self, idx: int):
        """Access memory by index."""
        return self.memories[idx]
