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

"""
Lightweight skills-only memory system.

This is a simplified version of RetrievalMemory that only uses Claude-style skills
without the overhead of loading and indexing trajectory memories.

Supports two retrieval modes:
  - "template": keyword-based task type detection + return all task-specific skills
    (original behaviour, zero latency, no GPU needed)
  - "embedding": encode the task description with Qwen3-Embedding-0.6B and rank
    both general and task-specific skills by cosine similarity, so only the
    top-k most relevant ones are injected into the prompt
"""

import json
import os
from typing import Dict, Any, List, Optional
from .base import BaseMemory


class SkillsOnlyMemory(BaseMemory):
    """
    Lightweight memory system that only uses Claude-style skills.

    Retrieval mode is controlled by the ``retrieval_mode`` constructor argument:

    * ``"template"`` (default) – keyword matching selects the task category;
      *all* task-specific skills for that category are returned, and the first
      ``top_k`` general skills are returned in document order.  No embedding
      model is needed.

    * ``"embedding"`` – the task description is encoded with a
      SentenceTransformer model (Qwen3-Embedding-0.6B by default).  Both
      general skills and task-specific skills (searched across **all**
      categories) are ranked by cosine similarity and the top-k are returned.
      Skill embeddings are pre-computed once and cached in memory.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
    ):
        """
        Args:
            skills_json_path:     Path to Claude-style skills JSON file.
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            embedding_model_path: Local path (or HF model ID) for the
                                  SentenceTransformer embedding model.  Only
                                  used when ``retrieval_mode="embedding"``.
                                  Defaults to ``"Qwen/Qwen3-Embedding-0.6B"``.
            task_specific_top_k:  Maximum number of task-specific skills to
                                  return.  ``None`` means *return all* in
                                  template mode and use ``top_k`` (general
                                  skills count) in embedding mode.
        """
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        with open(skills_json_path, 'r') as f:
            self.skills = json.load(f)

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k

        # Lazy-initialised embedding state (only used in embedding mode)
        self._embedding_model = None
        self._skill_embeddings_cache: Optional[Dict] = None

        n_general = len(self.skills.get('general_skills', []))
        n_task = sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())
        n_mistakes = len(self.skills.get('common_mistakes', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes  "
            f"| retrieval_mode={retrieval_mode}"
        )

        # In embedding mode, pre-compute skill embeddings eagerly so the first
        # retrieve() call is not slower than subsequent ones.
        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Task-type detection (template mode)                                  #
    # ------------------------------------------------------------------ #

    def _detect_task_type(self, task_description: str) -> str:
        """
        Infer the task category from ``task_description`` using keyword rules.

        Auto-detects whether the loaded skills belong to ALFWorld or WebShop
        by inspecting the task-specific skill keys.
        """
        task_specific = self.skills.get('task_specific_skills', {})
        goal = task_description.lower()

        # ---- ALFWorld categories ----------------------------------------
        if 'pick_and_place' in task_specific or 'clean' in task_specific:
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
            else:
                return 'pick_and_place'

        # ---- WebShop categories -----------------------------------------
        elif 'apparel' in task_specific or 'electronics' in task_specific:
            if any(kw in goal for kw in [
                'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
                'blouse', 'clothing', 'clothes', 't-shirt',
            ]):
                return 'apparel'
            elif any(kw in goal for kw in [
                'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper',
                'footwear',
            ]):
                return 'footwear'
            elif any(kw in goal for kw in [
                'laptop', 'phone', 'computer', 'tablet', 'charger',
                'cable', 'headphone', 'speaker', 'camera', 'electronic',
            ]):
                return 'electronics'
            elif any(kw in goal for kw in [
                'necklace', 'ring', 'bracelet', 'earring', 'watch',
                'jewelry', 'bag', 'purse', 'wallet',
            ]):
                return 'accessories'
            elif any(kw in goal for kw in [
                'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
                'decor', 'candle', 'vase', 'rug',
            ]):
                return 'home_decor'
            elif any(kw in goal for kw in [
                'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
                'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
            ]):
                return 'beauty_health'
            else:
                return 'other'

        # ---- Fallback: first key in task_specific_skills, or 'unknown' --
        else:
            return next(iter(task_specific), 'unknown')

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model (thread-safe for single process)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                )
            print(f"[SkillsOnlyMemory] Loading embedding model: {self.embedding_model_path}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
            print("[SkillsOnlyMemory] Embedding model ready.")
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        """Concatenate the skill fields most useful for semantic matching."""
        parts = []
        for field in ('title', 'principle', 'when_to_apply'):
            val = skill.get(field, '').strip()
            if val:
                parts.append(val)
        return ". ".join(parts)

    def _compute_skill_embeddings(self) -> Dict:
        """
        Pre-compute and cache normalised embeddings for every skill.

        The cache holds:
          ``items``      – flat list of ``(kind, task_type, skill_dict)``
          ``embeddings`` – numpy array of shape ``(n_skills, dim)``
          ``n_general``  – how many of the first rows correspond to general skills
        """
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        import numpy as np

        general_items = [
            ('general', None, s)
            for s in self.skills.get('general_skills', [])
        ]
        task_items = [
            ('task_specific', task_type, s)
            for task_type, skills in self.skills.get('task_specific_skills', {}).items()
            for s in skills
        ]
        all_items = general_items + task_items
        texts = [self._skill_to_text(item[2]) for item in all_items]

        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        self._skill_embeddings_cache = {
            'items': all_items,
            'embeddings': embeddings,
            'n_general': len(general_items),
        }
        print(
            f"[SkillsOnlyMemory] Cached embeddings for {len(all_items)} skills "
            f"({len(general_items)} general + {len(task_items)} task-specific)"
        )
        return self._skill_embeddings_cache

    def _embedding_retrieve(
        self,
        task_description: str,
        top_k_general: int,
        top_k_task_specific: int,
    ):
        """
        Retrieve the most relevant general and task-specific skills using
        cosine similarity between the task description and all cached skill
        embeddings.

        Args:
            task_description:   Free-form task goal string.
            top_k_general:      Number of general skills to return.
            top_k_task_specific: Number of task-specific skills to return
                                 (searched across **all** categories).

        Returns:
            Tuple of (general_skills, task_specific_skills).
        """
        import numpy as np

        cache = self._compute_skill_embeddings()
        model = self._get_embedding_model()

        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]  # shape: (dim,)

        sims = cache['embeddings'] @ query_emb  # cosine similarity, shape: (n,)

        n_general = cache['n_general']
        general_sims = sims[:n_general]
        task_sims = sims[n_general:]

        # Top-k general skills
        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache['items'][int(i)][2] for i in general_idx]

        # Top-k task-specific skills (cross-category search)
        task_idx = np.argsort(task_sims)[::-1][:top_k_task_specific]
        task_skills = [cache['items'][n_general + int(i)][2] for i in task_idx]

        return general_skills, task_skills

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve skills for a given task description.

        Args:
            task_description: Current task goal string.
            top_k:            Number of *general* skills to include.
                              In embedding mode this also serves as the
                              default for task-specific skills when
                              ``task_specific_top_k`` is not set.

        Returns:
            Dictionary with keys:
              - ``general_skills``       – list of skill dicts
              - ``task_specific_skills`` – list of skill dicts
              - ``mistakes_to_avoid``    – list of common-mistake dicts
              - ``task_type``            – detected task type string
              - ``task_specific_examples`` – always ``[]`` (reserved)
              - ``retrieval_mode``       – which mode was used
        """
        common_mistakes = self.skills.get('common_mistakes', [])[:5]

        # ----------------------------------------------------------------
        # Embedding mode: semantic ranking of all skills
        # ----------------------------------------------------------------
        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general_skills, task_skills = self._embedding_retrieve(
                task_description=task_description,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
            )
            # Still detect task type for bookkeeping / formatting labels
            task_type = self._detect_task_type(task_description)
            return {
                'general_skills': general_skills,
                'task_specific_skills': task_skills,
                'mistakes_to_avoid': common_mistakes,
                'task_type': task_type,
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
            }

        # ----------------------------------------------------------------
        # Template mode: keyword detection + return (sub)set of category skills
        # ----------------------------------------------------------------
        task_type = self._detect_task_type(task_description)
        general_skills = self.skills.get('general_skills', [])[:top_k]
        all_task_skills = self.skills.get('task_specific_skills', {}).get(task_type, [])

        if self.task_specific_top_k is not None:
            task_skills = all_task_skills[:self.task_specific_top_k]
        else:
            task_skills = all_task_skills  # original behaviour: return all

        return {
            'general_skills': general_skills,
            'task_specific_skills': task_skills,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'task_specific_examples': [],
            'retrieval_mode': 'template',
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills into a string suitable for prompt injection.

        Args:
            retrieved_memories: Dict returned by :meth:`retrieve`.

        Returns:
            Formatted multi-section string to insert into the agent prompt.
        """
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')
        mode = retrieved_memories.get('retrieval_mode', 'template')

        # General skills
        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        # Task-specific skills
        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace('_', ' ').title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        # Common mistakes
        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # BaseMemory interface (not used in skills-only memory)               #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return (
            len(self.skills.get('general_skills', [])) +
            sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()) +
            len(self.skills.get('common_mistakes', []))
        )

    def __getitem__(self, idx: int):
        return self.skills

    # ------------------------------------------------------------------ #
    # Dynamic update methods                                               #
    # ------------------------------------------------------------------ #

    def add_skills(self, new_skills: List[Dict], category: str = 'general') -> int:
        """
        Add new skills to the bank and invalidate the embedding cache.

        Args:
            new_skills: List of skill dicts to add.
            category:   ``'general'`` or a task-type key (e.g. ``'clean'``).

        Returns:
            Number of skills actually added (duplicates are skipped).
        """
        added = 0
        existing_ids = self._get_all_skill_ids()

        for skill in new_skills:
            skill_id = skill.get('skill_id')
            if skill_id in existing_ids:
                print(f"[SkillsOnlyMemory] Skipping duplicate skill: {skill_id}")
                continue

            if category == 'general':
                self.skills.setdefault('general_skills', []).append(skill)
            else:
                self.skills.setdefault('task_specific_skills', {}).setdefault(category, []).append(skill)
            added += 1
            print(f"[SkillsOnlyMemory] Added skill: {skill_id} - {skill.get('title', 'N/A')}")

        if added > 0:
            # Invalidate embedding cache so it is recomputed on next retrieve
            self._skill_embeddings_cache = None

        return added

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID and invalidate the embedding cache."""
        removed = False

        original_len = len(self.skills.get('general_skills', []))
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s.get('skill_id') != skill_id
        ]
        if len(self.skills.get('general_skills', [])) < original_len:
            removed = True

        for task_type in self.skills.get('task_specific_skills', {}):
            original_len = len(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = [
                s for s in self.skills['task_specific_skills'][task_type]
                if s.get('skill_id') != skill_id
            ]
            if len(self.skills['task_specific_skills'][task_type]) < original_len:
                removed = True

        if removed:
            self._skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def save_skills(self, path: str):
        """Persist the current skill bank to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id'):
                    ids.add(s['skill_id'])
        return ids

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'general': len(self.skills.get('general_skills', [])),
            'task_specific': sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()),
            'common_mistakes': len(self.skills.get('common_mistakes', [])),
            'total': len(self),
        }
