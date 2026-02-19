"""
Generate Claude-style skills for Search agent using o3 API.

Claude-style skills have these characteristics:
1. Concise and actionable - focus on what to do, not verbose descriptions
2. General principles - transferable across similar tasks
3. Failure-aware - include common pitfalls and how to avoid them
4. Hierarchical - general skills + query-type-specific skills

The search agent operates across multiple QA datasets:
- nq (Natural Questions): direct factoid retrieval
- popqa: entity attribute lookup (occupation, birthplace, etc.)
- triviaqa: trivia-style factoid questions
- hotpotqa: multi-hop comparison/bridge questions
- 2wikimultihopqa: multi-hop reasoning across Wikipedia
- musique: multi-step compositional questions
- bamboogle: multi-hop questions requiring chained reasoning

Output format:
{
    "general_skills": [
        {
            "skill_id": "...",
            "title": "...",
            "principle": "...",
            "when_to_apply": "...",
            "example": "..."  # optional
        }
    ],
    "query_type_skills": {
        "direct_retrieval": [...],
        "multi_hop_reasoning": [...],
        "entity_attribute_lookup": [...],
        "comparison": [...]
    },
    "common_mistakes": [
        {
            "mistake_id": "...",
            "description": "...",
            "why_it_happens": "...",
            "how_to_avoid": "..."
        }
    ]
}
"""

import json
import os
from typing import List, Dict, Any
from openai import AzureOpenAI


class OpenAIClient:
    def __init__(self, max_new_tokens: int = 4096, model: str = "o3"):
        self.max_new_tokens = max_new_tokens
        self.model = model
        self.client = AzureOpenAI(
            api_key="",
            azure_endpoint="",
            api_version=""
        )

    def generate_response(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content


def load_memories(json_path: str) -> List[Dict]:
    """Load existing memory data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def classify_query_type(mem: Dict) -> str:
    """Classify a memory into a query type based on data source and question structure."""
    data_source = mem['tags'].get('data_source', '')
    goal = mem['content']['task_meta']['original_goal'].lower()

    # Multi-hop datasets
    if data_source in ('hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'):
        # Further classify multi-hop into subtypes
        if any(kw in goal for kw in ['both', 'are the', 'which of', 'same', 'common', 'more', 'less', 'older', 'younger', 'taller', 'shorter']):
            return 'comparison'
        else:
            return 'multi_hop_reasoning'

    # Entity attribute lookup (popqa-style)
    if data_source == 'popqa':
        return 'entity_attribute_lookup'

    # Direct retrieval (nq, triviaqa)
    return 'direct_retrieval'


def categorize_by_query_type(memories: List[Dict]) -> Dict[str, Dict[str, List]]:
    """Categorize memories by query type and outcome."""
    categorized = {
        'direct_retrieval': {'success': [], 'failure': []},
        'multi_hop_reasoning': {'success': [], 'failure': []},
        'entity_attribute_lookup': {'success': [], 'failure': []},
        'comparison': {'success': [], 'failure': []},
    }

    for mem in memories:
        query_type = classify_query_type(mem)
        outcome = 'success' if mem['tags']['outcome'] == 'Success' else 'failure'
        categorized[query_type][outcome].append(mem)

    return categorized


def extract_patterns(memories: List[Dict]) -> str:
    """Extract key patterns from memories for prompt."""
    patterns = []
    for mem in memories[:10]:  # Limit to 10 for context
        goal = mem['content']['task_meta']['original_goal']
        data_source = mem['tags'].get('data_source', '')
        trajectory = mem['content'].get('refined_trajectory') or []
        # Handle both list and dict trajectory formats
        if isinstance(trajectory, dict):
            trajectory = trajectory.get('refined_trajectory', [])
        strategic = mem['content'].get('strategic_guidelines') or {}
        # Handle nested strategic_guidelines
        if 'strategic_guidelines' in strategic:
            strategic = strategic['strategic_guidelines']
        mistakes = strategic.get('mistakes_to_avoid', []) if strategic else []
        planning = strategic.get('planning_pattern', '') if strategic else ''

        pattern = {
            'goal': goal,
            'data_source': data_source,
            'steps': [{'action': s.get('action', ''), 'reasoning': s.get('reasoning', '')}
                     for s in trajectory[:5]] if isinstance(trajectory, list) else [],
            'planning_pattern': planning,
            'mistakes': mistakes[:3] if mistakes else []
        }
        patterns.append(pattern)

    return json.dumps(patterns, indent=2)


def generate_general_skills(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
    """Generate general skills using o3."""

    # Collect sample data across all query types
    all_successes = []
    all_failures = []
    for query_type, data in categorized_memories.items():
        all_successes.extend(data['success'][:5])
        all_failures.extend(data['failure'][:5])

    success_patterns = extract_patterns(all_successes)
    failure_patterns = extract_patterns(all_failures)

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Analyze these successful and failed trajectories from a Search AI agent that answers questions by issuing search queries and reading retrieved documents.

The agent operates in a search environment where it can:
- Issue search queries to retrieve relevant documents
- Read and analyze retrieved documents
- Formulate answers based on evidence found

The agent handles various question types across datasets:
- Direct factoid retrieval (Natural Questions, TriviaQA)
- Entity attribute lookup (PopQA)
- Multi-hop reasoning (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle)

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 8-12 GENERAL SKILLS that apply across ALL question types. These should be:
1. **Concise** - Each skill should be 1-2 sentences max
2. **Actionable** - Clear what to do, not vague principles
3. **Transferable** - Apply to direct retrieval, multi-hop, comparison, and entity lookup tasks
4. **Failure-aware** - Derived from what went wrong in failures

Format as JSON array:
[
    {{
        "skill_id": "gen_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight in 1-2 sentences",
        "when_to_apply": "Specific trigger condition"
    }}
]

Focus on:
- Query formulation strategies (how to construct effective search queries)
- Evidence extraction and verification
- Multi-step decomposition for complex questions
- Handling ambiguous entities or questions
- Knowing when to refine vs. when to answer
- Avoiding hallucination (answering without evidence)

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_query_type_skills(client: OpenAIClient, query_type: str,
                                successes: List[Dict], failures: List[Dict]) -> List[Dict]:
    """Generate query-type-specific skills."""

    if not successes and not failures:
        return []

    success_patterns = extract_patterns(successes[:8])
    failure_patterns = extract_patterns(failures[:8]) if failures else "[]"

    type_descriptions = {
        'direct_retrieval': 'Answer factoid questions (who/what/when/where) by searching and extracting answers directly from documents. Sources: Natural Questions, TriviaQA.',
        'multi_hop_reasoning': 'Answer questions requiring chained reasoning across multiple entities or facts. Must decompose the question, search for intermediate facts, and combine them. Sources: HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle.',
        'entity_attribute_lookup': 'Look up specific attributes (occupation, birthplace, genre, etc.) of named entities. Sources: PopQA.',
        'comparison': 'Compare two or more entities on a specific attribute (e.g., same nationality, both in same city). Requires retrieving info about each entity then synthesizing. Sources: HotpotQA, 2WikiMultiHopQA.',
    }

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Query Type: {query_type.upper().replace('_', ' ')}
Description: {type_descriptions.get(query_type, '')}

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 4-6 QUERY-TYPE-SPECIFIC SKILLS for {query_type} tasks. These should be:
1. **Concise** - 1-2 sentences max per skill
2. **Specific** - Apply specifically to {query_type} questions
3. **Actionable** - Clear steps or decision rules
4. **Pattern-based** - Identify what makes success vs failure

Format as JSON array:
[
    {{
        "skill_id": "{query_type[:3]}_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight",
        "when_to_apply": "Specific trigger condition"
    }}
]

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_common_mistakes(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
    """Generate common mistakes to avoid."""

    # Collect all failure patterns
    all_failures = []
    for query_type, data in categorized_memories.items():
        for mem in data['failure'][:5]:
            sg = mem['content'].get('strategic_guidelines', {})
            if 'strategic_guidelines' in sg:
                sg = sg['strategic_guidelines']
            mistakes = sg.get('mistakes_to_avoid', [])
            if mistakes:
                all_failures.append({
                    'query_type': query_type,
                    'goal': mem['content']['task_meta']['original_goal'],
                    'data_source': mem['tags'].get('data_source', ''),
                    'description': mem.get('contextual_description', '')[:200],
                    'mistakes': mistakes[:3]
                })

    failure_data = json.dumps(all_failures[:20], indent=2)

    prompt = f"""You are an expert at analyzing agent failures and distilling them into avoidable mistakes.

Analyze these failure patterns from a Search AI agent that answers questions by issuing search queries and reading retrieved documents:

{failure_data}

Generate 8-12 COMMON MISTAKES to avoid. Format as JSON array:
[
    {{
        "mistake_id": "err_001",
        "description": "What the mistake is (1 sentence)",
        "why_it_happens": "Why agents make this mistake (1 sentence)",
        "how_to_avoid": "Concrete actionable fix (1-2 sentences)"
    }}
]

Focus on:
- Query formulation errors (too vague, wrong entity, not decomposing multi-hop)
- Evidence handling failures (hallucinating without evidence, misreading documents)
- Ambiguous entity resolution failures
- Repeating the same ineffective query
- Failing to decompose complex questions into sub-questions
- Premature answering before gathering sufficient evidence
- Misinterpreting retrieved documents

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def main():
    # Configuration
    memory_json_path = os.path.expanduser(
        "memory_data/search/generated_memories_search.json"
    )
    output_path = os.path.expanduser(
        "memory_data/search/claude_style_skills_search.json"
    )

    print("Loading existing memories...")
    memories = load_memories(memory_json_path)
    print(f"Loaded {len(memories)} memories")

    print("\nCategorizing by query type...")
    categorized = categorize_by_query_type(memories)
    for query_type, data in categorized.items():
        print(f"  {query_type}: {len(data['success'])} success, {len(data['failure'])} failure")

    # Initialize client
    client = OpenAIClient(max_new_tokens=4096, model="o3")

    # Generate skills
    print("\n=== Generating General Skills ===")
    general_skills = generate_general_skills(client, categorized)
    print(f"Generated {len(general_skills)} general skills")

    print("\n=== Generating Query-Type-Specific Skills ===")
    query_type_skills = {}
    for query_type, data in categorized.items():
        print(f"  Processing {query_type}...")
        skills = generate_query_type_skills(
            client, query_type,
            data['success'], data['failure']
        )
        query_type_skills[query_type] = skills
        print(f"    Generated {len(skills)} skills")

    print("\n=== Generating Common Mistakes ===")
    common_mistakes = generate_common_mistakes(client, categorized)
    print(f"Generated {len(common_mistakes)} mistakes")

    # Compile final output
    output = {
        "general_skills": general_skills,
        "query_type_skills": query_type_skills,
        "common_mistakes": common_mistakes,
        "metadata": {
            "source": "generated from Search agent trajectories using o3",
            "total_memories_analyzed": len(memories),
            "query_type_distribution": {
                query_type: {
                    "success": len(data['success']),
                    "failure": len(data['failure'])
                }
                for query_type, data in categorized.items()
            }
        }
    }

    # Save
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    print(f"General skills: {len(general_skills)}")
    print(f"Query-type-specific skills: {sum(len(s) for s in query_type_skills.values())}")
    print(f"Common mistakes: {len(common_mistakes)}")
    print(f"\nSaved to: {output_path}")

    # Print sample output
    print("\n=== Sample General Skills ===")
    for skill in general_skills[:3]:
        print(f"\n[{skill.get('skill_id', 'N/A')}] {skill.get('title', 'N/A')}")
        print(f"  {skill.get('principle', 'N/A')}")


if __name__ == "__main__":
    main()