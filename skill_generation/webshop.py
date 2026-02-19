"""
Generate Claude-style skills for WebShop agent using o3 API.

Claude-style skills have these characteristics:
1. Concise and actionable - focus on what to do, not verbose descriptions
2. General principles - transferable across similar tasks
3. Failure-aware - include common pitfalls and how to avoid them
4. Hierarchical - general skills + task-specific skills

Output format:
{
    "general_skills": [
        {
            "skill_id": "...",
            "title": "...",  # Short title
            "principle": "...",  # The core insight/rule
            "when_to_apply": "...",  # Triggering condition
            "example": "..."  # Brief concrete example (optional)
        }
    ],
    "task_specific_skills": {
        "apparel": [...],
        "footwear": [...],
        "home_decor": [...],
        "electronics": [...],
        "accessories": [...],
        "other": [...]
    },
    "common_mistakes": [
        {
            "mistake_id": "...",
            "description": "...",  # What the mistake is
            "why_it_happens": "...",  # Why agents make this mistake
            "how_to_avoid": "..."  # Concrete fix
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


def load_memories(json_paths: List[str]) -> List[Dict]:
    """Load existing memory data from multiple files."""
    all_memories = []
    for path in json_paths:
        with open(path, 'r') as f:
            memories = json.load(f)
            all_memories.extend(memories)
    return all_memories


def categorize_by_product_type(memories: List[Dict]) -> Dict[str, Dict[str, List]]:
    """Categorize memories by product type and outcome."""
    categorized = {
        'apparel': {'success': [], 'failure': []},
        'footwear': {'success': [], 'failure': []},
        'home_decor': {'success': [], 'failure': []},
        'electronics': {'success': [], 'failure': []},
        'accessories': {'success': [], 'failure': []},
        'beauty_health': {'success': [], 'failure': []},
        'other': {'success': [], 'failure': []},
    }

    apparel_keywords = [
        'shirt', 'dress', 't-shirt', 'polo', 'pants', 'jeans', 'jacket', 'coat',
        'sweater', 'blouse', 'skirt', 'shorts', 'underwear', 'swimsuit', 'swimwear',
        'hoodie', 'vest', 'cardigan', 'suit', 'blazer', 'tee', 'top'
    ]
    footwear_keywords = [
        'shoe', 'boot', 'sandal', 'sneaker', 'slipper', 'loafer', 'heel', 'flat',
        'oxford', 'pump', 'moccasin', 'flip-flop', 'footwear'
    ]
    home_keywords = [
        'pillow', 'curtain', 'rug', 'mat', 'blanket', 'bedding', 'towel', 'lamp',
        'decor', 'furniture', 'cushion', 'sheet', 'tablecloth', 'vase'
    ]
    electronics_keywords = [
        'phone', 'laptop', 'tablet', 'computer', 'headphone', 'earphone', 'speaker',
        'charger', 'cable', 'mouse', 'keyboard', 'monitor', 'camera', 'watch',
        'smartwatch', 'electronic', 'device', 'gadget', 'armoires'
    ]
    accessories_keywords = [
        'bag', 'wallet', 'belt', 'hat', 'cap', 'scarf', 'glove', 'jewelry',
        'necklace', 'bracelet', 'ring', 'earring', 'sunglasses', 'glasses', 'watch',
        'purse', 'backpack', 'handbag', 'tie', 'bow'
    ]
    beauty_health_keywords = [
        'makeup', 'cosmetic', 'skincare', 'lotion', 'cream', 'shampoo', 'conditioner',
        'perfume', 'cologne', 'brush', 'bathing', 'soap', 'body wash', 'nail',
        'lipstick', 'mascara', 'foundation', 'serum', 'moisturizer'
    ]

    for mem in memories:
        goal = mem['content']['task_meta']['original_goal'].lower()
        outcome = 'success' if mem['tags']['outcome'] == 'Success' else 'failure'

        # Categorize based on goal text
        product_type = 'other'

        for kw in apparel_keywords:
            if kw in goal:
                product_type = 'apparel'
                break

        if product_type == 'other':
            for kw in footwear_keywords:
                if kw in goal:
                    product_type = 'footwear'
                    break

        if product_type == 'other':
            for kw in home_keywords:
                if kw in goal:
                    product_type = 'home_decor'
                    break

        if product_type == 'other':
            for kw in electronics_keywords:
                if kw in goal:
                    product_type = 'electronics'
                    break

        if product_type == 'other':
            for kw in accessories_keywords:
                if kw in goal:
                    product_type = 'accessories'
                    break

        if product_type == 'other':
            for kw in beauty_health_keywords:
                if kw in goal:
                    product_type = 'beauty_health'
                    break

        categorized[product_type][outcome].append(mem)

    return categorized


def extract_patterns(memories: List[Dict], limit: int = 10) -> str:
    """Extract key patterns from memories for prompt."""
    patterns = []
    for mem in memories[:limit]:
        goal = mem['content']['task_meta']['original_goal']
        refined_traj = mem['content'].get('refined_trajectory') or {}
        trajectory = refined_traj.get('refined_trajectory', []) if refined_traj else []
        strategic = mem['content'].get('strategic_guidelines') or {}

        # Handle nested strategic_guidelines structure
        if isinstance(strategic, dict) and 'strategic_guidelines' in strategic:
            strategic = strategic['strategic_guidelines']

        mistakes = strategic.get('mistakes_to_avoid', []) if strategic else []
        planning = strategic.get('planning_pattern', '') if strategic else ''

        pattern = {
            'goal': goal,
            'steps': [{'action': s.get('action', ''), 'reasoning': s.get('reasoning', '')}
                     for s in trajectory[:5]],
            'planning_pattern': planning,
            'mistakes': mistakes[:3] if mistakes else []
        }
        patterns.append(pattern)

    return json.dumps(patterns, indent=2)


def generate_general_skills(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
    """Generate general skills using o3."""

    # Collect sample data across all product types
    all_successes = []
    all_failures = []
    for product_type, data in categorized_memories.items():
        all_successes.extend(data['success'][:8])
        all_failures.extend(data['failure'][:8])

    success_patterns = extract_patterns(all_successes, limit=15)
    failure_patterns = extract_patterns(all_failures, limit=15) if all_failures else "[]"

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Analyze these successful and failed trajectories from an AI agent operating in an online shopping environment (WebShop).

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 10-15 GENERAL SKILLS that apply across ALL product types in web shopping. These should be:
1. **Concise** - Each skill should be 1-2 sentences max
2. **Actionable** - Clear what to do, not vague principles
3. **Transferable** - Apply to apparel, footwear, electronics, home decor, accessories, etc.
4. **Failure-aware** - Derived from what could go wrong

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
- Search query formulation strategies (how to encode constraints)
- Product selection heuristics (which product to click)
- Option configuration order (size, color, etc.)
- Constraint verification before purchase
- Navigation and exploration patterns
- Price constraint handling
- Attribute matching strategies

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    # Parse JSON from response
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_task_specific_skills(client: OpenAIClient, product_type: str,
                                   successes: List[Dict], failures: List[Dict]) -> List[Dict]:
    """Generate task-specific skills for a particular product type."""

    if not successes and not failures:
        return []

    success_patterns = extract_patterns(successes[:10])
    failure_patterns = extract_patterns(failures[:10]) if failures else "[]"

    product_descriptions = {
        'apparel': 'Clothing items like shirts, dresses, pants, jackets - often requiring size and color selection',
        'footwear': 'Shoes, boots, sandals, slippers - requiring size and sometimes color/style selection',
        'home_decor': 'Home decoration items like pillows, curtains, rugs - often with size and color options',
        'electronics': 'Electronic devices and accessories like phones, chargers, computer accessories',
        'accessories': 'Fashion accessories like bags, wallets, jewelry, hats',
        'beauty_health': 'Beauty and health products like skincare, cosmetics, bathing accessories',
        'other': 'Miscellaneous products that do not fit into other categories'
    }

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Product Type: {product_type.upper().replace('_', ' ')}
Description: {product_descriptions.get(product_type, '')}

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 4-6 TASK-SPECIFIC SKILLS for shopping {product_type.replace('_', ' ')} products. These should be:
1. **Concise** - 1-2 sentences max per skill
2. **Specific** - Apply specifically to {product_type.replace('_', ' ')} shopping
3. **Actionable** - Clear steps or decision rules
4. **Pattern-based** - Identify what makes success vs failure

Format as JSON array:
[
    {{
        "skill_id": "{product_type[:3]}_001",
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

    # Collect all success patterns to infer potential mistakes
    all_successes = []
    for product_type, data in categorized_memories.items():
        for mem in data['success'][:5]:
            strategic = mem['content'].get('strategic_guidelines', {})
            if isinstance(strategic, dict) and 'strategic_guidelines' in strategic:
                strategic = strategic['strategic_guidelines']
            mistakes = strategic.get('mistakes_to_avoid', [])
            planning = strategic.get('planning_pattern', '')
            all_successes.append({
                'product_type': product_type,
                'goal': mem['content']['task_meta']['original_goal'],
                'planning_pattern': planning,
                'mistakes': mistakes[:3] if mistakes else []
            })

    success_data = json.dumps(all_successes[:20], indent=2)

    prompt = f"""You are an expert at analyzing agent behaviors and identifying potential failure modes.

Analyze these successful shopping trajectories from an AI agent to infer COMMON MISTAKES that could happen:

{success_data}

Based on the task patterns, generate 10-15 COMMON MISTAKES to avoid in web shopping. Think about:
- What could go wrong during search query formulation
- What could go wrong during product selection
- What could go wrong during option configuration
- What could go wrong before purchasing

Format as JSON array:
[
    {{
        "mistake_id": "err_001",
        "description": "What the mistake is (1 sentence)",
        "why_it_happens": "Why agents make this mistake (1 sentence)",
        "how_to_avoid": "Concrete actionable fix (1-2 sentences)"
    }}
]

Focus on:
- Search query errors (too broad, too narrow, missing constraints)
- Product selection errors (wrong category, ignoring price)
- Option configuration errors (forgetting size/color, wrong order)
- Constraint verification failures (not checking price before buy)
- Navigation mistakes (going back unnecessarily, missing products)

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
    memory_json_paths = [
        os.path.expanduser("memory_data/webshop/generated_memories_webshop_100.json"),
        os.path.expanduser("memory_data/webshop/generated_memories_webshop_101-200.json"),
    ]
    output_path = os.path.expanduser(
        "memory_data/webshop/claude_style_skills.json"
    )

    print("Loading existing memories...")
    memories = load_memories(memory_json_paths)
    print(f"Loaded {len(memories)} memories")

    print("\nCategorizing by product type...")
    categorized = categorize_by_product_type(memories)
    for product_type, data in categorized.items():
        print(f"  {product_type}: {len(data['success'])} success, {len(data['failure'])} failure")

    # Initialize client
    client = OpenAIClient(max_new_tokens=4096, model="o3")

    # Generate skills
    print("\n=== Generating General Skills ===")
    general_skills = generate_general_skills(client, categorized)
    print(f"Generated {len(general_skills)} general skills")

    print("\n=== Generating Task-Specific Skills ===")
    task_specific_skills = {}
    for product_type, data in categorized.items():
        if data['success'] or data['failure']:  # Only process if we have data
            print(f"  Processing {product_type}...")
            skills = generate_task_specific_skills(
                client, product_type,
                data['success'], data['failure']
            )
            task_specific_skills[product_type] = skills
            print(f"    Generated {len(skills)} skills")

    print("\n=== Generating Common Mistakes ===")
    common_mistakes = generate_common_mistakes(client, categorized)
    print(f"Generated {len(common_mistakes)} mistakes")

    # Compile final output
    output = {
        "general_skills": general_skills,
        "task_specific_skills": task_specific_skills,
        "common_mistakes": common_mistakes,
        "metadata": {
            "source": "generated from WebShop trajectories using o3",
            "total_memories_analyzed": len(memories),
            "product_distribution": {
                product_type: {
                    "success": len(data['success']),
                    "failure": len(data['failure'])
                }
                for product_type, data in categorized.items()
            }
        }
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    print(f"General skills: {len(general_skills)}")
    print(f"Task-specific skills: {sum(len(s) for s in task_specific_skills.values())}")
    print(f"Common mistakes: {len(common_mistakes)}")
    print(f"\nSaved to: {output_path}")

    # Print sample output
    print("\n=== Sample General Skills ===")
    for skill in general_skills[:3]:
        print(f"\n[{skill.get('skill_id', 'N/A')}] {skill.get('title', 'N/A')}")
        print(f"  {skill.get('principle', 'N/A')}")


if __name__ == "__main__":
    main()
