from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from groq import Groq

from restaurant_reco_phase3.models import RecommendationItem, RecommendationRequest

logger = logging.getLogger(__name__)


def generate_explanations(
    request: RecommendationRequest,
    items: List[RecommendationItem],
    client: Groq | None = None,
) -> tuple[List[RecommendationItem], str | None]:
    """
    Given a list of recommended items, use the Groq API to generate a personalized
    explanation for each item and a final overall summary.

    Returns:
        (updated_items, global_summary)
    """
    if not items:
        return items, None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment; skipping LLM explanations.")
        return items, None

    groq_client = client or Groq(api_key=api_key)

    # Prepare context for the prompt
    context = []
    for idx, item in enumerate(items, 1):
        context.append(
            f"{idx}. {item.name} | Cuisines: {', '.join(item.cuisines)} | Rating: {item.rating} | "
            f"Budget: {item.cost_for_two} | Match Reasons: {', '.join(item.reasons)}"
        )
    context_str = "\n".join(context)

    req_details = []
    req_details.append(f"Place: {request.place}")
    if request.cuisines:
        req_details.append(f"Cuisines: {', '.join(request.cuisines)}")
    if request.budget:
        req_details.append(f"Budget: {request.budget.min} - {request.budget.max}")
    if request.min_rating:
        req_details.append(f"Min Rating: {request.min_rating}")
    req_str = " | ".join(req_details)

    system_prompt = """You are a helpful and persuasive restaurant recommendation assistant.
The user has provided their preferences, and the system has filtered and ranked the top restaurants.
Your job is to generate:
1. A short, persuasive 1-2 sentence explanation for EACH recommended restaurant, highlighting why it's a great match given the user's constraints and the system's match reasons.
2. A very brief 1-2 sentence overall summary for the final response.

You must respond ONLY with a valid JSON document in the following format:
{
  "summary": "...",
  "explanations": {
    "restaurant_id_1": "Explanation for restaurant 1...",
    "restaurant_id_2": "Explanation for restaurant 2..."
  }
}
Do not include markdown code block formatting (like ```json), just the raw JSON text. Do not invent details not present in the input.
"""
    user_prompt = f"""User Preferences:
{req_str}

Top Recommendations:
{context_str}

Use the specific `restaurant_id` as the keys in the JSON explanations map.

Here is the structured data to reference:
"""
    # Append structured data so the LLM has exact IDs
    for item in items:
        user_prompt += f"\n- ID: {item.restaurant_id}, Name: {item.name}"

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to supported model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if not content:
            return items, None

        parsed = json.loads(content)
        summary = parsed.get("summary")
        exps = parsed.get("explanations", {})

        # Apply explanations to items
        for item in items:
            if item.restaurant_id in exps:
                item.explanation = exps[item.restaurant_id]

        return items, summary

    except Exception as e:
        logger.error(f"Failed to generate LLM explanations: {e}")
        return items, None
