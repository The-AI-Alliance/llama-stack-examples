import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
from langchain_openai import ChatOpenAI


custom_model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="NA",
    openai_api_base="http://localhost:8321/v1/openai/v1"
)


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,model=custom_model
)

name_normalizer_prompt = """
Find and return the following information about the vendor:
  1. legal_name
  2. official_website
  3. ultimate_parent_name

requirements:
  - If a value cannot be reliably determined, return "NotAvailable".
  - The final output must be formatted strictly in valid YAML.

vendor_name: {vendor}

expected_output_format:
  legal_name: <string or NotAvailable>
  official_website: <string or NotAvailable>
  ultimate_parent_name: <string or NotAvailable>
"""
'''
result = agent.invoke({
    "messages": [
        {"role": "user", "content": name_normalizer_prompt.format(vendor="IBM Canada")}
    ]
})

#print(result['messages'][-1].content)

output_text = result['messages'][-1].content

fields = {
    "legal_name": "NotAvailable",
    "official_website": "NotAvailable",
    "ultimate_parent_name": "NotAvailable",
}

for line in output_text.splitlines():
    s = line.strip()
    for key in fields.keys():
        prefix = f"{key}:"
        if s.startswith(prefix):
            value = s[len(prefix):].strip()
            # strip surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            fields[key] = value

legal_name = fields["legal_name"]
official_website = fields["official_website"]
ultimate_parent_name = fields["ultimate_parent_name"]

print("legal_name:", legal_name)
print("official_website:", official_website)
print("ultimate_parent_name:", ultimate_parent_name)
'''

legal_name= "IBM Canada Ltd."
official_website = "https://www.ibm.com/ca-en"
ultimate_parent_name = "International Business Machines Corporation"

emission_commitments_prompt = """
You are a precise research assistant.

Goal:
- Find the vendor's public greenhouse-gas emissions commitments (e.g., "carbon neutral by 2030", "net zero by 2050") and their target dates.
- Prefer the vendor’s own sources (official site, ESG/CSR reports, press releases). If not available, use reputable third parties (SBTi, CDP) and cite them.

Inputs:
  legal_name: {legal_name}
  official_website: {official_website}
  ultimate_parent_name: {ultimate_parent_name}

Rules:
  - Return a list of distinct commitments; include both near-term milestones (e.g., 2030) and long-term targets (e.g., 2050) if stated.
  - Sort by date ascending when multiple dates are available.
  - "date" should be the most specific available: YYYY-MM-DD, else YYYY-MM, else YYYY.
  - If nothing can be reliably determined, return an empty list.
  - Each citation must be a single URL.
  - Output must be strictly valid YAML. No explanations or code fences.

expected_output_format:
  commitments:
    - commitment: <string>
      date: <YYYY[-MM][-DD]>
      citation: <URL>
"""

# Example invocation
commitment_result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": emission_commitments_prompt.format(
                legal_name=legal_name,
                official_website=official_website,
                ultimate_parent_name=ultimate_parent_name,
            ),
        }
    ]
})

output_text = commitment_result["messages"][-1].content

# --- ultra-simple extractor (line starts only) ---
commitments = []
current = None
in_list = False

for raw in output_text.splitlines():
    line = raw.strip()
    if not line or line.startswith("```"):
        continue

    if line.startswith("commitments:"):
        in_list = True
        continue
    if not in_list:
        continue

    if line.startswith("- "):
        # start a new item
        current = {"commitment": "NotAvailable", "date": "NotAvailable", "citation": "NotAvailable"}
        commitments.append(current)
        # handle inline "- commitment: <value>"
        rest = line[2:].strip()
        if rest.startswith("commitment:"):
            val = rest[len("commitment:"):].strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            current["commitment"] = val
        continue

    if current is None:
        continue

    for key in ("commitment", "date", "citation"):
        prefix = f"{key}:"
        if line.startswith(prefix):
            val = line[len(prefix):].strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            current[key] = val

# Example usage/output
for i, it in enumerate(commitments, 1):
    print(f"[{i}] commitment:", it["commitment"])
    print("    date:", it["date"])
    print("    citation:", it["citation"])

