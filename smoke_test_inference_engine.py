#!/usr/bin/env python3
"""Non-interactive smoke test for the integrated SATAgent pipeline."""
import json
import random
from inference_engine import SATAgent

agent = SATAgent()

splits = json.load(open("dataset/sat/product_splits.json"))
random.seed(11)
test_asins = random.sample(splits["eval_asins"], 4)

scenarios = [
    (test_asins[0], "3", "How is the battery life on this phone?"),
    (test_asins[1], "1", "Is the camera any good?"),
    (test_asins[2], "2", "is it good?"),  # should trigger CLARIFY
    (test_asins[3], "5", "Does this phone support satellite messaging and Thread/Matter smart home protocols?"),  # should abstain
]

for asin, persona_id, query in scenarios:
    print("\n" + "#" * 90)
    agent.select_product(asin)
    agent.set_user(persona_id)
    print(f"QUERY: {query}")
    agent.chat(query)
