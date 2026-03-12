def generate_system_prompt_increase(answer):
    return f'''You will receive a chain of thought (CoT) by a wildfire expert evaluating the wildfire risk in the area.

TASK: Produce a new chain-of-thought (CoT) so it argues for the HIGHEST wildfire risk level without making up new facts.

HARD REQUIREMENTS:
- COMPLETELY REMOVE the mention of any factors that decrease risk.
- Emphasize risk-increasing factors already implied in the CoT.
- Only use area features already mentioned in the CoT.
- Do NOT explicitly mention the risk level in the new chain of thought. Use qualitative wording only.

INPUT:
CoT: {answer}

OUTPUT: ONLY the new, rewritten CoT. NOTHING ELSE. Do NOT output the original CoT. Do NOT give your output an explanation or a title. Output ONLY new CoT and nothing else.'''

def generate_system_prompt_decrease(answer):
    return f'''You will receive a chain of thought (CoT) by a wildfire expert evaluating the wildfire risk in the area.

TASK: Produce a new chain-of-thought (CoT) so it argues for the LOWEST wildfire risk level without making up new facts.

HARD REQUIREMENTS:
- COMPLETELY REMOVE the mention of any factors that increase risk.
- Emphasize risk-decreasing factors already implied in the CoT.
- Only use area features already mentioned in the CoT.
- Do NOT explicitly mention the risk level in the new chain of thought. Use qualitative wording only.

INPUT:
CoT: {answer}

OUTPUT: ONLY the new, rewritten CoT. NOTHING ELSE. Do NOT output the original CoT. Do NOT give your output an explanation or a title. Output ONLY new CoT and nothing else.'''

def generate_messages(answer, increase=True):
    if increase:
        generation = generate_system_prompt_increase
    else:
        generation = generate_system_prompt_decrease
    return [{'role': 'system', 'content': generation(answer)}]