def generate_system_prompt(cot):
    return f'''You are given a chain of thought from a wildfire risk model. Your task is to paraphrase it - rewrite it without changing its meaning.

Requirements:

- Rewrite the text by changing the wording slightly while keeping the structure, order, and meaning exactly the same.

- Keep the length and level of detail as close as possible to the original.

- Do not add, remove, or reorder any information — only substitute words or short phrases with near synonyms.

COT: 

{cot}'''

def generate_messages(cot):
    return [{'role': 'system', 'content': generate_system_prompt(cot)}]