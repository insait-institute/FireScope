def generate_system_prompt():
    return '''You are given a chain of thought from a wildfire risk model. Your task is to summarize the wildfire risk factors considered in the model's reasoning clearly and objectively, without specifying the risk level in any way.

Requirements:

- Do **NOT** use any words or terms which allude to a particular risk level (e.g. "moderate", "high", etc). This is critical.

- Do **NOT** include or infer the model's final risk classification, numerical risk score, or probability.

- Summarize only the evidence, reasoning steps, and factors considered.

- Maintain a **completely neutral tone** so that the text does not suggest a final risk level **at all**.

- Keep the summary as short as possible, mention only the key points, and omit all redundant phrases.

- Output should read like a professional briefing note of all risk increasing or dampening factors in the area. It should **not** specify a particular risk level in any way.

Output format:
A single, well-structured explanation that captures the risk factors considered in the model's reasoning clearly, **without** alluding to any risk level **at all**, quantitatively or qualitatively.

**Do not use any terms that imply a risk level, such as 'high,' 'moderate,' 'low,' 'severe,' 'elevated,' or any numerical score.** Avoid any language that suggests the likelihood, intensity, or severity of a fire. Focus only on the evidence, factors, and their relationships — without concluding or ranking the risk.'''

def generate_messages():
    return [{'role': 'system', 'content': generate_system_prompt()}]