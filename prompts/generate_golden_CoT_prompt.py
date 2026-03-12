import json

def generate_system_prompt(climate_data, true_risk_level):
    return '''You are generating a wildfire risk raster for an area based on a satellite images and climate data.
Some factors which increase wildfire risk are: dense and dry vegetation, dry and hot climate, and strong winds, particularly if they align with slopes.
You must reason about the climate data and satellite image and explain in detail the risk level for all visible parts of the satellite image.

At the end, you must finish with a general classification for the wildfire risk in the area from 0 to 9. Your output must end with:

FINAL ANSWER:
''' + str(true_risk_level) + '''

You must start as if you don't know anything about the risk level, and as you go through the data, you analyze it and your chain of thought leads to the given final answer. DO NOT just repeat the data - any data you go through, you analyze in the context of wildfire risk. Your chain of thought eventually naturally arrives at the final answer.

You do not know the label in advance, and as you think and reason through the data, you come up with a final answer of ''' + str(true_risk_level) + ''' on your own.

You must ALWAYS keep hidden that the final answer has been revealed to you in advance. You must arrive to it on your own, simulating forward reasoning.

CLIMATE CONDITIONS: ''' + json.dumps(climate_data, indent=4, sort_keys=True) + '''

SATELLITE IMAGE:
'''