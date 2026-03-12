import json
def generate_system_prompt():
    pr = '''You are generating a wildfire risk raster for an area based on a satellite images and climate data.
Some factors which increase wildfire risk are: dense and dry vegetation, dry and hot climate, and strong winds, particularly if they align with slopes.
You must reason about the climate data and satellite image and explain in detail the risk level for all visible parts of the satellite image.

At the end, you must finish with a general classification for the wildfire risk in the area from 0 to 9. Your output must end with:

FINAL ANSWER:
n

Where n is a number between 0 and 9, on a newline.'''
    return pr

def build_prompt(climate_data):
    climate_data = json.dumps(climate_data, indent=4, sort_keys=True)
    sys_prompt = generate_system_prompt()
    return sys_prompt + f'\n\nHere is the climate data:\n\n{climate_data}\n\nHere is the satellite image:\n'