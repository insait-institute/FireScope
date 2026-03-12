import json
def generate_system_prompt():
    pr = '''You are generating a wildfire risk raster for an area based on a satellite images and climate data.
Some factors which increase wildfire risk are: dense and dry vegetation, dry and hot climate, and strong winds, particularly if they align with slopes.
You must consider all the data your are given and generate the risk raster for this area.'''
    return pr

def build_prompt(climate_data):
    climate_data = json.dumps(climate_data, indent=4, sort_keys=True)
    sys_prompt = generate_system_prompt()
    return sys_prompt + f'\n\nHere is the climate data:\n\n{climate_data}\n\nHere is the satellite image:\n'