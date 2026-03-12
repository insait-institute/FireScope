import json
def generate_system_prompt():
    pr = '''You are evaluating the risk of wildfire in an area based on a satellite image and climate data.
Some factors which increase wildfire risk are: dense and dry vegetation, dry and hot climate, and strong winds, particularly if they align with slopes.
You must respond with a single digit: the risk of wildfire in the area from 0 to 9, where 0 is lowest possible risk and 9 is highest.'''
    return pr

def build_prompt(climate_data):
    climate_data = json.dumps(climate_data, indent=4, sort_keys=True)
    sys_prompt = generate_system_prompt()
    return sys_prompt + f'\n\nHere is the climate data:\n\n{climate_data}\n\nHere is the satellite image:\n'