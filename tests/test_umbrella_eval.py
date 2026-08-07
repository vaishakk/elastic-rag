from umbrella_eval import *

def test_read_system_prompt():
    system_prompt = read_system_prompt()
    print(system_prompt)
    assert system_prompt is not None