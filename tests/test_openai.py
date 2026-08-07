import json

from rag.rag.LLMs.open_ai import *

class TestResponse(BaseModel):
    art1: str
    art2: int
    art3: list[int]

def test_openai_embedding():
    texts = [
        'text 1',
        'text 2',
    ]
    response = call_api(OpenAI(), input_data=texts, call_type='embed')
    assert response is not None

def test_openai_chat_call():
    messages = [
        {
            'role': 'system',
            'content': 'Always respond only "Hi!", whatever the user message is.'
        },
        {
            'role': 'user',
            'content': 'Hello!',
        }
    ]
    response = call_api(OpenAI(), input_data=messages, call_type='chat')
    assert response is not None
    assert response.choices[0].message.content == 'Hi!'

def test_openai_chat():
    response = chat_completion(
        OpenAI(),
        system_prompt='Always respond only "Hi!", whatever the user message is.',
        query='Hello!')
    assert response is not None
    assert response == 'Hi!'

def test_openai_parse():
    messages = [
        {
            'role': 'system',
            'content': ''
        },
        {
            'role': 'user',
            'content': 'Give a random integer, a list of 5 integers and a random string',
        }
    ]
    response = call_api(OpenAI(), input_data=messages, response_format=TestResponse, call_type='parse')
    parsed = json.loads(response.output[0].content[0].text)
    assert response is not None
    assert len(parsed) == 3
    assert all(art in parsed for art in ['art1', 'art2', 'art3'])
    assert type(parsed['art1']) == str
    assert type(parsed['art2']) == int
    assert type(parsed['art3']) == list