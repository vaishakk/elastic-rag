from rag.rag.LLMs.open_ai import *

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