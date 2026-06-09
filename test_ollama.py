"""Test Ollama connection and model availability"""
import ollama

def test_ollama():
    try:
        models = ollama.list()
        print("Available models:")
        for model in models.get('models', []):
            print(f"  - {model['name']} ({model['size']})")
        
        response = ollama.chat(model='deepseek-r1:7b', messages=[
            {'role': 'user', 'content': 'Hello!'} 
        ])
        print(f"\nTest response: {response['message']['content']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama()