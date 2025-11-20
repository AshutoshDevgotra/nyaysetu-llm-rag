import requests
import json

def test_backend():
    url = "http://localhost:8082/ask"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "What is the punishment for theft under IPC?"
    }
    
    print(f"Testing {url} with query: '{data['query']}'")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response Received:")
            print(f"Answer: {result.get('answer')}")
            
            print("\n📚 Sources:")
            sources = result.get('sources', [])
            if sources:
                for i, source in enumerate(sources):
                    print(f"[{i+1}] {source.get('metadata', {}).get('source', 'unknown')}")
                    print(f"    Preview: {source.get('content', '')[:100]}...")
            else:
                print("❌ No sources returned!")
                
            print("\n🔍 Full JSON:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_backend()
