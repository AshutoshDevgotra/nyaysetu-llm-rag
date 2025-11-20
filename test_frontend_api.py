import requests
import json

def test_frontend_api():
    # Test the Next.js API route
    url = "http://localhost:3000/api/query"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "What is the punishment for theft under IPC?"
    }
    
    print(f"Testing Frontend API: {url}")
    print(f"Query: '{data['query']}'")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=35)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response Received:")
            print(f"Answer: {result.get('answer', 'N/A')[:200]}...")
            print(f"\nMetadata: {json.dumps(result.get('metadata', {}), indent=2)}")
            
            sources = result.get('sources')
            if sources:
                print(f"\n📚 Sources: {len(sources)} found")
            else:
                print("\n❌ No sources in response")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_frontend_api()
