import urllib.request
import sys

def main():
    url = "http://localhost:8501"
    print(f"Testing responsiveness of Streamlit server at {url}...")
    try:
        # Fetch the root of the Streamlit server
        response = urllib.request.urlopen(url, timeout=5)
        status_code = response.getcode()
        print(f"Server responded with status code: {status_code}")
        
        if status_code == 200:
            print("SUCCESS: Streamlit server is running and responsive!")
            sys.exit(0)
        else:
            print(f"FAILURE: Server returned unexpected status code {status_code}")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILURE: Could not connect to Streamlit server at {url}. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
