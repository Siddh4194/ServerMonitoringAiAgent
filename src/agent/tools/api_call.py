import requests


def get_request(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()       # Assumes the response is JSON
    except Exception as e:
        return {"error": f"GET request failed: {str(e)}"}