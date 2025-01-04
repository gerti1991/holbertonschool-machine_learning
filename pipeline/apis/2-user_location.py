#!/usr/bin/env python3
"""
Test
"""

import requests
import sys
import time


def get_user_location(url):
    """
    Fetches the location of a GitHub user from the given API URL.
    """
    response = requests.get(url)
    if response.status_code == 404:
        print("Not found")
        return
    if response.status_code == 403:
        reset_time = int(response.headers.get('X-RateLimit-Reset'))
        current_time = int(time.time())
        reset_in_minutes = (reset_time - current_time) // 60
        print(f"Reset in {reset_in_minutes} min")
        return
    if response.status_code == 200:
        data = response.json()
        if 'location' in data and data['location']:
            print(data['location'])
        else:
            print("Location not available")
    else:
        print("Error: Unable to fetch data")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <github_api_url>")
        sys.exit(1)
    url = sys.argv[1]
    get_user_location(url)
