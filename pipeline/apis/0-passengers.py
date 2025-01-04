#!/usr/bin/env python3
"""
Test
"""

import requests


def availableShips(passengerCount):
    """
    Test
    """
    url = "https://swapi.dev/api/starships/"
    ships = []
    response = requests.get(url, verify=False)
    data = response.json()
    for ship in data["results"]:
        if int(ship["passengers"]) >= passengerCount:
            ships.append(ship["name"])
    url = data["next"]
    return ships
