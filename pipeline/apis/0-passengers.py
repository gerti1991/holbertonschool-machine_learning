#!/usr/bin/env python3
"""
Test
"""

import requests


def availableShips(passengerCount):
    """
    Test
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    response = requests.get(url, verify=False)
    data = response.json()
    for ship in data["results"]:
        if ship["passengers"] != "n/a":
            passengers = int(ship["passengers"].replace(
                ",", "").replace(".", ""))
            if passengers >= passengerCount:
                ships.append(ship["name"])
    url = data["next"]
    return ships
