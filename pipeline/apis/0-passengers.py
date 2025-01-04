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

    while url:
        response = requests.get(url, verify=False)
        data = response.json()
        for ship in data["results"]:
            passengers_str = ship["passengers"]
            if passengers_str not in ["n/a", "unknown"]:
                try:
                    passengers = int(passengers_str.replace(
                        ",", "").replace(".", ""))
                    if passengers >= passengerCount:
                        ships.append(ship["name"])
                except ValueError:
                    pass

        url = data["next"]
    return ships
