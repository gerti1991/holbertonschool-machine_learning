#!/usr/bin/env python3
"""
Test
"""

import requests


def sentientPlanets():
    """
    Test
    """
    # url = "https://swapi-api.hbtn.io/api/planets/"
    urls = speciesUrl()
    planets = []
    for url in urls:
        url_ = url
        while url_:
            response = requests.get(url_)
            data = response.json()
            planets.append(data["name"])
            url_ = data.get("next")
    return planets


def speciesUrl():
    """
    Test
    """
    urlPlanets = []
    url = "https://swapi-api.hbtn.io/api/species/"
    while url:
        response = requests.get(url)
        data = response.json()
        for species in data["results"]:
            if species["designation"] == "sentient" and species["homeworld"]:
                urlPlanets.append(species["homeworld"])
        url = data.get("next")
    return urlPlanets
# print(speciesUrl())
