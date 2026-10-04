import requests

tiles = {
    "Forest": "T44QKD",
    "Urban": "T44QKE",
    "Agriculture": "T44QKF",
    "Water": "T44PMV",
}

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

for category, tile in tiles.items():
    print(f"\n--- {category} ({tile}) ---")

    params = {
        "$filter": (
            f"contains(Name,'{tile}') and "
            "startswith(Name,'S2C_MSIL2A_')"
        ),
        "$select": "Name,Id",
        "$orderby": "ContentDate/Start desc",
        "$top": 5,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        products = response.json().get("value", [])

        if not products:
            print("No matching products found.")
        else:
            for product in products:
                print(product["Name"])

    except requests.RequestException as error:
        print("Search error:", error)
