import requests

product_id = None
product_name = "S2C_MSIL2A_20260924T050651_N0513_R019_T44QKD_20260924T100421.SAFE"

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
response = requests.get(
    url,
    params={
        "$filter": f"Name eq '{product_name}'",
        "$select": "Name,Id,ContentLength",
    },
    timeout=30,
)
response.raise_for_status()

products = response.json().get("value", [])

if not products:
    print("Product not found.")
else:
    product = products[0]
    size_gb = product.get("ContentLength", 0) / (1024 ** 3)
    print("Product:", product["Name"])
    print("Product ID:", product["Id"])
    print(f"Size: {size_gb:.2f} GB")
