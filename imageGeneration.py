#NEED TO LOOK INTO AND EDIT


import requests


class GetImage:
    def __init__(self, query):
        self.query= query

    def run(self):
        #Creating an empty list to store all the images
        links=[]

        API_KEY = "AIzaSyAyy1rWtZbXsHAk6lUGXSiAPtbbCs3UcnA"  # Replace with your Google API key
        CSE_ID = "c3c5aefa71f5b4e29"  # Replace with your Custom Search Engine ID
        query = self.query

        url = f"https://www.googleapis.com/customsearch/v1?q={query}&searchType=image&key={API_KEY}&cx={CSE_ID}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "items" in data:
                for item in data["items"]:
                    links.append(item["link"])
                    print(item["link"])
            else:
                print("No items found in the response.")
        except requests.exceptions.RequestException as e:
            print("Request error:", e)
        except ValueError as e:
            print("Error decoding JSON response:", e)
        except Exception as e:
            print("Unexpected error:", e)

        print(links)
        return links