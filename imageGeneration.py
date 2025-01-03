#NEED TO LOOK INTO AND EDIT
import requests


def getImage(query:str): #Note: Annotate the function type
        #Creating an empty list to store all the images
        links=[]

        API_KEY = "AIzaSyCdUX4c6TjixefRAef98sW9P-i5oGRHNQQ"  # Replace with your Google API key
        CSE_ID = "c3c5aefa71f5b4e29"  # Replace with your Custom Search Engine ID
        query=query.replace("\"", "")#Replacing all of the quotes

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
                print(data)
        except requests.exceptions.RequestException as e:
            print("Request error:", e)
        except ValueError as e:
            print("Error decoding JSON response:", e)
        except Exception as e:
            print("Unexpected error:", e)

        print(links)
        return links