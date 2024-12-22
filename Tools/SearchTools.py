import os
import requests
import json
from langchain.tools import tool


class SearchTools:
    """
    Tools for performing internet searches using the Serper.dev API.
    """

    @staticmethod
    @tool("Search the internet")
    def search_internet(query: str) -> str:
        """
        Search the internet for information based on the provided query.

        Parameters:
            query (str): The search query to execute.

        Returns:
            str: A formatted string of search results or an error message if no results are found.
        """
        if not query:
            return "No query provided. Please specify what you'd like to search for."

        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            return "API key for Serper.dev is missing. Please set SERPER_API_KEY in your environment."

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()

            if "organic" not in data:
                return "No results found or API response was invalid."

            results = data.get("organic", [])
            formatted_results = []
            for result in results[:4]:  # Limit to top 4 results
                formatted_results.append(f"Title: {result.get('title', 'N/A')}\n"
                                         f"Link: {result.get('link', 'N/A')}\n"
                                         f"Snippet: {result.get('snippet', 'N/A')}\n"
                                         "-----------------")

            return "\n".join(formatted_results) if formatted_results else "No relevant results found."
        except requests.exceptions.RequestException as e:
            return f"An error occurred while connecting to the API: {e}"
        except json.JSONDecodeError:
            return "An error occurred while decoding the API response."
