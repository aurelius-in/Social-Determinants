import requests
from bs4 import BeautifulSoup

# specify the URL of the website to scrape
url = 'https://www.example.com'

# make a GET request to the URL
response = requests.get(url)

# parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# extract the relevant data from the HTML
data = soup.find_all('div', class_='some-class')

# process the data as needed
for item in data:
    # do something with each item
