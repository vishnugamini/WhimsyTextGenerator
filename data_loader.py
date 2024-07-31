import requests
url = ''
response = requests.get(url)

with open("input.txt",'wb') as file:
    file.write(response.content)
    