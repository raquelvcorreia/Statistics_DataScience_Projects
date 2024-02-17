import requests
import json

import pandas as pd

# url = "https://api2.openreview.net/notes?content.venue=NeurIPS%202023%20oral&details=replyCount%2Cpresentation&domain=NeurIPS.cc%2F2023%2FConference&limit=100&offset=0"
#
#
# response = requests.get(url)
#
#
# # Convert json into dictionary
# response_dict = response.json()
# # Pretty Printing JSON string back
# json_object_rawdata = json.dumps(response_dict, indent=4, sort_keys=True)
#
# # Writing to raw_data.json
# with open("raw_data.json", "w") as outfile:
#     outfile.write(json_object_rawdata)


# Opening JSON file
with open('raw_data.json', 'r') as openfile:
    # Reading from json file
    json_object_rawdata = json.load(openfile)



abstracts = []
for i in range(len(json_object_rawdata['notes'])):
    record = json_object_rawdata['notes'][i]
    title = record['content']['title']['value']
    keywords = record['content']['keywords']['value']
    authors = record['content']['authors']['value']
    keywords = record['content']['keywords']['value']
    abstract_body = record['content']['abstract']['value']
    link_data = 'https://openreview.net/' + record['content']['pdf']['value']
    abstracts.append([title, keywords, authors, abstract_body, link_data])

# Store the information in a pandas dataframe
df_data = pd.DataFrame(abstracts, columns=['Title', 'Keywords', 'Authors', 'Abstract_body', 'Link_data'])

df_data.to_csv('oral_presentation_data.csv', index=False)
print(df_data.head())



