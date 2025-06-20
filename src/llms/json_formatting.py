#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json

def json_format(data_path, json_path):
    samples = []
    with open(data_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                line = line.strip()
                line = line.lower()
                if line != '':
                    dic = {}
                    words, tuples = line.split('####')
                    dic["Text"] = words
                    dic["Sentiment Elements"] = tuples
                    samples.append(dic)
                
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


# In[10]:


json_format("data/asqp/R15/train.txt", "R15_train.json")
json_format("data/asqp/R15/dev.txt", "R15_dev.json")
json_format("data/asqp/R16/train.txt", "R16_train.json")
json_format("data/asqp/R16/dev.txt", "R16_dev.json")

json_format("data/acos/Lap/train.txt", "Lap_train.json")
json_format("data/acos/Lap/dev.txt", "Lap_dev.json")
json_format("data/acos/Rest/train.txt", "Rest_train.json")
json_format("data/acos/Rest/dev.txt", "Rest_dev.json")

json_format("data/memd/M-Rest/train.txt", "M-Rest_train.json")
json_format("data/memd/M-Rest/dev.txt", "M-Rest_dev.json")
json_format("data/memd/M-Lap/train.txt", "M-Lap_train.json")
json_format("data/memd/M-Lap/dev.txt", "M-Lap_dev.json")
json_format("data/memd/Books/train.txt", "Books_train.json")
json_format("data/memd/Books/dev.txt", "Books_dev.json")
json_format("data/memd/Clothing/train.txt", "Clothing_train.json")
json_format("data/memd/Clothing/dev.txt", "Clothing_dev.json")
json_format("data/memd/Hotel/train.txt", "Hotel_train.json")
json_format("data/memd/Hotel/dev.txt", "Hotel_dev.json")

