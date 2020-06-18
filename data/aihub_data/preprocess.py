import pandas as pd
import glob
from konlpy.tag import Kkma
from konlpy.utils import pprint
from tqdm import tqdm

kkma = Kkma()

existing_xlsx_list = glob.glob("./existing/*.xlsx")
emerging_xlsx_list = glob.glob("./emerging/*.xlsx")
print(existing_xlsx_list)

existing_dataset = {}
emerging_dataset = {}

def parse_sentence_from_dataset(dataset_path, max_len):
    intent = dataset_path.split(" ")[1].split("(")[0]
    dataset=pd.read_excel(dataset_path)
    df = dataset[["CATEGORY","SENTENCE","지식베이스"]].dropna()
    if intent=='음식점':
        df_ex = df[df['CATEGORY']=='홀서빙음식점']
        df_ex = df_ex[df_ex.지식베이스.str.contains('메뉴|재료')]
        print(len(df_ex))
        df_em = df[df['CATEGORY']=='배달음식점']
        df_em = df_em[df_em.지식베이스.str.contains('메뉴|재료')]
        print(len(df_em))

        existing_dataset[intent]=df_ex['SENTENCE'].to_list()
        emerging_dataset['배달 음식점']=df_em['SENTENCE'].to_list()   
    elif intent=='의류':
        df_ex = df[df['CATEGORY'].isin(['가방','의류'])]
        df_ex = df_ex[df_ex.지식베이스.str.contains('품|사이즈')]
        print(len(df_ex))
        df_em = df[df['CATEGORY']=='신발']
        df_em = df_em[df_em.지식베이스.str.contains('품|사이즈')]
        print(len(df_em))

        existing_dataset[intent]=df_ex['SENTENCE'].to_list()
        emerging_dataset['신발']=df_em['SENTENCE'].to_list()
    elif intent=='학원':
        df_ex = df[~pd.isnull(df['지식베이스'])]
        # df_ex = df[df['CATEGORY']!='피아노']
        print(len(df_ex))

        existing_dataset[intent]=df_ex['SENTENCE'].to_list()
        #emerging_dataset[intent]=df_em['SENTENCE'].to_list()
    elif intent=='소매점':
        df_ex = df[df['CATEGORY'].isin(['제과점','청과물'])]
        df_ex = df_ex[df_ex.지식베이스.str.contains('품|류')]
        print(len(df_ex))
        df_ex2 = df[df['CATEGORY']=='화장품']
        df_ex2 = df_ex2[df_ex2.지식베이스.str.contains('품|류')]
        print(len(df_ex2))

        existing_dataset[intent]=df_ex['SENTENCE'].to_list()
        existing_dataset['화장품']=df_ex2['SENTENCE'].to_list()

for existing_xlsx in existing_xlsx_list:
    parse_sentence_from_dataset(existing_xlsx, 3)
print(len(existing_dataset), len(emerging_dataset))

existing_text_file = open("./existing_e.txt", "w", encoding="utf-8")
emerging_text_file = open("./emerging_e.txt", "w", encoding="utf-8")

# collect existing intent and emerging intents

existing_text_list = []
emerging_text_list = []

for existing_intent in tqdm(existing_dataset):
    existing_contents = existing_dataset[existing_intent]
    for existing_content in tqdm(existing_contents):
        #processed_existing_content = " ".join(kkma.nouns(existing_content)).strip()
        try:
            processed_existing_content = " ".join(list(map(lambda x : x[0], kkma.pos(existing_content)))).strip()
        except:
            continue
        if processed_existing_content == "" or (len(processed_existing_content.split()) <= 3):
            continue
        existing_text_string = existing_intent+ "\t" + processed_existing_content + "\n"
        existing_text_list.append(existing_text_string)
        
for emerging_intent in tqdm(emerging_dataset):
    emerging_contents = emerging_dataset[emerging_intent]
    for emerging_content in tqdm(emerging_contents):
        
        #processed_emerging_content = " ".join(kkma.nouns(emerging_content)).strip()
        try:
            processed_emerging_content = " ".join(list(map(lambda x : x[0], kkma.pos(emerging_content)))).strip()
        except:
            continue
        if processed_emerging_content == "" or (len(processed_emerging_content.split()) <= 3):
            continue
        emerging_text_string = emerging_intent+ "\t" + processed_emerging_content + "\n"
        emerging_text_list.append(emerging_text_string)

import random

random.shuffle(existing_text_list)
random.shuffle(emerging_text_list)

for existing_text in existing_text_list:
    existing_text_file.write(existing_text)

for emerging_text in emerging_text_list:
    emerging_text_file.write(emerging_text)