import pandas as pd
import glob
from konlpy.tag import Kkma, Okt
from konlpy.utils import pprint
from tqdm import tqdm
from hanspell import spell_checker
from khaiii import KhaiiiApi

# api=KhaiiiApi()
kkma = Kkma()
# okt=Okt()

existing_xlsx_list = glob.glob("./existing/*.xlsx")
emerging_xlsx_list = glob.glob("./emerging/*.xlsx")
print(existing_xlsx_list)

# total_dataset={}
existing_dataset = {}
emerging_dataset = {}
ex_label_list=[]
em_label_list=[]


def parse_sentence_from_dataset(dataset_path):
    intent = dataset_path.split(" ")[1].split("(")[0]
    dataset=pd.read_excel(dataset_path)
    df = dataset[["CATEGORY","SENTENCE","MAIN","지식베이스"]].dropna(subset=["CATEGORY","SENTENCE","MAIN"])
    #A 3
    if intent=='음식점':
        df0=df[df.CATEGORY.str.contains('배달음식점')]['SENTENCE'].to_list()[:2000]
        df1=df[df.CATEGORY.str.contains('셀프서비스')]['SENTENCE'].to_list()[:2000]
        df2=df[(df.CATEGORY.str.contains('홀서빙음식점'))&(df.지식베이스.str.contains('메뉴'))]['SENTENCE'].to_list()[:1000] #

        namelist=['배달 음식점','셀프 서비스']
        dflist=[df0, df1]

        for idx, dframe in enumerate(dflist):
            name=namelist[idx]
            existing_dataset[name]=dframe

        name='홀 서빙 음식점'
        emerging_dataset[name]=df2
    #B 4
    elif intent=='의류':
        df0=df[df.CATEGORY.str.contains('가방')]['SENTENCE'].to_list()[:2000]
        df1=df[df.CATEGORY.str.contains('신발')]['SENTENCE'].to_list()[:2000]
        df2=df[df.CATEGORY.str.contains('액세서리')]['SENTENCE'].to_list()[:2000]
        df3=df[(df.CATEGORY.str.contains('의류'))&(df.지식베이스.str.contains('/'))]['SENTENCE'].to_list()[:1000] #
        
        namelist=['가방','신발','액세서리']
        dflist=[df0, df1, df2]

        for idx, dframe in enumerate(dflist):
            name=namelist[idx]
            existing_dataset[name]=dframe

        name='의류'
        emerging_dataset[name]=df3

    #C 1
    elif intent=='학원':
        df=df['SENTENCE'].to_list()[:2000]
        name='학원'
        existing_dataset[name]=df
    #D 5
    elif intent=='소매점':
        df0=df[df.CATEGORY.str.contains('떡집')]['SENTENCE'].to_list()[:2000]
        df1=df[df.CATEGORY.str.contains('제과점')]['SENTENCE'].to_list()[:2000]
        # df2=df[df.CATEGORY.str.contains('반찬가게')].loc[1:1500]
        df2=df[df.CATEGORY.str.contains('정육점')]['SENTENCE'].to_list()[:2000]
        df3=df[df.CATEGORY.str.contains('청과물')]['SENTENCE'].to_list()[:2000]
        df4=df[(df.CATEGORY.str.contains('화장품'))&(df.지식베이스.str.contains('/'))]['SENTENCE'].to_list()[:1000] #
        # df0=df[(df.CATEGORY.str.contains('떡집'))&(df.지식베이스.str.contains('떡|류'))]
        # df1=df[(df.CATEGORY.str.contains('제과점'))&(df.지식베이스.str.contains('류'))]
        # #df2=df[(df.CATEGORY.str.contains('반찬가게'))&(df.지식베이스.str.contains('반찬|류'))]
        # df3=df[(df.CATEGORY.str.contains('정육점'))&(df.지식베이스.str.contains('고기|류'))]
        # df4=df[(df.CATEGORY.str.contains('청과물'))&(df.지식베이스.str.contains('류'))]
        # df5=df[(df.CATEGORY.str.contains('화장품'))&(df.지식베이스.str.contains('류'))]

        namelist=['떡','제과','정육','농수산물']
        dflist=[df0, df1, df2, df3]

        for idx, dframe in enumerate(dflist):
            name=namelist[idx]
            existing_dataset[name]=dframe

        name='화장품'
        emerging_dataset[name]=df4
    #E 2
    elif intent=='생활서비스':
        # df0=df[df.CATEGORY.str.contains('독서실')]
        df0=df[df.CATEGORY.str.contains('미용실')]['SENTENCE'].to_list()[:2000]
        # df2=df[df.CATEGORY.str.contains('세탁소')]
        df1=df[(df.CATEGORY.str.contains('약국'))&(df.지식베이스.str.contains('/'))]['SENTENCE'].to_list()[:1000] #

        name='미용실'
        existing_dataset[name]=df0

        name='약국'#
        emerging_dataset[name]=df1
        # df4=df[df.CATEGORY.str.contains('옷수선')]
        # df0=df[(df.CATEGORY.str.contains('미용실'))&(df.지식베이스.str.contains('시술|펌|염색|커트|머리카락|스타일|헤어'))]
        # df1=df[(df.CATEGORY.str.contains('약국'))&(df.지식베이스.str.contains('약|증상|감기|성분|류'))]

        # dflist=[df0, df1]
        # namelist=['미용실 스타일','약국 의약품']
        # namelist=morphlist(namelist)
        # # total_label_list.append(namelist)
        # for idx, dframe in tqdm(enumerate(dflist)):
        #     ex_label_list.append(namelist[idx])
        #     existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]
    #F 1
    elif intent=='카페':
        df=df['SENTENCE'].to_list()[:2000]
        name='카페'
        existing_dataset[name]=df
    #G 2
    elif intent=='숙박업':
        # df0=df[df.CATEGORY.str.contains('모텔여관')]
        df0=df[df.CATEGORY.str.contains('펜션캠핑장')]['SENTENCE'].to_list()[:2000]
        df1=df[(df.CATEGORY.str.contains('호텔'))&(df.지식베이스.str.contains('/'))]['SENTENCE'].to_list()[:1000] #

        name='캠핑'
        existing_dataset[name]=df0

        name='호텔'#
        emerging_dataset[name]=df1

    #H 2
    elif intent=='관광여가오락':
        df0=df[df.CATEGORY.str.contains('당구장')]['SENTENCE'].to_list()[:2000]
        #df1=df[(df.CATEGORY.str.contains('pc방'))&(df.지식베이스.str.contains('/'))]['SENTENCE'].to_list()[:1800] #

        name="당구"
        existing_dataset[name]=df0

        #name=" ".join(list(map(lambda x : x[0], kkma.pos('피시 방')))).strip()
        #emerging_dataset[name]=df1


for existing_xlsx in existing_xlsx_list:
    parse_sentence_from_dataset(existing_xlsx)

# print(total_label_list)
print("EXISTING LABEL:", ex_label_list)
for key in existing_dataset:
    print(key, len(existing_dataset[key]))
print("EMERGING LABEL:", em_label_list)
for key in emerging_dataset:
    print(key, len(emerging_dataset[key]))


existing_text_file = open("./existing_kor_single_final_r.txt", "w", encoding="utf-8")
emerging_text_file = open("./emerging_kor_single_final_r.txt", "w", encoding="utf-8")

# collect existing intent and emerging intents

existing_text_list = []
emerging_text_list = []

for existing_intent in tqdm(existing_dataset):
    existing_contents = existing_dataset[existing_intent]
    for existing_content in tqdm(existing_contents):
        try:
            processed_existing_content = " ".join(list(map(lambda x : x[0], kkma.pos(existing_content)))).strip()
        except:
            continue
        if processed_existing_content == "":
            continue
        if len(processed_existing_content.split()) >= 5 and len(processed_existing_content.split()) <= 15:
            existing_text_string = existing_intent+ "\t" + processed_existing_content + "\n"
            existing_text_list.append(existing_text_string)
        
for emerging_intent in tqdm(emerging_dataset):
    emerging_contents = emerging_dataset[emerging_intent]
    for emerging_content in tqdm(emerging_contents):
        
        try:
            processed_emerging_content = " ".join(list(map(lambda x : x[0], kkma.pos(emerging_content)))).strip()
        except:
            continue
        if processed_emerging_content == "":
            continue
        if len(processed_emerging_content.split()) >= 5 and len(processed_emerging_content.split()) <= 15:
            emerging_text_string = emerging_intent+ "\t" + processed_emerging_content + "\n"
            emerging_text_list.append(emerging_text_string)

import random

random.shuffle(existing_text_list)
random.shuffle(emerging_text_list)

for existing_text in existing_text_list:
    existing_text_file.write(existing_text)

for emerging_text in emerging_text_list:
    emerging_text_file.write(emerging_text)