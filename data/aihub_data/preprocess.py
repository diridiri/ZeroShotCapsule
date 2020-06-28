import pandas as pd
import glob
from konlpy.tag import Kkma, Okt
from konlpy.utils import pprint
from tqdm import tqdm
from hanspell import spell_checker
# from khaiii import KhaiiiApi

# api=KhaiiiApi()
kkma = Kkma()
okt=Okt()

existing_xlsx_list = glob.glob("./existing/*.xlsx")
emerging_xlsx_list = glob.glob("./emerging/*.xlsx")
print(existing_xlsx_list)

total_dataset={}
existing_dataset = {}
emerging_dataset = {}
ex_label_list=[]
em_label_list=[]
#total_label_list=[]

class KhaiiiExcept(Exception):
    pass

# def morphlist(target_list):
#     morphex=[]
#     for label in target_list:
#         temp=[]
#         for word in api.analyze(label):
#             for morph in word.morphs:
#                 temp.append(morph.lex)
#         morphex.append(" ".join(temp))
#     return morphex

def parse_sentence_from_dataset(dataset_path, max_len):
    intent = dataset_path.split(" ")[1].split("(")[0]
    dataset=pd.read_excel(dataset_path)
    df = dataset[["CATEGORY","SENTENCE","MAIN","개체명", "지식베이스"]].dropna()
    if intent=='음식점':
        df_order = df[df.MAIN.str.contains('배달')]
        df = df[df.MAIN.str.contains('배달')==False]
        df_food = df[(df.MAIN.str.contains('메뉴'))&(df.지식베이스.str.contains('메뉴'))]

        dflist=[df_order,df_food]
        namelist=['음식 배달 문의','음식점']
        #namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            if idx!=0:
                ex_label_list.append(namelist[idx])
                existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]
            else:
                em_label_list.append(namelist[idx])
                emerging_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]

    elif intent=='의류':
        df_color = df[df.지식베이스.str.contains('색')]
        df = df[df.지식베이스.str.contains('색')==False]
        df_cloth = df[df.지식베이스.str.contains('제품|사이즈')]

        dflist=[df_color, df_cloth]
        namelist=['의류 색상 문의', '의류']
        #namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            if idx!=0:
                ex_label_list.append(namelist[idx])
                existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]
            else:
                em_label_list.append(namelist[idx])
                emerging_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]

    elif intent=='학원':
        df= df[df.지식베이스.str.contains('카드|결제|금액|비율')==False]
        existing_dataset['학원']=df['SENTENCE'].to_list()

        dflist=[df]
        namelist=['학원']
        #namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            ex_label_list.append(namelist[idx])
            existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]
        
    elif intent=='소매점':
        df0=df[(df.CATEGORY.str.contains('떡집'))&(df.지식베이스.str.contains('떡|류'))]
        df1=df[(df.CATEGORY.str.contains('제과점'))&(df.지식베이스.str.contains('류'))]
        #df2=df[(df.CATEGORY.str.contains('반찬가게'))&(df.지식베이스.str.contains('반찬|류'))]
        df3=df[(df.CATEGORY.str.contains('정육점'))&(df.지식베이스.str.contains('고기|류'))]
        df4=df[(df.CATEGORY.str.contains('청과물'))&(df.지식베이스.str.contains('류'))]
        df5=df[(df.CATEGORY.str.contains('화장품'))&(df.지식베이스.str.contains('류'))]

        dflist=[df0, df1, df3, df4, df5]
        namelist=['떡집','제과점','정육점','청과물','화장품']
        # namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            ex_label_list.append(namelist[idx])
            existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]

    elif intent=='생활서비스':
        df0=df[(df.CATEGORY.str.contains('미용실'))&(df.지식베이스.str.contains('시술|펌|염색|커트|머리카락|스타일|헤어'))]
        df1=df[(df.CATEGORY.str.contains('약국'))&(df.지식베이스.str.contains('약|증상|감기|성분|류'))]

        dflist=[df0, df1]
        namelist=['미용실','약국']
        # namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            ex_label_list.append(namelist[idx])
            existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]

    # elif intent=='카페':
    #     df0=df[df.지식베이스.str.contains('메뉴|음료')]
    #     total_dataset['카페']=df0['SENTENCE'].to_list()

    elif intent=='숙박업':
        df_room=df[df.지식베이스.str.contains('방|룸|층|숙박|인원')]
        df=df[df.지식베이스.str.contains('방|룸|층|숙박|인원')==False]
        df_reserve=df[df.MAIN.str.contains('예약')]

        dflist=[df_reserve, df_room]
        namelist=['방 예약', '숙박']
        # namelist=morphlist(namelist)
        # total_label_list.append(namelist)
        for idx, dframe in tqdm(enumerate(dflist)):
            if idx!=0:
                ex_label_list.append(namelist[idx])
                existing_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]
            else:
                em_label_list.append(namelist[idx])
                emerging_dataset[namelist[idx]]=[item.as_dict()['checked'] for item in spell_checker.check(dframe['SENTENCE'].to_list())]

    # elif intent=='관광여가오락':
    #     df0=df[(df.CATEGORY.str.contains('당구장'))&(df.지식베이스.str.contains('실력|구'))]
    #     existing_dataset['당구장']=df0['SENTENCE'].to_list()
        #df1=df[(df.CATEGORY.str.contains('PC방'))&(df.지식베이스.str.contains('이용'))]
    #     df_ex = df[df['CATEGORY'].isin(['제과점','청과물'])]
    #     df_ex = df_ex[df_ex.지식베이스.str.contains('품|류')]
    #     df_ex2 = df[df['CATEGORY']=='화장품']
    #     df_ex2 = df_ex2[df_ex2.지식베이스.str.contains('품|류')]
    #     existing_dataset[intent]=[item.as_dict()['checked'] for item in spell_checker.check(df_ex['SENTENCE'].to_list())] 
    #     existing_dataset['화장품']=[item.as_dict()['checked'] for item in spell_checker.check(df_ex2['SENTENCE'].to_list())] 

for existing_xlsx in existing_xlsx_list:
    parse_sentence_from_dataset(existing_xlsx, 3)

# print(total_label_list)
# print("EXISTING LABEL:", ex_label_list)
# for key in existing_dataset:
#     print(key, len(existing_dataset[key]))
# print("EMERGING LABEL:", em_label_list)
# for key in emerging_dataset:
#     print(key, len(emerging_dataset[key]))


existing_text_file = open("./existing_full.txt", "w", encoding="utf-8")
emerging_text_file = open("./emerging_full.txt", "w", encoding="utf-8")

# collect existing intent and emerging intents

existing_text_list = []
emerging_text_list = []

for existing_intent in tqdm(existing_dataset):
    existing_contents = existing_dataset[existing_intent]
    for existing_content in tqdm(existing_contents):
        try:
            # processed_existing_content = " ".join(list(map(lambda x : x[0], okt.pos(existing_content)))).strip()
            # processed_existing_content = " ".join(list(map(lambda x : x[0], kkma.pos(existing_content)))).strip()
            """sentencelist=[]
            for word in api.analyze(existing_content):
                for morph in word.morphs:
                    sentencelist.append(morph.lex)
            processed_existing_content = " ".join(sentencelist).strip()"""
            processed_existing_content = existing_content
        except:
            continue
        if processed_existing_content == "" or (len(processed_existing_content.split()) <= 5):
            continue
        existing_text_string = existing_intent+ "\t" + processed_existing_content + "\n"
        existing_text_list.append(existing_text_string)
        
for emerging_intent in tqdm(emerging_dataset):
    emerging_contents = emerging_dataset[emerging_intent]
    for emerging_content in tqdm(emerging_contents):
        
        try:
            """sentencelist=[]
            for word in api.analyze(emerging_content):
                for morph in word.morphs:
                    sentencelist.append(morph.lex)
            processed_emerging_content = " ".join(sentencelist).strip()"""
            processed_emerging_content = emerging_content
        except:
            continue
        if processed_emerging_content == "" or (len(processed_emerging_content.split()) <= 5):
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