# with open('existing_khaiii_single.txt') as f:
#     with open('existing_khaiii_multi.txt','w') as wf:
#         for line in f.readlines():
#             if line.split('\t')[0]=='음식점':
#                 wf.write('음식 메뉴\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='의류':
#                 wf.write('의류 제품\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='학원':
#                 wf.write('학원 학생\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='떡':
#                 wf.write('떡 집\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='제과':
#                 wf.write('제과 점\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='정육':
#                 wf.write('정육 점\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='농수산물':
#                 wf.write('농수산물 시장\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='화장품':
#                 wf.write('화장품 종류\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='미용실':
#                 wf.write('미용실 스타일\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='약국':
#                 wf.write('약국 의약품\t%s' % line.split('\t')[1])
#             elif line.split('\t')[0]=='숙박':
#                 wf.write('숙박 시설\t%s' % line.split('\t')[1])
#             # elif line.split('\t')[0]=='정육점':
#             #     wf.write('정육\t%s' % line.split('\t')[1])
#             # elif line.split('\t')[0]=='청과물':
#             #     wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
#             else:
#                 wf.write(line)

'''
ex_intent = ['음식 메뉴','의류 제품 및 사이즈','학원 수업','떡 집','제과 점 빵','정육 점 고기','농수산물 시장','화장품 종류','미용실 스타일','약국 의약품','숙박 시설']
em_intent = ['배달 주문','옷 색상','방 예약']

ex_intent = ['음식 메뉴','의류 제품','학원 학생','떡 집','제과 점','정육 점','농수산물 시장','화장품 종류','미용실 스타일','약국 의약품','숙박 시설']
em_intent = ['음식 배달','의류 색상','방 예약']

ex_intent = ['음식점','의류','학원','떡 집','제과','정육','과일 채소 판매','화장품','미용실','약국','숙박']
em_intent = ['음식 배달 문의','의류 색상 문의','방 예약']

ex_intent = ['음식점','의류','학원','떡','제과','정육','농수산물','화장품','미용실','약국','숙박']
em_intent = ['배달','색상','예약']
'''
'''
with open('existing_kkma_multi.txt') as f:
    with open('existing_kkma_multi_.txt','w') as wf:
        for line in f.readlines():
            # if line.split('\t')[0]=='':
            #     continue
            if line.split('\t')[0]=='의류 제품':
                wf.write('의류 제품 및 사이즈\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='학원 학생':
                wf.write('학원 수업\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='제과 점':
                wf.write('제과 점 빵\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='정육 점':
                wf.write('정육 점 고기\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='정육 점':
            #     wf.write('정육 점 고기\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='청과물':
            #     wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
            else:
                wf.write(line)
'''
with open('emerging_kkma_multi.txt') as f:
    with open('emerging_kkma_multi_.txt','w') as wf:
        for line in f.readlines():
            # if line.split('\t')[0]=='':
            #     continue
            if line.split('\t')[0]=='음식 배달':
                wf.write('배달 주문\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='의류 색상':
                wf.write('옷 색상\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='제과 점':
            #     wf.write('제과 점 빵\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='정육 점':
            #     wf.write('정육 점 고기\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='정육 점':
            #     wf.write('정육 점 고기\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='청과물':
            #     wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
            else:
                wf.write(line)
