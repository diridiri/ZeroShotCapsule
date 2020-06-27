with open('existing_khaiii_.txt') as f:
    with open('existing_khaiii_single.txt','w') as wf:
        for line in f.readlines():
            if line.split('\t')[0]=='떡 집':
                wf.write('떡\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='과일 채소 판매':
                wf.write('농수산물\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='정육점':
            #     wf.write('정육\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='청과물':
            #     wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
            else:
                wf.write(line)
'''
ex_intent = ['음식점','의류','학원','떡 집','제과','정육','과일 채소 판매','화장품','미용실','약국','숙박']
em_intent = ['음식 배달','의류 색상','방 예약']

ex_intent = ['음식점','의류','학원','떡','제과','정육','농수산','화장품','미용실','약국','숙박']
em_intent = ['배달','색상','예약']
'''
with open('emerging_khaiii_.txt') as f:
    with open('emerging_khaiii_single.txt','w') as wf:
        for line in f.readlines():
            if line.split('\t')[0]=='음식 배달':
                wf.write('배달\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='의류 색상':
                wf.write('색상\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='방 예약':
                wf.write('예약\t%s' % line.split('\t')[1])
            # elif line.split('\t')[0]=='청과물':
            #     wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
            else:
                wf.write(line)