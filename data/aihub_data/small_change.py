'''with open('existing_khaiii.txt') as f:
    with open('existing_khaiii_.txt','w') as wf:
        for line in f.readlines():
            if line.split('\t')[0]=='떡집':
                wf.write('떡 집\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='제과점':
                wf.write('제과\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='정육점':
                wf.write('정육\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='청과물':
                wf.write('과일 채소 판매\t%s' % line.split('\t')[1])
            else:
                wf.write(line)'''

with open('emerging_khaiii.txt') as f:
    with open('emerging_khaiii_.txt','w') as wf:
        for line in f.readlines():
            if line.split('\t')[0]=='의류 색상 문의':
                wf.write('의류 색상\t%s' % line.split('\t')[1])
            elif line.split('\t')[0]=='음식 배달 문의':
                wf.write('음식 배달\t%s' % line.split('\t')[1])
            else:
                wf.write(line)