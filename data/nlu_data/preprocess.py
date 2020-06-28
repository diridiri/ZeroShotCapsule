
xintent= ['music', 'search', 'movie', 'weather', 'restaurant']
mintent = ['playlist', 'book']

def choose_intent(before):
    if before=='music':
        return 'play music'
    elif before=='search':
        return 'search creative work'
    elif before=='movie':
        return 'search screening event'
    elif before=='weather':
        return 'get weather'
    elif before=='restaurant':
        return 'book restaurant'
    elif before=='playlist':
        return 'add to playlist'
    else:
        return 'rate book'

with open('train_shuffle.txt') as f:
    l=f.readlines()
    with open('train_multi.txt','w') as ft:
        for line in l:
            pair=line.split('\t')
            multi=choose_intent(pair[0])
            ft.write('%s\t%s' % (multi, pair[1]))

with open('test.txt') as f:
    l=f.readlines()
    with open('test_multi.txt','w') as ft:
        for line in l:
            pair=line.split('\t')
            multi=choose_intent(pair[0])
            ft.write('%s\t%s' % (multi, pair[1]))
