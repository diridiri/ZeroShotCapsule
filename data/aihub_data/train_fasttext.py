import gensim
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model

"""
existing_f = open("existing_full.txt", "r")
emerging_f = open("emerging_full.txt", "r")

sentences = []
for l in existing_f.readlines():
    sentences.append(l.split(" "))

for l in emerging_f.readlines():
    sentences.append(l.split(" "))
"""
ko_txt = open("ko.txt", "r")
sentences = []
for l in ko_txt.readlines():
    sentences.append(l.split(" "))


model = load_facebook_model("./ko.bin")
print(model.wv.most_similar("립스틱"))
sent = [['lord', 'of', 'the', 'rings'], ['lord', 'of', 'the', 'semi-groups']]
model.build_vocab(sentences, update=True)
model.train(sentences=sentences, total_examples = len(sentences), epochs=100)

print(model.wv.most_similar("립스틱"))
model.save("./ko_aihub_100.bin")
"""

#ko_model = gensim.models.Word2Vec.load('./ko.vec')
#model = KeyedVectors.load_word2vec_format(
#            "./ko.bin", binary=True)
#print(model.wv.most_similar("강아지"))

ko_model = Word2Vec(size=200, min_count=54)
ko_model.build_vocab(sentences)
##ko_model.build_vocab([list(model.vocab.keys())], update=True)
ko_model.intersect_word2vec_format(fname="ko.bin", binary=True)
ko_model.train(sentences, total_examples=ko_model.corpus_count, epochs=ko_model.epochs)
#a = ko_model.wv.most_similar("강아지")

ko_model.save('./ko_new.vec') """