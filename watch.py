import pickle

print('valid_seen')
data = pickle.load(open('result/alfred/prompt10/vanilla/valid_seen_1.p', 'rb'))
print(len(data))

print('valid_unseen')
data = pickle.load(open('result/alfred/prompt10/vanilla/valid_unseen_0.p', 'rb'))
print(len(data))