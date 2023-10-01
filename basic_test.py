# Created by Ansari at 02/2/2019

x1 = "Hello Hello How are you?"
x2 = "Hello Hello are you doing?"

print(len(x1))
print((len(x2)))

#character length
x3 = len(''.join(set(x1.replace(' ', '')))) #character-set no duplication
x4 = len(''.join(set(x2.replace(' ', '')))) #character-set no duplicate

print(x3)
print(x4)

numberOfWords_1 = len(x1.split())
numberOfWords_2 = len(x2.split())
print(numberOfWords_1)
print(numberOfWords_2)


common_words = len(set(x1.lower().split()).intersection(set(x2.lower().split())))
print(common_words)




