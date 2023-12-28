print("wori")
import json

#read

with open('document.json','r',encoding='utf8') as fr:
    json_data = json.load(fr)
    
    print('AAAAAAAAAAAAAAAAA',json_data[1])
    print('BBBBBBBBBBBBBBBBB', type(json_data))
    
#write 

dict1 = {'name': 'sunwukong'}
dict2 = {'name': 'shaseng'}
with open('null.json','w',encoding='utf8') as to:
    #json_data = json.load(to)
    
    json.dump(dict1,to,ensure_ascii=False)
    json.dump(dict2,to,ensure_ascii=False)


dict1 = {'a':100, 'b':42, 'c':9}
list3 = list(dict1)
print(list3)

list = []

list.append(dict1)
print(list)