import json



import json

studentsList = []
print("Started Reading JSON file which contains multiple JSON document")
with open('logs.json') as f:
    for jsonObj in f:
        studentDict = json.loads(jsonObj)
        studentsList.append(studentDict)

print("Printing each JSON Decoded Object")
for student in studentsList:
    print(student)
    