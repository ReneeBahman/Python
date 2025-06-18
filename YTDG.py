greeting = 'Hello World!'
print(greeting)

line1 = '********************'
line2 = '*                  *'
line3 = '*     WELCOME!     *'

line4 = 'adkljsf'


print('')  # see on kommentaar, et faili ette tuleb tühi rida
print(line1)  # see on on headeri rida
print(line2)  # seda saab taaskasutada
print(line3)
print(line2)
print(line1)

del line4

# print(line4)


a = 10
# print(a)
print("")

meaning = 42
# print(meaning)

if meaning > 10:
    print('Right on!')
else:
    print('Not today!')

# ternary operator
# print('Right on!!') if meaning > 10 else print('Not today!!')

# Data Types

# Strings

# Literal assignment

first = 'Renee'  # ctrl + ä kommenteerib asjad välja
last = 'Bahman'


# print(first + " " + last)
# print(type(first))
# print(type(first) == str)
# print(isinstance(first, str))

# Constructor function

# pizza = str("Pepperoni")
# print(type(pizza))
# print(type(pizza) == str)
# print(isinstance(pizza, str))


# Concatenation

fullname = first + " " + last
print(fullname)

fullname += "!"
print(fullname)

# Casting a number to a string

decade = str(1980)
print(type(decade))


statement = "I like rock music from the " + decade + "s!"
print(statement)

# Multiple lines

multiline = """
Hey, how are you? 


I was just checking in.                 

                    All good? 
                    
"""

print(multiline)

# Escaping special characters

sentence = 'I\'m back at work!\tHey!\n\nWhere\'s this at\\located?'
print(sentence)

# String Methods

print(first)
print(first.lower())
print(first.upper())
print(first)

print(multiline.title())  # muudab kõik sõnad suurte tähtedega nagu pealkirjas
print(multiline.replace("good", "ok"))
print(multiline)


print(len(multiline))
multiline += "                            "
multiline = "                        " + multiline
print(len(multiline))

print(len(multiline.strip()))
print(len(multiline.lstrip()))
print(len(multiline.rstrip()))
