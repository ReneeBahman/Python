import math
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

print("")

# Build a menu S

title = "menu".upper()
print(title.center(20, "="))
print("Coffee".ljust(16, ".") + "€1".rjust(4))
print("Muffin".ljust(16, ".") + "€2".rjust(4))
print("Cheescake".ljust(16, ".") + "€4".rjust(4))

print("")

# string index values
print(first[1])  # indexes start at 0
print(first[-1])
print(first[1:-1])
# range - from start to end, keeping empty after: brings all of the range
print(first[0:])

print("")

# some methods to return boolean data
print(first.startswith("R"))
print(first.endswith("Z"))
print("")

# Boolean values
myvalue = True
x = bool(False)
print(type(x))
print(isinstance(myvalue, bool))

# numberic data types

# integer

price = 100
best_price = int(80)
print(type(price))
print(isinstance(best_price, int))

# float - have decimals

gpa = 3.28  # literal assigment
y = float(1.14)
print(type(gpa))


# complex type

comp_calue = 5+3J
print(type(comp_calue))
print(comp_calue.real)
print(comp_calue.imag)
print("")

# Built in functions for numbers

print(abs(gpa))
print(abs(gpa * -1))
print(round(gpa))  # rouds to the nearest integer
print(round(gpa, 1))  # rounds to the nearest decimal point specified


print(math.pi)
print(math.sqrt(64))
print(math.ceil(gpa))
print(math.floor(gpa))


print("")

# Castinf a string to a number
zipcode = "10001"
zip_value = int(zipcode)
print(type(zip_value))

# Error if we attempt to cast inccorect data

# zip_value = int("New York") # Sample
