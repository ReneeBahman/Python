users = ['Dave', 'John', 'Renee']

data = ['Renee', 42, True]

emptylist = []

print("Dave" in users)
print("Dave" in data)

print("")
# prints first value from the list, note indexing starts with 0
print(users[0])
print(users[-1])  # prints last value from the list
print(users[-2])  # prints second to last value from the list
print("")
print(users.index("Renee"))

print(users[0:2])  # does not include 2nd position
print(users[1:])  # prints all from position of 1
print(users[0:])  # prints all members of the list
print(users[-3:-1])  # prints also with negative signs

print("")

print(len(users))  # prints how many items are in the list
print(len(data))  # prints how many items are in the list
print(len(emptylist))  # prints how many items are in the list

# adding

print("")
users.append('Elsa')
print(users)

users += ['Jason']
print(users)

users.extend(['Robert', 'Jimmy'])
print(users)

print("")
print("")
print("")

# users.extend(data)  # adding another list to existing list
# print(users)

users.insert(0, 'Bob')
print(users)

users[2:2] = ['Eddie', 'Alex']  # adding new values to the list with indexes
print(users)

print("")
print("")
print("")

users[1:3] = ['Robert', 'JPJ']  # replacing values in the list, aka slice
print(users)

print("")

# Deleting values

users.remove("Bob")
print(users)

print(users.pop())
print(users)

del users[0]
print(users)

# del data
data.clear()
print(data)

# sorting

users[1:2] = ['dave']  # lower case names come after upper case names
users.sort()
print(users)

users.sort(key=str.lower)
print(users)

nums = [4, 42, 78, 1, 5]
nums.reverse()
print(nums)

# nums.sort(reverse=True)
# print(nums)

# if we want to keep list in original order and sort in printing

print(sorted(nums, reverse=True))
print(nums)

print("")

# ways to copy
numscopy = nums.copy()
mynums = list(nums)
mycopy = nums[:]


print(numscopy)
print(mynums)
mycopy.sort()
print(mycopy)
print(nums)


print(type(nums))
print(type(numscopy))
print(type(mynums))
print(type(mycopy))

mylist = list([1, 'Dave', True])
print(mylist)

# Tuples - lists with arranged order

mytuple = (('Dave', 42, True))
anothertuple = (1, 3, 42, 55, 2, 2)  # process called "packing the tuple"

print(mytuple)
print(type(mytuple))
print(type(anothertuple))


newlist = list(mytuple)
newlist.append('Neil')

newtuple = tuple(newlist)
print(newtuple)

(one, two, *hey) = anothertuple
print(one)
print(two)
print(hey)

print("")
print(anothertuple)
print(anothertuple.count(2))


# additional training with GPT

mytuple = (10, 20, 30)
a, b, c = mytuple

print(a)  # 10
print(b)  # 20
print(c)  # 30

print("")
mytuple = (10, 20, 30, 40, 50)

a, b, *rest = mytuple

print(a)     # 10
print(b)     # 20
print(rest)  # [30, 40, 50]

# sample
mydata = ("Python", 3.13, "Renee", True, 2025)

lang, version, *details = mydata

print(lang)
print(version)
print(details)

print("")

mydata = ("Python", 3.13, "Renee", True, 2025)

lang, *middle, year = mydata

print(lang)
print(middle)
print(year)

print("")


tiny = (1, 2, 3)

a, *b, c = tiny

print(a)
print(b)
print(c)

# unpacking a string

print("")

language = "Python"

a, b, *middle, y, z = language

print(a)
print(b)
print(middle)
print(y)
print(z)

numbers = [100, 200, 300, 400, 500, 600, 700]

first, *middle, second_last, last = numbers

print(first)
print(middle)
print(second_last)
print(last)
