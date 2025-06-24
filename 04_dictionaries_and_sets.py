# Dictionaries
band = {
    "vocals": "Plant",
    "guitar": "Page"
}

band2 = dict(vocals="Plant", guitar="Page")
band2 = dict(vocals="Plant", guitar="Page")

print(band)
print(band2)
print(type(band))
print(len(band))

# access items
print(band["vocals"])
print(band.get("guitar"))

# list all keys and values

print("")
print(band.keys())
print(band.values())

# list of key/value pairs as tuples

print(band.items())

# verify key exists
print("guitar" in band)
print("cowbell" in band)

# change values
band["vocals"] = "Coverdale"
band.update({"bass": "JPJ"})

print(band)

# remove items
print(band.pop("bass"))
print(band)

band["drums"] = "Bonham"
print(band)

print(band.popitem())
print(band)

# delete and clear
print("")

band["drums"] = "Bonham"
del band["drums"]
print(band)

band2.clear()
print(band2)

# del band2
# print(band2)

# copying dictionaries

# band2 = band  # creates a reference not a copy, both refer to the same place in memory, so all the things in band will affect the things in band2
# print("Bad copy!")

# print(band)
# print(band2)

# band2["drums"] = 'Dave'
# print(band)
print("")

# proper way to create copy

band2 = band.copy()
band2["drums"] = 'Dave'
print("Good copy!")
print(band)
print(band2)

# or use dict() construction function

band3 = dict(band)
print("Good copy!")
print(band3)

# nested dictionaries
member1 = {
    "name": "Plant",
    "instrument": "vocals"
}
member2 = {
    "name": "Page",
    "instrument": "guitar"
}
band = {
    "member1": member1,
    "member2": member2
}

print(type(band))
print(band)
print(band["member1"]["name"])  # level deeper

# sets

nums = {1, 2, 3, 4}

nums2 = set((1, 2, 3, 4))

print(nums)
print(nums2)
print(type(nums2))
print(len(nums2))

# no tuplicates are allowed

nums = {1, 2, 2, 3}
print(nums)
print(len(nums))

# True is a dupe of 1, False is a dupe of zero

print("")
nums = {1, True, 2, False, 3, 4, 0}
print(nums)

# check if value is in a set

print(2 in nums)
print(23 in nums)

# but you cannot refer to an element in the set with an index position or a key, that would not work

# add a new element ot a set

nums.add(8)
print(nums)

# add elements from one set to another
morenums = {5, 6, 7}
nums.update(morenums)

print(morenums)
print(nums)

# you can use update with lists, tuples adn dictionaries, too.
# merge sets to create new sets

one = {1, 2, 3}
two = {5, 6, 7}

mynewset = one.union(two)

print(mynewset)
print(one)
print(two)


# keep only the duplicates

one = {1, 2, 3}
two = {2, 3, 4}

# updates one to only include intersecting duplicates
one.intersection_update(two)
print(one)


# keep everything except the duplicates

one = {1, 2, 3}
two = {2, 3, 4}

# updates one to have everything excelt tuplicates - a oppisite of upper sample
one.symmetric_difference_update(two)
print(one)
