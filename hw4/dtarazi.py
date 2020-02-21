import math
import numpy.random as random


# Implementing Exercise 1's food finding algorithm

max_sum = 0
best_list = []

def get_max(foods, max_sum, best_list):
    # returns the sublist of which foods to get based on how much happiness they provide

    if (len(foods) == 1):
        return best_list
    
    sum = 0
    for i in range(1, len(foods)+1):
        sum += foods[-i]
        if sum > max_sum:
            max_sum = sum
            best_list = foods[-i:]

    # cut out the last element and recurse
    return get_max(foods[0:len(foods)-1], max_sum, best_list)


def test_finding_foods():
    # tests the function to find which foods to get

    my_foods = [-6, 3, 7, -3, 2, -5]
    assert get_max(my_foods, max_sum, best_list) == [3, 7]

    my_foods = [-2, -3, -6]
    assert get_max(my_foods, max_sum, best_list) == []


# Generate 100 random instances of happiness value lists of length n = 100 where
# each value hi is drawn uniformly between -10 and 10. This represents a situation in which
# you have a wide and evenly distributed set of preferences. Record the average length and
# value of the max interval returned by your function. Comment on the results

# create a list containing 100 lists of 100 random elements (between -10 and 10) each
n = 100
happy_lists = [ [] for i in range(n) ]
for i in range(n):
    for j in range(n):
        happy_lists[i].append(random.randint(-10,11))

# find average value and length over 100 lists of 100 food items
def get_avgs(happy_lists):
    avg_len = 0
    avg_val = 0
    for i in range(n):
        max_sum = 0
        best_list = []
        best_list = get_max(happy_lists[i], max_sum, best_list)
        avg_len += len(best_list)
        for val in best_list:
            avg_val += val

    avg_len = (int)(avg_len/n)
    avg_val = (int)(avg_val/n)
    return avg_len, avg_val

print(get_avgs(happy_lists))
# The average length was mostly in the low-mid 40s while the average value was mostly in the high 60s

# Using the normal distribution:
n = 100
happy_lists = [ [] for i in range(n) ]
for i in range(n):
    for j in range(n):
        # probability of .3 (1,2,3 / 10)
        if (random.randint(1,11) <= 3):
            happy_lists[i].append(random.normal(-7, 0.5))
        else:
            happy_lists[i].append(random.normal(6,1))

print(get_avgs(happy_lists))

# With this adjustment, it is more likely for you to like the food and as a result,
# the average length is around 90 foods, with a happiness of about 215-230.
# You are much happier with the adjustment.