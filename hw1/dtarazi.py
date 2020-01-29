def max_increasing(mylist):
    """
    Returns the sublist of input mylist with the most consecutive increasing ints

    mylist: input list of ints
    """
    max = 1     # max count
    count = 1   # current count
    index = 0   # start of max index
    for i in range(len(mylist) - 1):
        if (mylist[i+1] > mylist[i]):
            count += 1
            # updates the max increasing count and index
            if (count > max):
                max = count
                index = (i+2) - max
        else:
            count = 1   # resets count
    # for empty list, this is mylist[0:1] which is the input
    return mylist[index:index+max]

def test_func():
    # Test functions for the max_increasing function
    assert max_increasing([1,3,4,5,6,8]) == [1,3,4,5,6,8]
    assert max_increasing([]) == []
    assert max_increasing([35,23,4,2,1]) == [35]
    assert max_increasing([23,4,1,7,89]) == [1,7,89]
