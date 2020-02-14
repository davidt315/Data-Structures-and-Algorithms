# David Tarazi
# Homework 2: DLLs

import pytest
import timeit
import random
import matplotlib.pyplot as plt

class Node:
    # Creates a single node in a DLL
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class Dll:
    # Contains the connections to all the nodes in the DLL as a wrapper
    def __init__(self):
        self.start = None
        self.length = 0


    def push(self, data):
        """
        Inserts a new node in the front of the DLL
        
        data = data for the new node
        """
        newNode = Node(data)

        if self.start is None:
            # If the list starts empty, assign the end to the new node
            self.start = newNode

        else:
            # Assign next to the start, the current start's prev to the new node, and make the new node the start
            newNode.next = self.start
            self.start.prev = newNode
            self.start = newNode
        
        self.length += 1


    def insert(self, data, location):
        """
        Inserts a node after a specific item

        location = the previous item to insert after
        data = the new node data
        """
        # make sure the list isn't empty
        if self.start is None:
            print("Nowhere to insert, DLL is empty")
            return

        # nod to iterate through and find the location
        nod = self.start
        for _ in range(self.length):
            if nod.data == location:
                # break when you reach the location
                break
            else:
                nod = nod.next

        # If the given item didn't exist
        if nod is None:
            print("Item location doesn't exist")
            return
        else:
            newNode = Node(data)
            newNode.prev = nod
            # If we are at the end
            if nod.next is not None:
                newNode.next = nod.next
                nod.next.prev = newNode
            nod.next = newNode
        self.length += 1


    def delete(self, location):
        """
        Deletes a given node from the DLL

        location = node to delete
        """
        # Error handling
        # make sure the list isn't empty
        if self.start is None:
            print("Nowhere to delete, DLL is empty")
            return

        # if there is only one element    
        if self.start.next is None:
            if self.start.data == location:
                self.start = None
            else:
                print("Node not found")
                return 

        # if deleting the start
        if self.start.data == location:
            self.start = self.start.next
            self.start.prev = None
            return
        # finished error handling

        # nod to iterate through and find the location
        nod = self.start
        for _ in range(self.length):
            if nod.data == location:
                # break when you reach the location
                break
            else:
                nod = nod.next
        
        # node not found
        if nod is None:
            print("Node not found")
            return
        # reset end if it's the last one
        elif nod.next is None:
            nod.prev.next = None
        else:
            nod.next.prev = nod.prev
            nod.prev.next = nod.next
            
        self.length -= 1


    def getLen(self):
        """
        Returns the length of the list (number of nodes)
        """
        return self.length


    def index(self, index):
        """
        Returns the data stored at the list index

        index = index of desired node
        """
        # make sure the list isn't empty
        if self.start is None:
            print("Nowhere to index, DLL is empty")
            return
        
        # Traverse and return when you reach the index
        nod = self.start
        for i in range(self.length):
            if i == index:
                return nod.data
            else:
                nod = nod.next

        if i is None:
            print("Node not found")
            return


    def multiplyPairs(self):
        """
        Multiplies all unique pairs of nodes values for nodes i, j (i != j)         
        and returns the sum 
        """
        # make sure the list isn't empty
        if self.start is None or self.start.next is None:
            print("Nothing to multiply, DLL is empty or only has one element\n")
            return

        nod = self.start
        sum = 0
        for i in range(self.length-1):
            # first pair is first number and the next
            nod2 = nod.next
            for j in range(i+1, self.length):
                # if they're the same, don't multiply
                if (nod.data != nod2.data):
                    sum += nod.data * nod2.data
                # update the second pair element
                nod2 = nod2.next
            # update the first pair element
            nod = nod.next
        return sum



# TEST -------------------------------------------------------------------------------
dll = Dll()
dll.push(5)
dll.push(3)

def test_func():   
    # Test all functions

    # Test push()
    dll.push(2)
    assert dll.start.data == 2

    # Test insert()
    dll.insert(4,3)
    assert dll.index(2) == 4

    # Test delete()
    dll.delete(4)
    assert dll.index(2) == 5

    # Test getLen()
    assert dll.getLen() == 3

    # Test index(i)
    assert dll.index(0) == 2

    # Test multiplyPairs()
    assert dll.multiplyPairs() == 31


# QUESTION 2
def plot_all(timesDLL, timesList, timesPairs):
    """
    Plots the times for n ranging from 10 to 10000
    """
    plt.figure(figsize=(5, 2))

    plt.subplot(211)
    plt.plot(timesDLL)
    plt.xlabel('n (increments of 10)')
    plt.ylabel('time (seconds)')
    plt.title("DLL")

    plt.subplot(212)
    plt.plot(timesList)
    plt.xlabel('n (increments of 10)')
    plt.ylabel('time (seconds)')
    plt.title("Python List")

    plt.suptitle('DLL vs Lists index runtime')
    plt.show()

    plt.figure()
    plt.plot(timesPairs)
    plt.xlabel('n (increments of 10)')
    plt.ylabel('time (seconds)')
    plt.title("Multiply Pairs Function")
    plt.show()



n = 10
timesList = []
timesDLL = []
timesPairs = []
while n < 3001:
    dll = Dll()
    pylist = []
    for i in range(n):
        dll.push(i)
        pylist.append(i)

    t = timeit.Timer('dll.index(random.randrange(n))', 'import random', globals=locals())
    timesDLL.append(t.timeit(50)) # times the operation 50 times and averages

    t = timeit.Timer('pylist[random.randrange(n)]', 'import random', globals=locals())
    timesList.append(t.timeit(50)) # times the operation 50 times and averages

    t = timeit.Timer('dll.multiplyPairs()', 'import random', globals=locals())
    timesPairs.append(t.timeit(3)) # times the operation 10 times and averages
    
    if n%100 == 0:
        print(n)
    
    n += 10

plot_all(timesDLL, timesList, timesPairs)
 
