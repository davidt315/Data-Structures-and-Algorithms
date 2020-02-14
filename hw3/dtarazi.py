# David Tarazi
# Homework 3

import pytest

class Queue:
    def __init__(self):
        # elements contains the elements, min contains the current min
        self.elements = []
        self.min = None
    
    def enqueue(self, item):
        # inserts an item at the front of the list and keeps min in tact
        if self.min is None or item < self.min:
            self.min = item
        self.elements.insert(0, item)
    
    def dequeue(self):
        # dequeues an element from the list and keeps the min in tact
        if len(self.elements) == 0:
            print("Queue is empty, can't remove an element.")
            return

        if self.elements[-1] == self.min:
            # Reset min if the queue becomes empty
            if len(self.elements) == 1:
                self.min == None
                return self.elements.pop()
            # Find the min again
            else:
                val = self.elements.pop()
                self.min = min(self.elements)       # O(n)
                return val
        else:
            return self.elements.pop()

    def get_min(self):
        # returns min value
        if len(self.elements) == 0:
            print("Queue is empty, there is no min")
            return
        return self.min

# If I were to implement O(1) runtime for getting the min, I would've created a queue using two stacks 
# and included a third stack that holds the minimum. This isn't perfect though because you still have
# to run dequeue in O(n) runtime. 

def test_func():   
    # Test all functions
    q = Queue()
    q.enqueue(3)
    q.enqueue(4)

    #  Test enqueue()
    q.enqueue(2)
    assert q.elements[0] == 2

    # Test dequeue
    assert q.dequeue() == 3

    # Test get_min()
    assert q.get_min() == 2

