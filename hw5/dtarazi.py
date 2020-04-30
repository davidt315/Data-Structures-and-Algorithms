# David Tarazi
# HW 5: MinHeap Implementation
# 2-28-20

import pytest
from hypothesis import given
import hypothesis.strategies as st

# Min Heap class
class Heap: 
    def __init__(self, elements = None): 
        # initializes a heap given a list (default is empty)
        if elements == None:
            self.elements = []
        self.elements = elements.copy()
        self.len = len(elements)

        # construct a min heap in O(n) time 
        self.build_heap()
        # use copy when sorting
        self.copy = []
    
    def build_heap(self):
        # builds a heap in O(n) time
        # go in reverse from the first non-leaf element to the node
        # if there's one or less elements, this will not actually try to sort, but by going to -1, you still run if there are 2 or 3 elements
        for i in range((int)(self.len/2)-1, -1, -1):
            # gives starting location and swap_elements_down will 
            self.swap_elements_down(i)

    def swap_elements_down(self, index):
        # can recursively swap elements down the tree 
        # down the tree meaning compare a parent to both children and then swap if necessary
        # define left and right children of the node (as index)
        left_i = 2*index+1
        right_i = 2*index+2
        # define current node to be smallest, then check if the children are smaller (as index)
        smallest_i = index
        # don't try to check if the index is greater than or equal to the number of elements
        if (right_i < self.len and self.elements[right_i] < self.elements[smallest_i]):
            smallest_i = right_i
        if (left_i < self.len and self.elements[left_i] < self.elements[smallest_i]):
            smallest_i = left_i

        # recursion base case: current node is smaller than its children
        if smallest_i == index:
            return
        else:
            # otherwise, swap the smallest child up the tree and do the same operation with the place of the larger child
            temp = self.elements[index]
            self.elements[index] = self.elements[smallest_i]
            self.elements[smallest_i] = temp
            self.swap_elements_down(smallest_i)


    def length(self):
        # returns the length of the heap
        return self.len

    def insert(self, val):
        # inserts element (val) into the heap in O(logn) time
        # add the new element to the end of the list and increase the length
        self.elements.append(val)
        self.len += 1
        # if this was the first element added, don't try to sort
        if self.len == 1:
            return
        self.swap_elements_up(self.len-1)


    def swap_elements_up(self, index):
        # recursively swap elements going up if they are smaller
        # going up meaning compare the child to its parent and then swap if smaller

        # if it reached the min and is trying to go further, break
        if index <= 0:
            return

        # int truncation will find the right parent index
        parent_i = (int)((index-1)/2)
        # find the parent and compare working upwards
        # base case: parent is less than or equal to the current element
        if (self.elements[parent_i] <= self.elements[index]):
            return
        else:
            temp = self.elements[parent_i]
            self.elements[parent_i] = self.elements[index] 
            self.elements[index] = temp
            self.swap_elements_up(parent_i)

    
    def find_min(self):
        # returns the min (root) from the heap
        return self.elements[0]
    

    def delete_min(self):
        # deletes the min (root) from the heap 
        # handle 0 and 1 length cases
        if self.len == 0:
            print ("Error: can't remove an element because there are no elements")
            return
        elif self.len == 1:
            self.len -= 1
            return self.elements.pop(0)

        out = self.elements[0]
        self.elements[0] = self.elements.pop(self.len-1)
        self.len -= 1
        self.swap_elements_down(0)
        return out


    def sorted_list(self):
        # returns a sorted list of all the elements in the heap
        # error handling for length of 0
        if self.len == 0:
            return []
        if self.len == 1:
            return self.elements
        # temporary storage so that I can mutate the elements arraylist and the replace it later
        copy = self.elements.copy()
        length = self.len

        sorted = []
        for _ in range(self.len):
            sorted.append(self.delete_min())

        # reset the main elements from the copy
        self.elements = list(copy)
        self.len = length
        return sorted



@given(st.lists(st.integers()))
def test_heap_len(l):
    h = Heap(l)
    assert len(l) == h.length()

@given(st.lists(st.integers()))
def test_heap_insert_delete(l):
    # initialize empty heap and then insert every element of the list, l
    h = Heap()
    # create a sorted copy of the list
    sorted_list = l.copy()
    sorted_list.sort()
    # list for the minimum being deleted
    popped_list = []

    for i in l:
        h.insert(i)
    for _ in range(len(l)):
        # add the minimum to the popped_list
        popped_list.append(h.delete_min())
    
    # the popped list should equal the sorted list
    assert sorted_list == popped_list

@given(st.lists(st.integers()))
def test_heap_init_delete(l):
    # initialize heap with list l
    h = Heap(l)
    # create a sorted copy of the list
    sorted_list = l.copy()
    sorted_list.sort()
    # list for the minimum being deleted
    popped_list = []

    for _ in range(len(l)):
        # add the minimum to the popped_list
        popped_list.append(h.delete_min())
    
    # the popped list should equal the sorted list
    assert sorted_list == popped_list

@given(st.lists(st.integers()))
def test_sorted(l):
    h = Heap(l)
    l.sort()
    assert l == h.sorted_list()



# if __name__ == "__main__":
#     a = Heap([0, 0])
#     sorted = a.sorted_list()
#     print (sorted)
#     print (a.elements)