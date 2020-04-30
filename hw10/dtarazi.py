import numpy as np
import time

def read_tsp(filename):
    ''' Reads a TSPLIB instance given by filename and returns the corresponding    
    distance matrix C. Assumes that the edge weights are given in lower diagonal row    
    form. '''    
    f = open(filename,'r')    
    n, C = 0, None    
    i, j = -1, 0    
    for line in f:        
        words = line.split()        
        if words[0] == 'DIMENSION:':            
            n = int(words[1])            
            C = np.zeros(shape=(n,n))        
        elif words[0] == 'EDGE_WEIGHT_SECTION':            
            i = 0 # Start indexing of edges        
        elif i >= 0:            
            for k in range(len(words)):                
                if words[k] == 'EOF':                    
                    break                
                elif words[k] == '0':                    
                    i += 1 # Continue to next row of the matrix                    
                    j = 0                
                else:                    
                    C[i,j] = int(words[k])                    
                    C[j,i] = int(words[k])                    
                    j += 1    
    return C
                                


def greedy_TSP(cities, C):
    """
    greedily finds an "optimal" path for the TSP
    cities = list of cities labled {1, 2,..., n}
    C = matrix for distances C[i][j]
    """

    # create dictionary of visited or unvisited based on city name
    visited = {}
    for city in cities:
        visited[city] = False
    
    path = [cities[0]]
    opt = 0
    counter = 0

    while counter < len(cities)-1:
        # mark the last added city as visited
        visited[path[-1]] = True
        min = -1
        label = -1
        for i in range(len(cities)):
            # set min to first unvisited city
            if min == -1 and not visited[i+1]:
                # set min to the distance in the matrix
                min = C[path[-1]-1, i]
                label = i+1
            # otherwise check other unvisited cities to see if they're closer to the min
            elif not visited[i+1] and C[path[-1]-1, i] < min:
                min = C[path[-1]-1, i]
                label = i+1
        # add closest city to the path
        path.append(label)
        # add the distance to path length
        opt += min
        counter += 1

    # visited all cities and appending the starting city to the path to complete tour
    path.append(path[0])
    opt += C[path[-2]-1, path[-1]-1]

    return path, opt



def reinsert(path, opt, C):
    """
    pops a random city out of the path and reinserts it in the best place
    """
    # temp_opt is the distance without the city
    temp_opt = opt
    # choose a random city 
    index = np.random.randint(0,len(path)-1)
    # update optimality as if city wasn't there
    temp_opt -= (C[path[index-1]-1, path[index]-1] + C[path[index]-1, path[index+1]-1])
    temp_opt += C[path[index-1]-1, path[index+1]-1]

    # remove the city from the path and remove the finish
    path.pop(-1)
    c = path.pop(index)

    # start by setting the minimum to the current optimal solution
    min = opt
    place = index
    # try inserting the city at each point and insert in the best place
    for i in range(len(path)):
        temp = temp_opt
        # update optimality as if inserted before index i
        temp -= C[path[i-1]-1, path[i]-1]
        temp += C[path[i-1]-1, c-1] + C[c-1, path[i]-1]
        if temp < min:
            min = temp
            place = i

    # reconstruct the path
    opt = min
    path.insert(place, c)
    path.append(path[0])

    return path, opt


# simple example I made to test
"""
cities = [1, 2, 3, 4, 5, 6]
C = np.array([  [0, 30, 40, 45, 55, 50],
                [30, 0, 40, 20, 25, 70],
                [40, 40, 0, 20, 10, 35],
                [45, 20, 20, 0, 5, 50],
                [55, 25, 10, 5, 0, 45],
                [50, 70, 35, 50, 45, 0]])

path, opt = greedy_TSP(cities, C)
print(path, opt)
print(reinsert(path, opt, C))
"""


def get_all_optimal(files, solutions, num_reinserts):
    """
    Finds the runtimes and optimality values for all the files
    returns a list of runtimes and a list of optimality values
    """
    times = []
    optimals = []
    opt_gaps = []
    for i in range(len(files)):
        C = read_tsp(files[i])
        cities = [j for j in range(1,len(C))]
        # start the timer
        seconds = time.time()
        path, opt = greedy_TSP(cities, C)
        for _ in range(num_reinserts):
            path, opt = reinsert(path, opt, C)
        # add runtime for the opperation and the optimal value output
        times.append(int((time.time()-seconds)*100000)/100000)
        optimals.append(int(opt))
        opt_gaps.append(int((optimals[i] - solutions[i])/solutions[i]*1000)/1000)
    
    return optimals, opt_gaps, times


files = ['gr17.tsp', 'gr21.tsp', 'gr24.tsp', 'gr48.tsp']
solutions = [2085, 2707, 1272, 5046]

optimals, opt_gaps, times = get_all_optimal(files, solutions, 10000)

print(optimals, opt_gaps, times)