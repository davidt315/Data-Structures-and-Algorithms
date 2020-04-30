import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def find_num_components(G):
    """
    returns the number of components given a graph g using depth-first search
    """
    # no nodes in graph
    if len(G.nodes) == 0:
        return 0

    # create a list of unvisited nodes and dict for lookup
    unvisited_list = list(G.nodes)
    unvisited_dict = dict(G.nodes)
    components = 0
    # iterate through graph
    while (len(unvisited_list) > 0):
        # create starting point for DFS (unprocessed is like the stack)
        unprocessed = []
        unprocessed.append(unvisited_list[0])
        # keep unvisited updated
        unvisited_dict.pop(unvisited_list[0])
        unvisited_list.pop(0)
        # iterate through component
        while (len(unprocessed) > 0):
            # focus on most recently added vertex from unprocessed component verticies and remove it from list
            u = unprocessed.pop(len(unprocessed)-1)
            neighbors = list(G.neighbors(u))
            for i in range(len(neighbors)):
                if unvisited_dict.get(neighbors[i]) is not None:
                    # if the neighbor exists in unvisited dictionary, add to processing stack and remove from unvisited dict and list
                    unprocessed.append(neighbors[i])
                    unvisited_dict.pop(neighbors[i])
                    unvisited_list.remove(neighbors[i])
                    
        # once making it through the component, update counter and loop back if there are more unvisited nodes
        components += 1

    return components


# ============================================================================================================
# Test Functions
# ============================================================================================================
def test_components():
    # tests the functionality of find_num_components(G)
    G = nx.Graph()
    G.add_nodes_from([1,2,3,5,7])
    G.add_edges_from([(1,2), (1,5), (3,7)])
    
    assert find_num_components(G) == 2


    G.clear()
    G.add_nodes_from([1,2,3,5,7])
    G.add_edges_from([(1,2), (1,5), (3,7), (3,5)])

    assert find_num_components(G) == 1

# ============================================================================================================
# ============================================================================================================

def random_binomial_graph(n, p):
    # the function produces 10 random binomial graphs given a probability, p, and a max, n (exclusive).
    # returns True if there is 1 component for every graph, False if not
    G = nx.Graph()
    for _ in range(10):
        G.clear()
        G.add_nodes_from(list(range(5,n)))
        # fill the edges based on probability
        for j in range(n):
            for k in range(j+1, n):
                if (np.random.uniform(0,1) < p):
                    G.add_edge(j,k)
        # check if theres only one component
        components = find_num_components(G)
        if (components != 1):
            return False
    return True


def plot_probability(l):
    # takes in a list, l, which contains the min probability to reach optimal 
    plt.plot(range(5,len(l)+5),l)
    plt.xlabel("Number of Verticies")
    plt.ylabel("Smallest Probability for 1 Component")
    plt.title("Smallest Probability for a 1 Component Random Binomial Graph")
    plt.show()


def find_probabilities():
    l = []
    while len(l)+5 < 50:
        p = 1
        while random_binomial_graph(len(l)+6, p):
            p -= 0.01
        l.append(int(p*100)/100)
    return l

plot_probability(find_probabilities())