from PIL import Image

import numpy as np
from scipy.stats import norm
import math

from matplotlib import pyplot as plt
import networkx as nx
from collections import defaultdict


RED = (255, 0, 0)
BLUE = (0, 0, 255)

"""
The following function is modified from https://sandipanweb.wordpress.com/2018/02/11/interactive-image-segmentation-with-graph-cut/
to calculate the mean and standard deviation of the foreground and background in order to
fit the foreground and background to their own Normal distribution

Using the normal distributions on the RGB values, we can create a probability for each pixel in the image
"""
def compute_pdfs(rgb, rgb_s, yuv):
    '''
    Compute foreground and background pdfs
    
    '''
    # separately store background and foreground scribble pixels in the dictionary comps
    scrib = np.zeros(rgb.shape)
    comps = defaultdict(lambda:np.array([]).reshape(0,3))
    # finds the pixels that are different and creates two seperate dicts - one with the original
    # pixel colors for one scribble color and one for the other scribble color
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            # if the pixel is red
            if rgb_s[i,j,0] > 242 and rgb_s[i,j,1] < 12 and rgb_s[i,j,2] < 12:
                scrib[i,j,:] = [255,0,0]
                # scribble color as key of comps
                comps[RED] = np.vstack([comps[RED], yuv[i,j,:]])
            # if blue
            elif rgb_s[i,j,0] < 12 and rgb_s[i,j,1] < 12 and rgb_s[i,j,2] > 242:
                scrib[i,j,:] = [0,0,255]
                # scribble color as key of comps
                comps[BLUE] = np.vstack([comps[BLUE], yuv[i,j,:]])
    # compute MLE parameters for Gaussians
    mu, sigma = {}, {}

    mu["background"] = np.mean(comps[RED], axis=0)
    sigma["background"] = np.std(comps[RED], axis=0) 
    mu["foreground"] = np.mean(comps[BLUE], axis=0)
    sigma["foreground"] = np.std(comps[BLUE], axis=0)
    print("mean and standard deviation calculated, pdfs can be used") 
    return mu, sigma


def process_images(imfile, imfile_scrib):
    """
    turns images into np arrays 
    rgb = original picture
    rgb_s = picture with scribbles
    yuv = yiq colorscale of original image
    """
    rgb = np.asarray(Image.open(imfile))
    yuv = rgb2yiq(rgb)
    rgb_s = np.asarray(Image.open(imfile_scrib))
    print("images processed")
    return rgb, rgb_s, yuv

def rgb2yiq(arr):
    """
    Converts a numpy array from RGB to YIQ colorspace
    """
    yiq_from_rgb = np.array([[0.299,      0.587,        0.114],
                            [0.59590059, -0.27455667, -0.32134392],
                            [0.21153661, -0.52273617, 0.31119955]])
    return np.dot(arr, yiq_from_rgb.T.copy())


def create_graph(yuv, mu, sigma, load=False):
    """
    creates a graph with nodes as pixels labeled by their [i,j] position
    includes a node for the foreground and background

    allows for you to load from adjacency list file instead
    """
    if load:
        G = nx.read_adjlist("test.adjlist")
    else:
        G = nx.Graph()
        for i in range(yuv.shape[0]):
            for j in range(yuv.shape[1]):
                G.add_node((i, j)) 

        # background
        G.add_node((yuv.shape[0],0))
        # foreground
        G.add_node((yuv.shape[0]+1,0))

        connect_edges(G, yuv, mu, sigma)
    print("graph created")
    return G


def connect_edges(G, yuv, mu, sigma):
    """
    creates the edge weights between each pixel and the terminal nodes (back and fore)

    also connects each pixel to its neighbors
    """
    for i in range(yuv.shape[0]):
        print(str(i) + "/" + str(yuv.shape[0]) + " edge connection loops complete")
        for j in range(yuv.shape[1]):
            # terminal connections
            # connect to foreground by using mu and sigma from background scribbles
            G.add_edge((i,j), (yuv.shape[0]+1,0), capacity = compute_terminal_weight(yuv[i,j,:],mu["background"],sigma["background"]))
            # connect to background by using mu and sigma from foreground scribbles
            G.add_edge((i,j), (yuv.shape[0],0), capacity = compute_terminal_weight(yuv[i,j,:],mu["foreground"],sigma["foreground"]))

            # edge above the pixel
            if (i>0 and not G.has_edge((i-1,j),(i,j))):
                G.add_edge((i-1,j), (i,j), capacity = compute_pixel_weights(yuv[i-1,j,:], yuv[i,j,:]))
            # edge to the left
            if (j>0 and not G.has_edge((i,j-1),(i,j))):
                G.add_edge((i,j-1),(i,j), capacity = compute_pixel_weights(yuv[i,j-1,:], yuv[i,j,:]))
    print("image graph completed, saving to file")
    nx.write_adjlist(G,"test.adjlist")


def compute_terminal_weight(yiq,mu,sigma):
    """
    computes weight by finding the pdf given mean and std for each path in the yiq color
    returns the sum of these pdfs
    high probability for foreground = low weight to background
    low probability for foreground = high weight to background
    """
    # compute probabilities to terminal
    py = norm.pdf(yiq[0], mu[0], sigma[0])
    pi = norm.pdf(yiq[1], mu[1], sigma[1])
    pq = norm.pdf(yiq[2], mu[2], sigma[2])

    # compute weight to the other terminal
    wy = -math.log(py)
    wi = -math.log(pi) 
    wq = -math.log(pq)
    
    # return the sum of the three colorspace values weights
    return wy + wi + wq


def compute_pixel_weights(yiq1, yiq2):
    """
    computes the weights between two pixels
    """    
    # finds the difference between the two pixels yiq values
    wy = math.exp((-1/4)*(abs(yiq2[0]-yiq1[0]))**2)
    wi = math.exp((-1/4)*(abs(yiq2[1]-yiq1[1]))**2)
    wq = math.exp((-1/4)*(abs(yiq2[2]-yiq1[2]))**2)

    # return the sum of the three colorspace values weights
    return wy + wi + wq



# ==========================================================================
# ========================== MIN-CUT ALGORITHM =============================
# ==========================================================================

"""
Unfortunately, we weren't able to get our own min-cut to work properly, but here is
what we had before switching over to the premade algorithm
"""
def FF(G):
    """
    Given a graph, G and a source and sink (background and foreground),
    FF will use the Ford-Fulkerson to find the max flow of the graph = min cut

    rules:
    1. Weight = capacity and flow can't exceed capacity
    2. Incoming flow to a node = outgoing flow from that node

    sudocode:
    set the flow to 0
    while there exists a path from background to foreground not breaking rules:
        update the flow
        update the graph (residual graph)

    returns min cut path
    """
    # create index for background and foreground
    back = (G.number_of_nodes()-2,0)
    fore = (G.number_of_nodes()-1,0)

    # create empty path to start (will be filled by one iteration of BFS)
    # BFS will check all nodes (filling the path completely) and will update path
    path = [-1]*G.number_of_nodes()
    max_flow = 0

    # while a path exists between the foreground and background on the image
    while(find_path(G, path)):
        path_flow = float("inf")
        
        s = back
        # go until reaching the foreground
        while(s !=  fore):
            # take the min route to traverse the image
            path_flow = min(path_flow, G[path[(s[0])]][s]['weight'])
            s = path[s]

        # Add path flow to overall flow
        max_flow +=  path_flow

        # update residual capacities of the edges and reverse edges along path
        v = back
        while(v !=  fore):
            u = path[v]
            G[u][v] -= path_flow
            G[v][u] += path_flow
            v = path[v]

    return path

def find_path(G, path):
    """
    finds a path from the source to sink using BFS
    """
    # create index for background and foreground
    back = (G.number_of_nodes()-2,0)
    fore = (G.number_of_nodes()-1,0)

    # create dictionary of visited or unvisited based on city name
    visited = {}
    # make all pixels as unvisited
    for i in range(G.number_of_nodes()-2):
        visited[i] = False
    visited[fore] = False
    
    # Create a queue for BFS
    queue=[]

    # Mark the source node as visited and enqueue it
    queue.append(back)
    # background is starting node
    visited[back] = True

    while queue:

        # dequeue a node from queue
        u = queue.pop(0)

        # iterate through the pixels inc. fore and back
        for ind in range(len(G)):
            # check for weights
            if not visited[ind] and G[u,ind] > 0:
                queue.append(ind)
                visited[ind] = True
                path[ind] = u

    # If we reached sink in BFS starting from source return True
    return True if visited[fore] else False

def plot_segmented_images(reachable, unreachable, rgb):
    """
    takes a couple of sets from the segmentation and plots them
    """
    img_1 = np.zeros(rgb.shape)
    img_2 = np.zeros(rgb.shape)

    for pixel in reachable:
        i,j = pixel
        if(i<rgb.shape[0] and j<rgb.shape[1]):
            img_1[i,j,:] = rgb[i,j,:]

    for pixel in unreachable:
        i,j = pixel
        if(i<rgb.shape[0] and j<rgb.shape[1]):
            img_2[i,j,:] = rgb[i,j,:]

    img1_plot = Image.fromarray(img_1, 'RGB')
    img1_plot.save('seg1.png')
    img1_plot.show()

    img2_plot = Image.fromarray(img_2, 'RGB')
    img2_plot.save('seg2.png')
    img2_plot.show()


if __name__ == "__main__":
    rgb, rgb_s, yuv = process_images('elephant.jpg', 'elephant_scribbles.jpg')
    mu, sigma = compute_pdfs(rgb, rgb_s, yuv)
    G = create_graph(yuv, mu, sigma, load=False)
    cut_value, partition = nx.minimum_cut(G, (rgb.shape[0],0), (rgb.shape[0]+1,0))
    print(cut_value)
    reachable, unreachable = partition
    print(len(reachable), len(unreachable))
    plot_segmented_images(reachable, unreachable, rgb)
    
