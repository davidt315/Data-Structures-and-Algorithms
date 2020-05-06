from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# def process_image(im_file):
#     image_file = Image.open(im_file)
#     size = image_file.size
#     image_file = image_file.convert('L') # convert image to black and white
#     image_file.save('result.png')
#     ar = np.array(image_file)
#     foreground_crop_rectangle = (180, 180, 220, 220)
#     fore_im = image_file.crop(foreground_crop_rectangle)
#     background_crop_rectangle = (0, 0, 80, 40)
#     fore_im.show()
#     back_im = image_file.crop(background_crop_rectangle)
#     fore_im.save('fore_sample.png')
#     back_im.save('back_sample.png')
#     total_pixels = size[0] * size[1]
#     return ar, total_pixels + 2
# def create_histogram():

#     array, total_pixels = process_image('rose.jpeg')
#     img1 = cv.imread('fore_sample.png',0)
#     hist1 = img1.ravel(),256,[0,256]
#     img2 = cv.imread('back_sample.png',0)
#     hist2 = img2.ravel(),256,[0,256]
#     hist1_prob = hist1[0]/sum(hist1[0])
#     hist2_prob = hist2[0]/sum(hist2[0])
#     plt.hist(hist1_prob)
#     plt.show()
#     return hist1, hist2, array, total_pixels

# def create_adjacency_matrix():
#     hist1, hist2, array, total_pixels = create_histogram()

#     matrix = numpy.empty(shape=(total_pixels,total_pixels),dtype='object')
# create_histogram()

def FF(G, background, foreground):
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

    # create empty path to start (will be filled)
    path = [-1]*len(G)
    max_flow = 0

    # while a path exists between the foreground and background on the image
    while(find_path(G, background, foreground, path)):
        path_flow = float("inf")
        
        s = background
        # go until reaching the foreground
        while(s !=  foreground):
            # take the min route to traverse the image
            path_flow = min(path_flow, G[path[s],s])
            s = path[s]

        # Add path flow to overall flow
        max_flow +=  path_flow

        # update residual capacities of the edges and reverse edges along path
        v = background
        while(v !=  foreground):
            u = path[v]
            G[u][v] -= path_flow
            G[v][u] += path_flow
            v = path[v]

    return path



def find_path(G, background, foreground, path):
    """
    finds a path from the source to sink using BFS
    """

    # create dictionary of visited or unvisited based on city name
    visited = {}
    for i in range(len(G)):
        visited[i] = False
    visited[foreground] = False

    # Create a queue for BFS
    queue=[]

    # Mark the source node as visited and enqueue it
    queue.append(background)
    visited[background] = True

    while queue:

        # dequeue a pixel from queue
        u = queue.pop(0)

        # iterate through the pixels
        for ind in range(len(G)):
            # check for weights
            if visited[ind] == False and G[u,ind] > 0:
                queue.append(ind)
                visited[ind] = True
                path[ind] = u

    # If we reached sink in BFS starting from source return True
    return True if visited[foreground] else False

