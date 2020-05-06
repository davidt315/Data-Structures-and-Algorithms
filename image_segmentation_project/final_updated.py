from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def process_image(im_file):
    """
    Converts to grayscale, resizes, and creates a background and foreground image
    """
    image_file = Image.open(im_file)
    image_file = image_file.resize((64,64))
    size = image_file.size
    image_file = image_file.convert('L') # convert image to black and white
    image_file.save('result.png')
    ar = np.array(image_file)
    
    foreground_crop_rectangle = (20, 20, 30, 30)
    fore_im = image_file.crop(foreground_crop_rectangle)
    #fore_im.show()
    background_crop_rectangle = (0, 0, 10, 10)
    back_im = image_file.crop(background_crop_rectangle)
    #back_im.show()
    image_file.show()
    fore_im.save('fore_sample.png')
    back_im.save('back_sample.png')
    total_pixels = size[0] * size[1]
    return ar, total_pixels + 2


def create_histogram():
    array, total_pixels = process_image('bear.jpg')
    img1 = cv.imread('fore_sample.png',0)
    hist1,bins1 = np.histogram(img1.ravel(),64,[0,64])
    img2 = cv.imread('back_sample.png',0)
    hist2,bins2 = np.histogram(img2.ravel(),64,[0,64])
    plt.hist(hist1, bins1)
    plt.show()
    return hist1, hist2, array, total_pixels, bins1


def create_adjacency_matrix():
    hist1, hist2, array, total_pixels, bins1 = create_histogram()
    print(len(array[0]))
    hist1 = hist1 / sum(hist1)
    hist2 = hist2 / sum(hist2)
    print(len(hist1))
    matrix = np.zeros(shape=(total_pixels,total_pixels))
    matrix_size = int(np.sqrt(matrix.size))

    pixel_vals = []
    for i in range(len(array)):
        for j in range(len(array[0])):
            pixel = array[i][j]
            pixel_vals.append(pixel)
    for k in range(len(pixel_vals)):
        pixel = array[i][j]
        #print(pixel)
        if hist1[pixel] == 0 and hist2[pixel] == 0:
            prob_fore = 0.5
            prob_back = 0.5
        else:
            prob_fore = hist1[pixel]/(hist1[pixel] + hist2[pixel])
            prob_back = hist2[pixel]/(hist1[pixel] + hist2[pixel])
        matrix[matrix_size-2][k] = prob_fore
        matrix[k][matrix_size-1] = prob_back

    matrix_hor = 0
    matrix_ver = 0
    print(len(array))
    for k in range(0, len(pixel_vals), len(array[0])):
        for i in range(len(array[0])-1):
            pixel1 = pixel_vals[i+k]
            pixel2 = pixel_vals[i+k+1]
            prob1 = 2.71**(-(pixel1-pixel2)**2)
            prob2 = 2.71**(-(pixel2-pixel1)**2)
            matrix[i+k][i+k+1] = prob1
            matrix[i+k+1][i+k] = prob2

    for i in range(len(array[0])):
        for k in range(len(array)-1):
            pixel1 = pixel_vals[i+k*len(array[0])]
            pixel2 = pixel_vals[(k+1)*len(array[0])+i]
            prob1 = 2.71**(-(pixel1-pixel2)**2)
            prob2 = 2.71**(-(pixel2-pixel1)**2)
            matrix[i+k*len(array[0])][(k+1)*len(array[0])+i] = prob1
            matrix[(k+1)*len(array[0])+i][i+k*len(array[0])] = prob2
    return matrix
matrix = create_adjacency_matrix()
print(matrix)
print(len(matrix))


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
    back = len(G)-2
    fore = len(G)-1

    # create empty path to start (will be filled by one iteration of BFS)
    # BFS will check all nodes (filling the path completely) and will update path
    path = [-1]*len(G)
    max_flow = 0

    # while a path exists between the foreground and background on the image
    while(find_path(G, path)):
        path_flow = float("inf")
        
        s = back
        # go until reaching the foreground
        while(s !=  fore):
            # take the min route to traverse the image
            path_flow = min(path_flow, G[path[s],s])
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
    back = len(G)-2
    fore = len(G)-1
    # create dictionary of visited or unvisited based on city name
    visited = {}
    # make all pixels as unvisited
    for i in range(len(G)-2):
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
            if visited[ind] == False and G[u,ind] > 0:
                queue.append(ind)
                visited[ind] = True
                path[ind] = u

    # If we reached sink in BFS starting from source return True
    return True if visited[fore] else False
