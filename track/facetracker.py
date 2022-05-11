import numpy as np
from scipy.spatial import distance
from collections import OrderedDict


class FaceTracker(object):
    
    """Face Tracker class for tracking faces and assigning Unique Identifiers(ID) to each face.
    
    Attributes:
        nextObjectID (integer) : A counter used to assign unique IDs to each object
        objectsTracked (dictionary) : A dictionary that utilizes the object ID as key and centroid
        disappeared (dictionary) : The number of consecutive frames(value) an objectID(key) has been lost
        maxDisappeared (integer) : The number of consecutive frames an object is allowed to be lost
            until we deregister it.
        
    """
    def __init__(self, maxDisappeared = 13):

        self.nextObjectID = 1
        self.objectsTracked = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):

        """Function to store the centroid using the next available object ID
        """
        self.objectsTracked[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):

        """Function to deregister an object ID by deleting it from both of our 
        respective dictionaries
        """
        del self.objectsTracked[objectID]
        del self.disappeared[objectID]


    def update(self, rect):

        """Function that tracks the face centroid by accepting bounding box rectangles (rect) from a 
        face detector (Caffe face model).
        
        Args:
            rect (list) : The list of bounding boxes in tuple (beginX, beginY, endX, endY)
        Returns:
            objectsTracked (dictionary) : Representing the face ID and the face Centroid when a 
            face is detected or not.
        """

        if rect == []:
            # When no face is detected, loop over the existing tracked objects (faces) and mark
            # them as disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                # When we reach the maximum frame (maxDisappeared) when an object (face) has been 
                # marked as disappeared, deregister it.
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objectsTracked

        # Initialize a centroid of zeroes
        inputCentroid = np.zeros((len(rect), 2), dtype='int')
        # Loop over the detected face(s)
        for (i, (beginX, beginY, endX, endY)) in enumerate(rect):
            # Grab and store the center of each face 
            # in the input centroid
            cX = int((beginX + endX) / 2)
            cY = int((beginY + endY) / 2)
            inputCentroid[i] = (cX, cY)

        # if we are currently not tracking any objects, take the input centroid and register each 
        # of them. Otherwise, we are currently tracking objects so we have to try 
        # to match the input centroid with the existing object centroid.
        if len(self.objectsTracked) == 0:
            for i in range(0, len(inputCentroid)):
                self.register(inputCentroid[i])
        else:
            # Grab the set of objectIDs and corresponding centroid
            objectIDs = list(self.objectsTracked.keys())
            objectsCentroid = list(self.objectsTracked.values())
            # Compute the distance between each face centroids and input centroids
            # in order to match the input centroids with the existing centroid 
            objDist = distance.cdist(np.array(objectsCentroid), inputCentroid, 'euclidean')
            
            # When there is distance(objDist) between objects (faces) do the following
            if objDist.size > 0:
                # In order to perform this matching we must find the smallest value in each row,
                # sort the row indexes based on their minimum values so that the row
                # with the smallest value is at the front of the index list
                rows = objDist.min(axis=1).argsort()
                # we perform a similar process on the columns by finding the smallest value in 
                # each column and then sorting using the previously computed row index list
                cols = objDist.argmin(axis=1)[rows]
                # In order to determine if we need to register or deregister an object, we need 
                # to keep track of each of the rows and columns we have already examined
                usedRows = set()
                usedCols = set()


                for row, col in zip(rows, cols):
                    # Ignore, if we have already examined the row or column value before
                    if row in usedRows or col in usedCols:
                        continue


                    # Otherwise, grab the object ID for the current row, set it's new
                    # centroid, and reset the disappeared counter.
                    objectID = objectIDs[row]
                    self.objectsTracked[objectID] = inputCentroid[col]
                    self.disappeared[objectID] = 0


                    # Indicate that we have examined each row and
                    # column indexes, respectively.
                    usedRows.add(row)
                    usedCols.add(col)

                # Compute both the row and column index we have
                # not yet examine
                unusedRows = set(range(0, objDist.shape[0])).difference(usedRows)
                unusedCols = set(range(0, objDist.shape[1])).difference(usedCols)

                # In the event that the number of object centroid is equal or greater than
                # the number of input centroid we need to check and see if some
                # of these objects have potentially disappeared
                if objDist.shape[0] >= objDist.shape[1]:
                    for row in unusedRows:
                        # Grab the object ID for the corresponding row index and 
                        # increment the disappeared counter
                        objectID = objectIDs[row]
                        self.disappeared[objectID] += 1
                        # Check the frame to see if the object has been marked as disappeared
                        # which warrants deregistering the object
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
                else:
                    # otherwise, if the number of input centroids is greater than the number of 
                    # existing object centroids we need to register each new 
                    # input centroid as a trackable object
                    for col in unusedCols:
                        self.register(inputCentroid[col])

        return self.objectsTracked