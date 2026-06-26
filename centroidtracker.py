from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # Initialize the next unique object ID, two ordered dictionaries
        # (one to store the mapping of objectID to centroid, and one to
        # store the number of consecutive frames the object has been marked
        # as "disappeared"), and the max number of consecutive frames an
        # object is allowed to be marked as "disappeared" until we deregister it
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # When registering an object use the next available object ID
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = inputRect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # To deregister an object ID we delete the object ID from both
        # of our respective dictionaries
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # Check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # Loop over any existing tracked objects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # If we have reached a maximum number of consecutive frames
                # where a given object has been marked as missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Return early as there are no centroids or tracking info to update
            return self.bboxes

        # Initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []

        # Loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # Use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])

        # If we are currently not tracking any objects, take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])

        # Otherwise, there are currently tracking objects so we need to
        # try to match the input centroids to existing object centroids
        else:
            # Grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids
            # and input centroids, respectively -- our goal will be to
            # match an input centroid to an existing object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # In order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # Next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # In order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # Loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # If we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # If the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # Otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = inputRects[col]
                self.disappeared[objectID] = 0

                # Indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # Compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # In the event that the number of object centroids is
            # equal to or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # Loop over the unused row indexes
                for row in unusedRows:
                    # Grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # Check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # exceeds our max, and if so, deregister the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # Otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # Return the set of trackable objects
        return self.bboxes
