import numpy as np
import cv2

videoCapture = cv2.VideoCapture('test_images/result.mp4')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
numFrames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

classes = np.load("test_images/classes.npy")
# query results
query = 'cat'
query_frame = []
bi_query = [0]*numFrames
for i in range(len(classes)):
	for j in range(len(classes[i])):
		if classes[i][j] == query:
			query_frame.append(round(i/fps,2))
			# query_frame.append(i)
			bi_query[i] = 1
			break

print "Time of " + query
print query_frame

success, frame = videoCapture.read()
counter = 0
while success:
	if bi_query[counter] == 1:
		cv2.imshow('frame', frame)
		cv2.waitKey(1000/int(fps))
  	success, frame = videoCapture.read()
  	counter += 1

videoCapture.release()
cv2.destroyAllWindows()