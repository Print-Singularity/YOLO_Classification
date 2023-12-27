# Importing needed libraries
import numpy as np
import cv2
import time



image_BGR = cv2.imread("istockphoto-1078377006-612x612.jpg")
h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)

with open("classes.names") as f:
    labels = [line.strip() for line in f]

print('List with labels names:')
print(labels)

network = cv2.dnn.readNetFromDarknet("cfg\custom_train.cfg",
                                     "weights\yolov3_custom_train_best.weights")

layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]       

# # Check point
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
threshold = 0.5

# Generating colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


# Implementing
network.setInput(blob)                                  # setting blob as input to the network
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
print('Objects Detection took {:.5f} seconds'.format(end - start))

bounding_boxes = []
confidences = []
class_numbers = []


# Going through all output layers after feed forward pass
for result in output_from_network:
    # Going through all detections from current output layer
    for detected_objects in result:
        # print(detected_objects)
        # Getting 80 classes' probabilities for current detected object
        scores = detected_objects[5:]
        # Getting index of the class with the maximum value of probability
        class_current = np.argmax(scores)
        # Getting value of probability for defined class
        confidence_current = scores[class_current]
        # print("####", scores, class_current, confidence_current)
        # Eliminating weak predictions with minimum probability
        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # Adding results into prepared lists
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)
# Defining counter for detected objects
counter = 1

# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
        print("Percent object is ",confidences[i])

        # Incrementing counter
        counter += 1

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        # Drawing bounding box on the original image
        cv2.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        # Preparing text with label and confidence for current bounding box
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Putting text with label and confidence on the original image
        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
cv2.imshow('Detections', image_BGR)   
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)
cv2.waitKey(0)                                      # Waiting for any key being pressed
cv2.destroyWindow('Detections')                     # Destroying opened window with name 'Detections'
