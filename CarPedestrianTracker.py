import cv2

# Our Image
img_file = 'carImage.jpg'
video = cv2.VideoCapture('carDriving.mp4')
#video = cv2.VideoCapture('pedestrianWalk.mp4')

# Our pretrained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'pedestrian_detector.xml'

# Create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)



# Run while car video drives
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars AND pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw Rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw Rectangles around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


    # Display the image with the cars spotted
    cv2.imshow('Lothario Car Detector', frame)

    # Don't autoclose... wait here in the code & listen for a key press
    cv2.waitKey(1) #1 milisec



""" # Create opencv image... reads the pic into a multi-dimensional array
img = cv2.imread(img_file)

# Convert to grayscale... needed for haar cascade
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #rgb but they make it backwards

# Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_and_white)

# Draw Rectangles around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)




# Display the image with the cars spotted
cv2.imshow('Lothario Car Detector', img)

# Don't autoclose... wait here in the code & listen for a key press
cv2.waitKey()
 """
print("Code completed")