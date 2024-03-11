import cv2
import numpy as np

video = cv2.VideoCapture("road_view.mp4")

while True:
    ret, original_frame = video.read()
    if not ret:
        break

    # First, we are blurring the image by using the Gaussian Blur,
    # to reduce image noises and irrelevant details
    frame = cv2.GaussianBlur(original_frame, (5, 5), 0)

    # We convert the frame color space from BGR to HSV,
    # which helps us to segment objects based on its color better.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # In our use case, the solid road lanes are yellow,
    # so we set range of yellow values in "HSV format"
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])

    # We create a new HSV image in which only the objects
    # that met the color range we required appear
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # On the new frame we have created, we use the
    # Canny edge detector technique
    edges = cv2.Canny(mask, 75, 150)

    # Finally, we can use the Hough transform for
    # filtering only the lines in the edges image
    # and draw it on the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()