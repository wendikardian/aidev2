import cv2

# Initialize the video capture device
capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    _, frame = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Display the resulting image
    cv2.imshow("Threshold Image", threshold)

    # Check if the user pressed the 'q' key to exit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture device and close all windows
capture.release()
cv2.destroyAllWindows()