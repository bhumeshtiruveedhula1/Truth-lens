import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera NOT opening")
else:
    print("Camera OK")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()