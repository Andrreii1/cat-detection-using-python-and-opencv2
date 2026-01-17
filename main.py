import cv2
import time

cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

def on_cat_detected():
    print(" Cat detected!")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print(" Could not open webcam")
    exit()

print(" Camera is open, press 'q' to quit.")


last_box = None
last_seen_time = 0
cat_present = False
HOLD_TIME = 0.5  
SMOOTHING = 0.7  

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cats = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15
    )

    current_time = time.time()

    if len(cats) > 0:
        x, y, w, h = cats[0]  

        if last_box is None:
            last_box = (x, y, w, h)
        else:
            lx, ly, lw, lh = last_box
            x = int(lx * SMOOTHING + x * (1 - SMOOTHING))
            y = int(ly * SMOOTHING + y * (1 - SMOOTHING))
            w = int(lw * SMOOTHING + w * (1 - SMOOTHING))
            h = int(lh * SMOOTHING + h * (1 - SMOOTHING))
            last_box = (x, y, w, h)

        last_seen_time = current_time

        if not cat_present:
            on_cat_detected()
            cat_present = True

    if last_box is not None and (current_time - last_seen_time) < HOLD_TIME:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    else:
        last_box = None
        cat_present = False

    cv2.imshow(" Cat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
