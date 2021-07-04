from tkinter import *
from tkinter import messagebox,filedialog
import imutils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import cv2


class Application:
    def __init__(self):
        self.root = Tk()
        self.root.title=("File Organizer")
        self.root.iconbitmap("icons\Icon.ico")

        self.canvas = Canvas(self.root,width=400,height=200)
        self.canvas.configure(bg="#1c2461")
        self.canvas.pack()

        self.lbl1=Label(self.root,text="Object Detection app",fg="#dbe0ff",bg="#1c2461")
        self.lbl1.configure(font=("Times",18,"bold"))
        self.canvas.create_window(200,20,window=self.lbl1)

        self.entry=Entry(self.root,width=50)
        self.canvas.create_window(180,70,window=self.entry)

        self.lbl2=Label(self.root,text="Enter or browse the video file\n",fg="#dbe0ff",bg="#1c2461")
        self.canvas.create_window(200,105,window=self.lbl2)

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe("models\MobileNetSSD_deploy.prototxt.txt", "models\MobileNetSSD_deploy.caffemodel")

        def btn1_clicked():
            print("[INFO] starting video stream...")
            vs = VideoStream().start()
            time.sleep(1.0)
            fps = FPS().start()

            # loop over the frames from the video stream
            while True:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=400)

                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                             0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()
                # print(detections.shape)

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.2:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

                # update the FPS counter
                fps.update()

            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()

        def btn2_clicked():
            pth = self.entry.get()
            if pth == "":
                select_filename()
            pth = self.entry.get()
            pth = pth.replace("\\", "/")

            print("[INFO] starting video stream...")
            vs = FileVideoStream(pth).start()
            time.sleep(1.0)
            fps = FPS().start()

            # loop over the frames from the video stream
            while True:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                try:
                    frame = imutils.resize(frame, width=400)
                except:
                    print("Video Ended\n")
                    break

                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                             0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()
                # print(detections.shape)

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.4:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

                # update the FPS counter
                fps.update()

            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()

        def select_filename():
            initial_dir = os.path.dirname(os.path.abspath(__file__))
            directory=filedialog.askopenfilename(parent=self.root,initialdir=initial_dir)
            self.entry.insert(0,directory)

        self.photo=PhotoImage(file="icons\Browse-folder.png")
        self.photoimage=self.photo.subsample(5,5)
        self.btn2=Button(self.root,command=select_filename,image=self.photoimage)
        self.canvas.create_window(360,70,window=self.btn2)

        self.btn1 = Button(self.root, text="Click to start live stream", command=btn1_clicked)
        self.canvas.create_window(100, 150, window=self.btn1)

        self.btn2 = Button(self.root, text="Click to start video stream", command=btn2_clicked)
        self.canvas.create_window(300, 150, window=self.btn2)

        self.root.mainloop()

obj=Application()