# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import smtplib,ssl
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.utils import formataddr
import glob
from PIL import Image
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		result=label
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		#result=label
		if (result =="No Mask" ):
			img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			plt.imshow(img1)
			plt.savefig(r"C:\\Users\\sravya\\Desktop\\Face\\Face-Mask-Detection-master\\Face-Mask-Detection-master\\alert_images\\img.png")
			print("image saved")
			sender_name="sravya"
			sender_mail="178w1a1207@vrsiddhartha.ac.in"
			receiver_name="admin"
			receiver_mail="178w1a1207@vrsiddhartha.ac.in"
			email_html=open('alert.html')
			email_body=email_html.read()
			filename=r"C:\\Users\\sravya\\Desktop\\Face\\Face-Mask-Detection-master\\Face-Mask-Detection-master\\alert_images\\img.png"
			msg=MIMEMultipart()
			msg['To']=formataddr((receiver_name,receiver_mail))
			msg['From']=formataddr((sender_name,sender_mail))
			msg['Subject']="Alert Message-Not Wearing Mask"
			msg.attach(MIMEText(email_body,'html'))
			try:
				with open(filename,'rb') as attachment:
						part=MIMEBase("application","octet-stream")
						part.set_payload(attachment.read())
				encoders.encode_base64(part)
				part.add_header(
					"Content-Disposition",
					f"attachment; filename={filename}",
					)
				msg.attach(part)
			except Exception as e:
				print("Oh no we didn't found the attachment")
			try:
				mail=smtplib.SMTP('smtp.gmail.com',587)
				context=ssl.create_default_context()
				mail.starttls(context=context)
				print("sending mail")
				mail.login(sender_mail,'Class@1985')
				mail.sendmail(sender_mail,receiver_mail,msg.as_string())
				print("mail sent")
			except Exception as e:
				print("Something went wrong...")
			finally:
				mail.quit()



		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
