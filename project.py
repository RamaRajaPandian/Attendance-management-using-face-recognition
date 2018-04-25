#!/usr/bin/env python


from PyQt4 import QtCore, QtGui
import cv2,os
import numpy as np
from PIL import Image 
import pickle
import MySQLdb
import datetime

db=MySQLdb.connect("localhost","root","Lavairiswin1","r")
cursor=db.cursor()

temp={}

sql="select * from users"
cursor.execute(sql)
result=cursor.fetchall()


for row in result:
	roll_no=row[0]
	name=row[1]
	temp[roll_no]=name

db.commit()

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

def viewAttendance():
	db=MySQLdb.connect("localhost","root","Lavairiswin1","r")
	cursor=db.cursor()

	day=int(datetime.datetime.now().strftime("%d"))
	#sql="update practice set %s=1 where roll_no='%d'" %(today,1)
	sql="select * from project"
	cursor.execute(sql)
	results = cursor.fetchall()
	for row in results:
		roll_no=row[0]
		name=row[1]
		c=row.count(1)
		t=row[day+1]
		d=float(float(c*100)/30)
		
        	print "Roll_no=%d Name=%s Overall=%.2f%% today=%d" % (roll_no,name,d,t)
 
	db.commit()

def dataSetGenerator():
	db=MySQLdb.connect("localhost","root","Lavairiswin1","r")
	cursor=db.cursor()
	cam = cv2.VideoCapture(0)
	detector=cv2.CascadeClassifier('Classifiers/face.xml')
	i=0
	offset=50
	name=raw_input('enter your id')
	user=raw_input('Enter your name')
	temp[name]=user
	sql="insert into users values(%d,'%s')" %  (int(name),user)
	cursor.execute(sql)
	sql="insert into project values(%d,'%s',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)" % (int(name),user)
	cursor.execute(sql)
	db.commit()
	while True:
    		ret, im =cam.read()
    		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    		faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    		for(x,y,w,h) in faces:
        		i=i+1
        		cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        		cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        		cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        		cv2.waitKey(100)
   		if i>20:
        		cam.release()
        		cv2.destroyAllWindows()
        		break

def get_images_and_labels(path):

     recognizer = cv2.createLBPHFaceRecognizer()
     cascadePath = "Classifiers/face.xml"
     faceCascade = cv2.CascadeClassifier(cascadePath);
     path = 'dataSet'

     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     images = []
     labels = []
     for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L')
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         #print nbr
         faces = faceCascade.detectMultiScale(image)
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     return images, labels

def trainer():
	recognizer = cv2.createLBPHFaceRecognizer()
	cascadePath = "Classifiers/face.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath);
	path = 'dataSet'

	images, labels = get_images_and_labels(path)
	cv2.imshow('test',images[0])
	cv2.waitKey(1)

	recognizer.train(images, np.array(labels))
	recognizer.save('trainer/trainer.yml')
	cv2.destroyAllWindows()

def haze_removal(image, windowSize=24, w0=0.6, t0=0.1):

    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    result = Image.fromarray(J)

    return result

def recognize():
	
	db=MySQLdb.connect("localhost","root","Lavairiswin1","r")
	cursor=db.cursor()


	recognizer = cv2.createLBPHFaceRecognizer()
	recognizer.load('trainer/trainer.yml')
	cascadePath = "Classifiers/face.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath);
	path = 'dataSet'	
	
	cam = cv2.VideoCapture(0)
	font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) 
	while True:
    		ret, im =cam.read()
    
    		image=np.array(im)
    		result=haze_removal(image)
    		img=np.array(result)
    
    		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    		faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    		month=datetime.datetime.now().strftime("%B")
    		day=datetime.datetime.now().strftime("%d")
    		today=month[0:3]+"_"+day
    		today=str(today)
    
    		for(x,y,w,h) in faces:
        		nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        		cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        		sql="update project set %s=1 where roll_no='%d'" % (today,nbr_predicted)
			nbr_predicted=temp[nbr_predicted]
			cursor.execute(sql)
			db.commit()
        		
	
        		cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255)
        		cv2.imshow('im',im)
        		#cv2.waitKey(10)
    		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
	cam.release()
	cv2.destroyAllWindows()




class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(532, 341)
        self.addUser = QtGui.QPushButton(Form)
        self.addUser.setGeometry(QtCore.QRect(30, 40, 121, 41))
        self.addUser.setObjectName(_fromUtf8("addUser"))
        self.trainDataSet = QtGui.QPushButton(Form)
        self.trainDataSet.setGeometry(QtCore.QRect(30, 100, 121, 41))
        self.trainDataSet.setObjectName(_fromUtf8("trainDataSet"))
        self.recognize = QtGui.QPushButton(Form)
        self.recognize.setGeometry(QtCore.QRect(30, 160, 121, 41))
        self.recognize.setObjectName(_fromUtf8("recognize"))
        self.viewAttendance = QtGui.QPushButton(Form)
        self.viewAttendance.setGeometry(QtCore.QRect(30, 220, 121, 41))
        self.viewAttendance.setObjectName(_fromUtf8("viewAttendance"))
        self.labelAddUser = QtGui.QLabel(Form)
        self.labelAddUser.setGeometry(QtCore.QRect(190, 40, 311, 41))
        self.labelAddUser.setObjectName(_fromUtf8("labelAddUser"))
        self.labelTrainDataSet = QtGui.QLabel(Form)
        self.labelTrainDataSet.setGeometry(QtCore.QRect(190, 100, 291, 31))
        self.labelTrainDataSet.setObjectName(_fromUtf8("labelTrainDataSet"))
        self.labelRecognize = QtGui.QLabel(Form)
        self.labelRecognize.setGeometry(QtCore.QRect(190, 160, 291, 31))
        self.labelRecognize.setObjectName(_fromUtf8("labelRecognize"))
        self.labelViewAttendance = QtGui.QLabel(Form)
        self.labelViewAttendance.setGeometry(QtCore.QRect(190, 220, 281, 31))
        self.labelViewAttendance.setObjectName(_fromUtf8("labelViewAttendance"))

        self.retranslateUi(Form)
        QtCore.QObject.connect(self.addUser, QtCore.SIGNAL(_fromUtf8("clicked()")), dataSetGenerator)
        QtCore.QObject.connect(self.trainDataSet, QtCore.SIGNAL(_fromUtf8("clicked()")), trainer)
        QtCore.QObject.connect(self.recognize, QtCore.SIGNAL(_fromUtf8("clicked()")), recognize)
        QtCore.QObject.connect(self.viewAttendance, QtCore.SIGNAL(_fromUtf8("clicked()")), viewAttendance)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Automatic attendance managenment using face recognition ", None))
        self.addUser.setText(_translate("Form", "Add User", None))
        self.trainDataSet.setText(_translate("Form", "Train DataSet", None))
        self.recognize.setText(_translate("Form", "Recognize", None))
        self.viewAttendance.setText(_translate("Form", "View Atendance", None))
        self.labelAddUser.setText(_translate("Form", "Add new user to DataSet", None))
        self.labelTrainDataSet.setText(_translate("Form", "Train the DataSet", None))
        self.labelRecognize.setText(_translate("Form", "Click to recognize and update attendance", None))
        self.labelViewAttendance.setText(_translate("Form", "Click to view overall and today's Attendance", None))



if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Form = QtGui.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

