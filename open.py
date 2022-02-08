import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os

from streamlit_player import st_player

# Embed a music from SoundCloud
lt_music = ['Demons by Imagine Dragons', 'Heat Waves by Glass Animals', 'What If Told You That I Love You by Ali Gatie', '']


def load_img(img):
	im = Image.open(img)
	return im 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_faces(original_image):

	new_img = np.array(original_image.convert('RGB'))
	img = cv2.cvtColor(new_img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Detect Faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

	return img, faces    

def detect_smiles(original_image):

	new_img = np.array(original_image.convert('RGB'))
	img = cv2.cvtColor(new_img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Detect Faces
	faces = smile_cascade.detectMultiScale(gray, 1.1, 4)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

	return img, faces    

def detect_eyes(original_image):

	new_img = np.array(original_image.convert('RGB'))
	img = cv2.cvtColor(new_img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Detect Faces
	faces = eyes_cascade.detectMultiScale(gray, 1.1, 4)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

	return img, faces    
	
def main():

	st.title('OpenCV Web App')

	st.text('OpenCV was started at Intel in 1999 by Gary Bradsky')
	st.text('It is a Python library which is designed to solve computer vision problems')

	st.sidebar.title('OpenCV Web App')

	st.header('play some music while you explore the app')

	#Music Player
	select_music = st.sidebar.selectbox("Select a Song:",['Demons by Imagine Dragons', 'Heat Waves by Glass Animals', 
		'What If Told You That I Love You by Ali Gatie', 'Just the two of us by Bill Withers and Grover Washington, jr.', 
		'Shivers by Ed Sheeran', 'STAY by Justin Bieber'])

	if select_music == 'Demons by Imagine Dragons':
		st_player("https://soundcloud.com/imaginedragons/demons")

	if select_music == 'Heat Waves by Glass Animals':
		st_player("https://soundcloud.com/glassanimals/heat-waves")

	if select_music == 'What If Told You That I Love You by Ali Gatie':
		st_player("https://soundcloud.com/aligatie/what-if-i-told-you-that-i-love")

	if select_music == 'Just the two of us by Bill Withers and Grover Washington, jr.':
		st_player("https://soundcloud.com/clouddigga/bill-withers-grover-washington-jr-just-the-two-of-us-pop-ups-flip")

	if select_music == 'Shivers by Ed Sheeran':
		st_player("https://soundcloud.com/edsheeran/ed-sheeran-shivers")

	if select_music == 'STAY by Justin Bieber':
		st_player("https://soundcloud.com/thekidlaroi/stay")

	#Image Enhancements
	choice = st.sidebar.selectbox("Select Task:",['About', 'Image Enhancements', 'Face Detection'])
	
	if choice == "About":
		st.subheader("About Me")

		st.write("Hi, I am Ankit, a grad student at SFU. I love creating and deploying beautiful ML, DL, and CV web-apps, designed with Streamlit")
		st.write("Please visit: https://ankit1202.github.io to check out more projects")

	if choice == "Image Enhancements":
		st.subheader("Image Enhancements")

		img = st.file_uploader("Upload Image", type = ["jpg", "png", "jpeg", "tiff"])

		if img is not None:
			original_image = Image.open(img)
			st.text('Original Image')
			st.image(original_image)
			
		enhance_type = st.sidebar.radio("Image Enhancement Options:", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blur", "Re-size"])

		
		if enhance_type == "Gray-Scale":

			new_img = np.array(original_image.convert('RGB'))
			img = cv2.cvtColor(new_img, 1)#Converting the Image into Gray ScaleimgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#Converting the Image into Gray Scale
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#st.write(new_img)
			st.text('Gray-Scale Image')
			st.image(img)

		if enhance_type == "Contrast":
			rate = st.sidebar.slider("Contrast Rate", 0.5, 10.5)
			enhancer = ImageEnhance.Contrast(original_image)
			img_output = enhancer.enhance(rate)

			st.text("Contrast Image")
			st.image(img_output)

		if enhance_type == "Brightness":
			rate = st.sidebar.slider("Brightness Rate", 0.5, 4.5)
			enhancer = ImageEnhance.Brightness(original_image)
			img_output = enhancer.enhance(rate)

			st.text("Contrast Image")
			st.image(img_output)

		if enhance_type == "Blur":

			new_img = np.array(original_image.convert('RGB'))
			rate = st.sidebar.slider("Blur Rate", 0.5, 5.5)
			
			img = cv2.cvtColor(new_img, 1)#Converting the Image into Gray ScaleimgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#Converting the Image into Gray Scale
			img = cv2.GaussianBlur(img, (11, 11), rate)

			#st.write(new_img)
			st.text('Blurry Image')
			st.image(img)

		if enhance_type == "Re-size":

			new_img = np.array(original_image.convert('RGB'))

			h_rate = st.sidebar.slider("Width Rate", 50, 800)
			W_rate = st.sidebar.slider("Height Rate", 50, 800)
			
			img = cv2.cvtColor(new_img, 1)#Converting the Image into Gray ScaleimgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#Converting the Image into Gray Scale
			img_resize = cv2.resize(img, (h_rate, W_rate))

			#st.write(new_img)
			st.text('Re-sized Image')
			st.image(img_resize)

	if choice == "Face Detection":
		st.subheader("Face Detection")

		img = st.file_uploader("Upload Image", type = ["jpg", "png", "jpeg", "tiff", 'WebP'])

		if img is not None:
			original_image = Image.open(img)
			st.text('Original Image')
			st.image(original_image)
			
		face_type = st.sidebar.selectbox("Face Detection Options:", ["Face", "Smile", "Eyes"])

		if st.button("Process"):

			if face_type == 'Face':

				result_img, result_faces = detect_faces(original_image)
				st.image(result_img)
				st.success("Found {} faces".format(len(result_faces)))

			if face_type == 'Smile':

				result_img, result_faces = detect_smiles(original_image)
				st.image(result_img)
				st.success("Found {} Smiles".format(len(result_faces)))

			else:

				result_img, result_faces = detect_eyes(original_image)
				st.image(result_img)
				st.success("Found {} Eyes".format(len(result_faces)))

if __name__ == '__main__':
	main()