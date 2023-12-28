# Final Attractive UI Designs :

import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

# Function to train the model
def train_model(image_paths, labels):
    known_face_encodings = []
    known_face_names = []

    for path, label in zip(image_paths, labels):
        image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(face_encoding)
        known_face_names.append(label)

    return known_face_encodings, known_face_names

# Function to recognize faces using the trained model
def recognize_faces(frame, known_face_encodings, known_face_names):
    frame_copy = frame.copy()

    face_locations = face_recognition.face_locations(frame_copy)
    face_encodings = face_recognition.face_encodings(frame_copy, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame_copy, (left, top), (right, bottom), (216, 180, 0), 3)
        cv2.rectangle(frame_copy, (left, bottom - 35), (right, bottom), (216, 180, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_copy, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

    return frame_copy

# Streamlit app
def main():
    
    img = Image.open("Logo.jpg")
    st.set_page_config(page_title="Face Recognition",page_icon=img,layout="wide")

    # Hide Menu_Bar & Footer :

    hide_menu_style = """
        <style>
        #MainMenu {visibility : hidden;}
        footer {visibility : hidden;}
        </style>
    """
    st.markdown(hide_menu_style , unsafe_allow_html=True)
    
    # Set the background image :

    Background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main
    {
    background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-wallpaper-background_58702-3799.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");

    background-size : 100%
    background-position : top left;

    background-position: center;
    background-size: cover;
    background-repeat : repeat;
    background-repeat: round;


    background-attachment : local;

    background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-wallpaper-background_58702-3799.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");
    background-position: right bottom;
    background-repeat: no-repeat;
    }  

    [data-testid="stHeader"]
    {
    background-color : rgba(0,0,0,0);
    }

    </style>                                
    """
    st.markdown(Background_image,unsafe_allow_html=True)
    
    st.markdown("""<h1 style="color:#0d47a1;font-family:Maiandra GD;text-align:center;">FaceMate : Your Personal Face Recognition Webapp</h1>""",unsafe_allow_html=True)
    
    st.sidebar.image("Face_Recognition.jpg")
                
    # User input for image and label upload
    st.sidebar.markdown("""<h2 style="color:#bbdefb;font-family:Maiandra GD;">Upload Images</h2>""",unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader("Choose images for Training", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


    if uploaded_files:
        st.sidebar.markdown("""<h2 style="color:#bbdefb;font-family:Maiandra GD;">Enter Labels</h2>""",unsafe_allow_html=True)
        labels = st.sidebar.text_input("Enter labels separated by commas (e.g., label1, label2):").split(',')

        if st.sidebar.button("Train Model"):
            known_face_encodings, known_face_names = train_model(uploaded_files, labels)
            success = st.sidebar.info("Model trained successfully!")
            
            if success:
                st.markdown("""<h4 style="border-radius:30px;text-align:center;color:#2b2d42,font-family: Bell MT;background-color: #fff1e6;">Go to Upload Image then give your test images...</h4>""",unsafe_allow_html=True)
                st.markdown("""<br>""",unsafe_allow_html=True)
                st.markdown("""<h4 style="border-radius:30px;text-align:center;color:#2b2d42,font-family: Bell MT;background-color: #fff1e6;">It will recognize the images based on the labels...</h4>""",unsafe_allow_html=True)
                st.markdown("""<br>""",unsafe_allow_html=True)
            
            # No live camera feed, only recognition on uploaded images
            for path in uploaded_files:
                image = face_recognition.load_image_file(path)
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                modified_frame = recognize_faces(frame, known_face_encodings, known_face_names)

                # Display the resulting image
                st.image(cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
if __name__ == "__main__":
    main()
