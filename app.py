import cv2
import streamlit as st
from deepface import DeepFace
import google.generativeai as genai
import os
os.environ["Your_Api_key"] = "Your_Api_key"
genai.configure(api_key=os.environ["Your_Api_key"])
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
def get_genai_recommendation(emotion):
    prompt = f"""Provide a medical or well-being suggestion for someone feeling {emotion}. Include any professional advice, therapeutic suggestions, or medical methods that could help relieve the symptoms of {emotion}. Suggest techniques or possible treatments."""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text if response else "No response"
    except Exception as e:
        return f"An error occurred: {str(e)}"
emoji_to_emotion = {
    "😀": "Happy",
    "😞": "Sad",
    "😡": "Angry",
    "😟": "Anxious",
    "😱": "Fear",
    "😊": "Content",
    "😭": "Crying",
    "😕": "Confused",
    "😌": "Relaxed",
}
st.title("Emotion Detector and Stress Relief Suggestions")
st.markdown("This application uses AI to detect your emotional state and provides **personalized stress relief recommendations** based on your emotion. Alternatively, you can select an emoji to represent your emotion.")
selected_emoji = st.selectbox("Select your current emotion:", options=list(emoji_to_emotion.keys()))
emotion = emoji_to_emotion[selected_emoji]
genai_recommendation = get_genai_recommendation(emotion)
st.write(f"Selected Emoji: {selected_emoji}")
st.write(f"Detected Emotion: {emotion}")
st.write(f"Recommended Action: {genai_recommendation}")
cap = cv2.VideoCapture(0)
def process_webcam_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion_detected = result[0]["dominant_emotion"]
            return emotion_detected, get_genai_recommendation(emotion_detected), frame
        except Exception as e:
            return None, f"Error in emotion analysis: {e}", frame

    return None, None, frame
if st.checkbox("Use Webcam to Detect Emotion", value=False):
    emotion_detected, genai_recommendation, frame = process_webcam_frame()

    if emotion_detected:
        st.image(frame, channels="BGR", use_column_width=True)
        st.write(f"Detected Emotion: {emotion_detected}")
        st.write(f"Recommended Action: {genai_recommendation}")
    elif frame is not None:
        st.write("No face detected. Try again.")
cap.release()
