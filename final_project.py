import cv2
from fer import FER
import random
import threading
import webbrowser

# --------- Emotion to Spotify track URLs ---------
emotion_spotify_map = {
    'happy': ['https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH?si=14dfe9ca6c2449b8'],
    'sad': ['https://open.spotify.com/track/6wf7Yu7cxBSPrRlWeSeK0Q?si=186f14c5f7fd44c3'],
    'angry': ['https://open.spotify.com/track/0tZ3mElWcr74OOhKEiNz1x?si=66c9a440abc74bd6'],
    'neutral': ['https://open.spotify.com/track/2VGGQdroduj4dIMGPsBzDG?si=2cf3eefbec874c27']
}

# --------- Function to play song based on emotion ---------
def play_music(emotion):
    urls = emotion_spotify_map.get(emotion, [])
    if urls:
        url = random.choice(urls)
        print(f"🎵 Opening song for emotion '{emotion}': {url}")
        webbrowser.open(url)
    else:
        print(f"⚠️ No track found for emotion: {emotion}")

# --------- Initialize Emotion Detector ---------
detector = FER(mtcnn=True)

# --------- Open webcam ---------
cap = cv2.VideoCapture(0)
last_emotion = None

print("🟢 Starting emotion detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotion
    result = detector.detect_emotions(frame)
    if result:
        dominant_emotion = detector.top_emotion(frame)
        if dominant_emotion:
            emotion = dominant_emotion[0]

            # If emotion changed, play a new song
            if emotion != last_emotion:
                last_emotion = emotion
                threading.Thread(target=play_music, args=(emotion,), daemon=True).start()

            # Display detected emotion
            cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show video feed
    cv2.imshow('Emotion-Based Spotify Music Player', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
