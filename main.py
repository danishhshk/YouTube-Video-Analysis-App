# import streamlit as st
# import nltk
# import pytube
# import moviepy.editor as mp
# from moviepy.video.io.ffmpeg import ffmpeg_extract_subclip
# from moviepy.editor import AudioFileClip
# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import speech_recognition as sr
# from pydub import AudioSegment
# import cv2
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.image import img_to_array
# from io import BytesIO
# import string
# import time

# # Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# # Function to perform sentiment analysis
# def sentiment_analyse(sentiment_text):
#     score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
#     neg = score['neg']
#     pos = score['pos']
#     if neg > pos:
#         return "Negative sentiment"
#     elif pos > neg:
#         return "Positive sentiment"
#     else:
#         return "Neutral sentiment"

# # Load emotions from the file into a dictionary
# emotions = {}
# with open('emotions.txt', 'r') as file:
#     for line in file:
#         clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
#         word, emotion = clear_line.split(':')
#         emotions[word] = emotion.capitalize()  # Capitalize the first letter

# # Load the emotion classifier model
# emotion_dict = {0: 'angry', 1:'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
# with open("face_emotions.json", "r") as json_file:
#     loaded_model_json = json_file.read()
# classifier = model_from_json(loaded_model_json)
# classifier.load_weights("face_emotion.h5")

# # Load face detection model
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Function to download YouTube video
# def download_youtube_video(youtube_url):
#     yt = pytube.YouTube(youtube_url)
#     video_stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
#     video_path = video_stream.download(filename="temp_video.mp4")
#     return video_path

# # Function to process video for image analysis (frame extraction)
# def extract_frames(video_path):
#     clip = mp.VideoFileClip(video_path)
#     frames = []
#     for frame in clip.iter_frames(fps=1, dtype="uint8"):
#         frames.append(frame)
#     return frames

# # Function to process audio for sentiment analysis
# def extract_audio(video_path):
#     video = mp.VideoFileClip(video_path)
#     audio = video.audio
#     audio.write_audiofile("temp_audio.wav")
#     return "temp_audio.wav"

# # Function for audio sentiment analysis
# def analyze_audio_sentiment(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         return sentiment_analyse(text)
#     except Exception as e:
#         return "Error: Could not process audio."

# # Video processing for emotion detection
# def analyze_video_emotions(frames):
#     results = []
#     for frame in frames:
#         img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
#         for (x, y, w, h) in faces:
#             face_image = img_gray[y:y+h, x:x+w]
#             face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
#             if np.sum([face_image]) != 0:
#                 face_image = face_image.astype('float') / 255.0
#                 face_image = img_to_array(face_image)
#                 face_image = np.expand_dims(face_image, axis=0)
#                 prediction = classifier.predict(face_image)[0]
#                 max_index = int(np.argmax(prediction))
#                 emotion = emotion_dict[max_index]
#                 results.append(emotion)
#     return results

# # Streamlit UI
# def main():
#     st.title("YouTube Video Analysis App")

#     # Input for YouTube link
#     youtube_url = st.text_input("Enter YouTube Video URL")

#     if youtube_url:
#         st.write(f"Processing video: {youtube_url}")
        
#         # Download video from YouTube
#         video_path = download_youtube_video(youtube_url)
        
#         # Extract frames from the video for image analysis
#         frames = extract_frames(video_path)

#         # Extract audio for sentiment analysis
#         audio_file = extract_audio(video_path)

#         # Process video analysis (image and audio)
#         image_emotions = analyze_video_emotions(frames)
#         audio_sentiment = analyze_audio_sentiment(audio_file)

#         # Display image results
#         st.write("Image Emotion Detection Results:")
#         if image_emotions:
#             st.write(", ".join(set(image_emotions)))
#         else:
#             st.write("No faces detected in the video.")

#         # Display audio results
#         st.write("Audio Sentiment Analysis Result:")
#         st.write(audio_sentiment)

#         # Calculate final sentiment based on individual results
#         # Assuming simple majority rule (positive, neutral, or negative)
#         sentiments = [audio_sentiment]
#         if "Positive sentiment" in sentiments:
#             final_sentiment = "Positive sentiment"
#         elif "Negative sentiment" in sentiments:
#             final_sentiment = "Negative sentiment"
#         else:
#             final_sentiment = "Neutral sentiment"

#         # Display final sentiment
#         st.write(f"Final Sentiment of Video: {final_sentiment}")
        
# if __name__ == "__main__":
#     main()




# import streamlit as st
# import nltk
# import yt_dlp
# import os
# import moviepy.editor as mp
# import speech_recognition as sr
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from googleapiclient.discovery import build
# import cv2
# import numpy as np
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.image import img_to_array

# # Set explicit FFmpeg path
# os.environ["FFMPEG_BINARY"] = "C:/ffmpeg/bin/ffmpeg.exe"  # Change to your actual FFmpeg path

# # Download necessary NLTK data
# nltk.download('vader_lexicon')

# # Load emotion classifier
# emotion_dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
# with open("face_emotions.json", "r") as json_file:
#     loaded_model_json = json_file.read()
# classifier = model_from_json(loaded_model_json)
# classifier.load_weights("face_emotion.h5")

# # Load face detection model
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Function to perform sentiment analysis
# def sentiment_analyse(sentiment_text):
#     score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
#     return "Positive sentiment" if score['pos'] > score['neg'] else "Negative sentiment" if score['neg'] > score['pos'] else "Neutral sentiment"

# # Function to download YouTube video
# def download_youtube_video(youtube_url):
#     try:
#         st.write("📥 Downloading video...")
#         ydl_opts = {
#             'format': 'bestvideo+bestaudio',
#             'merge_output_format': 'mp4',
#             'outtmpl': 'temp_video.mp4',
#             'noplaylist': True,
#             'quiet': False
#         }
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([youtube_url])

#         if os.path.exists("temp_video.mp4"):
#             st.success("✅ Download complete: temp_video.mp4")
#             return "temp_video.mp4"
#         else:
#             st.error("❌ Download failed. File not found.")
#             return None
#     except Exception as e:
#         st.error(f"⚠ Error downloading video: {e}")
#         return None

# # Function to extract audio
# def extract_audio(video_path):
#     try:
#         st.write("🎵 Extracting audio...")
#         video = mp.VideoFileClip(video_path)
#         audio = video.audio
#         audio.write_audiofile("temp_audio.wav")
#         st.success("✅ Audio extracted: temp_audio.wav")
#         return "temp_audio.wav"
#     except Exception as e:
#         st.error(f"⚠ Error extracting audio: {e}")
#         return None

# # Function for speech-to-text and sentiment analysis
# def analyze_audio_sentiment(audio_file):
#     st.write("🎤 Analyzing audio sentiment...")
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         sentiment = sentiment_analyse(text)
#         st.success(f"✅ Audio Sentiment: {sentiment}")
#         return sentiment
#     except:
#         st.error("⚠ Error: Could not process audio.")
#         return "Error"

# # Function to extract frames for emotion analysis
# def extract_frames(video_path):
#     st.write("🎥 Extracting video frames for emotion detection...")
#     frames = []
#     try:
#         clip = mp.VideoFileClip(video_path)
#         frame_rate = min(clip.fps, 1)  # Extract 1 frame per second
#         duration = int(clip.duration)

#         progress_bar = st.progress(0)
#         for t in range(0, duration, int(frame_rate)):
#             frame = clip.get_frame(t)
#             frames.append(frame)
#             progress_bar.progress((t + 1) / duration)

#         st.success(f"✅ Extracted {len(frames)} frames.")
#     except Exception as e:
#         st.error(f"⚠ Error extracting frames: {e}")

#     return frames

# # Function for facial emotion analysis
# def analyze_video_emotions(frames):
#     st.write("🔍 Analyzing video frames for emotions...")
#     results = []
#     for frame in frames:
#         try:
#             img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
#             for (x, y, w, h) in faces:
#                 face_image = img_gray[y:y+h, x:x+w]
#                 face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
#                 if np.sum([face_image]) != 0:
#                     face_image = face_image.astype('float') / 255.0
#                     face_image = img_to_array(face_image)
#                     face_image = np.expand_dims(face_image, axis=0)
#                     prediction = classifier.predict(face_image)[0]
#                     emotion = emotion_dict[int(np.argmax(prediction))]
#                     results.append(emotion)
#         except Exception as e:
#             st.error(f"⚠ Error in emotion detection: {e}")

#     if results:
#         st.success(f"🎭 Detected emotions: {', '.join(set(results))}")
#     else:
#         st.warning("⚠ No faces detected.")

#     return results

# # Function to fetch YouTube comments using API
# def fetch_comments(youtube_url, api_key):
#     st.write("💬 Fetching YouTube comments...")
#     video_id = youtube_url.split("v=")[1].split("&")[0]
#     youtube = build("youtube", "v3", developerKey=api_key)
#     comments = []
#     try:
#         response = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText").execute()
#         while response:
#             comments.extend([item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]])
#             if "nextPageToken" in response:
#                 response = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", pageToken=response["nextPageToken"]).execute()
#             else:
#                 break
#     except Exception as e:
#         st.error(f"⚠ Error fetching comments: {e}")

#     return comments

# # Streamlit UI
# def main():
#     st.title("YouTube Video Analysis App")

#     youtube_url = st.text_input("📌 Enter YouTube Video URL")
#     api_key = st.text_input("🔑 Enter YouTube API Key", type="password")

#     if youtube_url and api_key:
#         st.write(f"📊 Processing video: {youtube_url}")

#         video_path = download_youtube_video(youtube_url)
#         if video_path:
#             frames = extract_frames(video_path)
#             audio_file = extract_audio(video_path)

#             if audio_file:
#                 audio_sentiment = analyze_audio_sentiment(audio_file)

#             comments = fetch_comments(youtube_url, api_key)
#             comment_sentiments = [sentiment_analyse(comment) for comment in comments] if comments else []

#             image_emotions = analyze_video_emotions(frames) if frames else []

#             st.write("🎭 **Detected Emotions:**", ", ".join(set(image_emotions)) if image_emotions else "No faces detected.")
#             st.write("🎤 **Audio Sentiment:**", audio_sentiment)
#             st.write("💬 **Comment Sentiments:**", ", ".join(set(comment_sentiments)) if comments else "No comments found.")

#             # Final sentiment decision
#             sentiments = [audio_sentiment] + comment_sentiments
#             final_sentiment = "Positive sentiment" if "Positive sentiment" in sentiments else "Negative sentiment" if "Negative sentiment" in sentiments else "Neutral sentiment"
#             st.write(f"✅ **Final Video Sentiment:** {final_sentiment}")

# if __name__ == "__main__":
#     main()



# third verion

# import os
# import cv2
# import numpy as np
# import streamlit as st
# import yt_dlp
# from multiprocessing import Pool

# # -------------------- 1️⃣ Download YouTube Video Using yt-dlp --------------------
# def download_youtube_video(youtube_url):
#     st.write("📥 Downloading YouTube Video using yt-dlp...")
#     output_path = "downloaded_video.mp4"

#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
#         'outtmpl': output_path,
#         'quiet': True
#     }

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([youtube_url])
#         st.success(f"✅ Video downloaded: {output_path}")
#         return output_path
#     except Exception as e:
#         st.error(f"⚠ Error downloading video: {e}")
#         return None

# # -------------------- 2️⃣ Downscale Video to 720p (Reduces Processing Time) --------------------
# def downscale_video(input_video, output_video="temp_video_resized.mp4"):
#     st.write("📉 Downscaling video to 720p...")
#     os.system(f'ffmpeg -i {input_video} -vf "scale=1280:720" -preset ultrafast {output_video} -y -loglevel error')
#     st.success("✅ Video downscaled.")
#     return output_video

# # -------------------- 3️⃣ OpenCV-Based Frame Extraction (Single-threaded) --------------------
# def extract_frames(video_path, fps=1, output_folder="frames"):
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
    
#     frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_interval = max(frame_rate // fps, 1)  # Ensure at least 1 frame is extracted
    
#     st.write("🚀 Extracting video frames using OpenCV...")
#     count, frame_count = 0, 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % frame_interval == 0:
#             frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
#             cv2.imwrite(frame_filename, frame)
#             frame_count += 1
#         count += 1

#     cap.release()
#     st.success(f"✅ Extracted {frame_count} frames.")
#     return output_folder

# # -------------------- 4️⃣ Parallel Frame Extraction Using Multiprocessing --------------------
# def process_frame(args):
#     frame_number, video_path, output_folder = args
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     ret, frame = cap.read()
#     if ret:
#         frame_filename = f"{output_folder}/frame_{frame_number:04d}.jpg"
#         cv2.imwrite(frame_filename, frame)
#     cap.release()

# def extract_frames_parallel(video_path, fps=1, output_folder="frames_parallel"):
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = max(frame_rate // fps, 1)
    
#     timestamps = [(frame_number, video_path, output_folder) for frame_number in range(0, total_frames, frame_interval)]
    
#     st.write("🚀 Extracting frames in parallel...")
#     with Pool(processes=4) as pool:
#         pool.map(process_frame, timestamps)

#     st.success(f"✅ Extracted {len(timestamps)} frames in parallel.")
#     return output_folder

# # -------------------- 5️⃣ FFmpeg-Based Frame Extraction (Fastest Method) --------------------
# def extract_frames_ffmpeg(video_path, fps=1, output_folder="frames_ffmpeg"):
#     os.makedirs(output_folder, exist_ok=True)
#     os.system(f'ffmpeg -i {video_path} -vf "fps={fps}" {output_folder}/frame_%04d.jpg -hide_banner -loglevel error')
#     st.success(f"✅ Frames extracted and saved in {output_folder}.")
#     return output_folder

# # -------------------- 6️⃣ Main Execution --------------------
# def main():
#     st.title("🎥 YouTube Video Frame Extractor (Optimized for Speed)")

#     youtube_url = st.text_input("Enter YouTube Video URL:")
#     if st.button("Download & Extract Frames"):
#         if youtube_url:
#             video_path = download_youtube_video(youtube_url)
#             if video_path:
#                 video_resized = downscale_video(video_path)  # Step 1: Downscale video
#                 st.write("🔽 Select Frame Extraction Method:")
#                 method = st.radio("", ["OpenCV", "Multiprocessing", "FFmpeg (Fastest)"])

#                 if method == "OpenCV":
#                     extracted_folder = extract_frames(video_resized, fps=1)
#                 elif method == "Multiprocessing":
#                     extracted_folder = extract_frames_parallel(video_resized, fps=1)
#                 elif method == "FFmpeg (Fastest)":
#                     extracted_folder = extract_frames_ffmpeg(video_resized, fps=1)

#                 st.success(f"✅ Frames saved in: {extracted_folder}")

# if __name__ == "__main__":
#     main()



# forth version

import os
import cv2
import numpy as np
import streamlit as st
import yt_dlp
import nltk
import moviepy.editor as mp
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- Initialize --------------------
nltk.download('vader_lexicon')

# Load emotion classifier
emotion_dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

try:
    with open("face_emotions.json", "r") as json_file:
        loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("face_emotion.h5")
except Exception as e:
    st.error(f"⚠ Error loading emotion detection model: {e}")
    classifier = None

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------- 1️⃣ Download YouTube Video --------------------
def download_youtube_video(youtube_url):
    st.write("📥 Downloading YouTube Video using yt-dlp...")
    output_path = "downloaded_video.mp4"
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        st.success(f"✅ Video downloaded: {output_path}")
        return output_path
    except Exception as e:
        st.error(f"⚠ Error downloading video: {e}")
        return None

# -------------------- 2️⃣ Downscale Video --------------------
def downscale_video(input_video, output_video="temp_video_resized.mp4"):
    st.write("📉 Downscaling video to 720p...")
    os.system(f'ffmpeg -i "{input_video}" -vf "scale=1280:720" -preset ultrafast "{output_video}" -y -loglevel error')
    st.success("✅ Video downscaled.")
    return output_video

# -------------------- 3️⃣ Extract Audio & Perform Sentiment Analysis --------------------
def extract_audio(video_path):
    try:
        st.write("🎵 Extracting audio...")
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile("temp_audio.wav")
        st.success("✅ Audio extracted: temp_audio.wav")
        return "temp_audio.wav"
    except Exception as e:
        st.error(f"⚠ Error extracting audio: {e}")
        return None

def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    return "Positive sentiment" if score['pos'] > score['neg'] else "Negative sentiment" if score['neg'] > score['pos'] else "Neutral sentiment"

def analyze_audio_sentiment(audio_file):
    st.write("🎤 Analyzing audio sentiment...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        sentiment = sentiment_analyse(text)
        st.success(f"✅ Audio Sentiment: {sentiment}")
        return sentiment
    except:
        st.error("⚠ Error: Could not process audio.")
        return "Error"

# -------------------- 4️⃣ Extract Frames --------------------
def extract_frames_ffmpeg(video_path, fps=1, output_folder="frames_ffmpeg"):
    os.makedirs(output_folder, exist_ok=True)
    os.system(f'ffmpeg -i "{video_path}" -vf "fps={fps}" "{output_folder}/frame_%04d.jpg" -hide_banner -loglevel error')
    st.success(f"✅ Frames extracted and saved in {output_folder}.")
    return output_folder

# -------------------- 5️⃣ Facial Emotion Detection --------------------
def analyze_video_emotions(frames):
    st.write("🔍 Analyzing video frames for emotions...")
    results = []
    for frame in frames:
        try:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_image = img_gray[y:y+h, x:x+w]
                face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([face_image]) != 0:
                    face_image = face_image.astype('float') / 255.0
                    face_image = img_to_array(face_image)
                    face_image = np.expand_dims(face_image, axis=0)
                    prediction = classifier.predict(face_image)[0]
                    emotion = emotion_dict[int(np.argmax(prediction))]
                    results.append(emotion)
        except Exception as e:
            st.error(f"⚠ Error in emotion detection: {e}")

    if results:
        st.success(f"🎭 Detected emotions: {', '.join(set(results))}")
    else:
        st.warning("⚠ No faces detected.")

    return results

# Missing function definition (Fixed)
def analyze_emotion_sentiment(emotions_detected):
    if not emotions_detected:
        return "Neutral sentiment"
    positive_emotions = ["happy", "surprise"]
    negative_emotions = ["angry", "fear", "sad"]
    if any(emotion in positive_emotions for emotion in emotions_detected):
        return "Positive sentiment"
    elif any(emotion in negative_emotions for emotion in emotions_detected):
        return "Negative sentiment"
    return "Neutral sentiment"

# -------------------- 6️⃣ Fetch & Analyze YouTube Comments --------------------
def fetch_comments(youtube_url, api_key):
    st.write("💬 Fetching YouTube comments...")
    video_id = youtube_url.split("v=")[1].split("&")[0]
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    try:
        response = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText").execute()
        while response:
            comments.extend([item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]])
            if "nextPageToken" in response:
                response = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", pageToken=response["nextPageToken"]).execute()
            else:
                break
    except Exception as e:
        st.error(f"⚠ Error fetching comments: {e}")

    return comments

# -------------------- 7️⃣ Streamlit UI --------------------
def main():
    st.title("YouTube Video Analysis App 🎥")

    youtube_url = st.text_input("📌 Enter YouTube Video URL")
    api_key = st.text_input("🔑 Enter YouTube API Key", type="password")

    if youtube_url and api_key:
        st.write(f"📊 Processing video: {youtube_url}")

        video_path = download_youtube_video(youtube_url)
        if video_path:
            video_resized = downscale_video(video_path)
            frames_folder = extract_frames_ffmpeg(video_resized)

            frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".jpg")])
            frames = [cv2.imread(f) for f in frame_files]

            emotions_detected = analyze_video_emotions(frames)
            video_emotion_sentiment = analyze_emotion_sentiment(emotions_detected)

            audio_file = extract_audio(video_resized)
            audio_sentiment = analyze_audio_sentiment(audio_file) if audio_file else "Error"

            comments = fetch_comments(youtube_url, api_key)
            comment_sentiments = [sentiment_analyse(comment) for comment in comments] if comments else []

            final_sentiment = analyze_emotion_sentiment(emotions_detected)
            st.write(f"✅ **Final Video Sentiment:** {final_sentiment}")

if __name__ == "__main__":
    main()
