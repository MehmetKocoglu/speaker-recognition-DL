import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf
import speech_recognition as sr

# Modelin yüklenmesi
model = load_model('model21.keras')  # Eğitilmiş modelin dosya yolu
def extract_features(sound_signal, sample_rate):
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=13)
    pad_width = 44 - mfcc_features.shape[1]
    if pad_width > 0:
        mfccs_scaled_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs_scaled_features = mfcc_features[:, :44]  # Gerekiyorsa MFCC'lerin boyutunu kırp
    return mfccs_scaled_features.reshape(1, 44, 13)

def start_recording_in_thread():
    thread = threading.Thread(target=start_recording)
    thread.start()

def start_recording():
    # Mikrofondan ses verisi almak için gerekli parametreler
    RATE = 44100
    DURATION = 5  # Kayıt süresi (saniye)

    # Kayıt durumu için label'ı güncelle
    status_label.config(text="Kayıt başladı...", fg="blue")

    # Ses kaydının geçici olarak kaydedilmesi
    sound_signal = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype='float32')
    sd.wait()  # Ses yakalamanın tamamlanmasını bekleyin

    # Kayıt durumu için label'ı güncelle
    status_label.config(text="Kayıt tamamlandı.", fg="green")

    # Ses kaydının geçici olarak kaydedilmesi
    temp_audio_file = "gecici_ses_kaydi.wav"
    sf.write(temp_audio_file, np.squeeze(sound_signal), RATE)

    # Ses verisinin özelliklere dönüştürülmesi
    mfccs_scaled_features = extract_features(np.squeeze(sound_signal), RATE)

    # Modelin kullanılması ve sonucun alınması
    result_array = model.predict(mfccs_scaled_features)

    # Tahmin edilen sınıflar
    result_classes = ['tahakocer','Julia_Gillard','Nelson_Mandela','Magaret_Tarcher','mehmetkocoglu','Benjamin_Netanyau','Jens_Stoltenberg']

    # En yüksek tahmin değeri ve indeksi
    max_index = np.argmax(result_array)
    max_value = result_array[0][max_index]
    predicted_class = result_classes[max_index]

    # Tahmin sonucunu gösterme
    if predicted_class == 'tahakocer':
        prediction_text = f"Tahmin edilen sanatçı: {predicted_class} (Tahmin Değeri: {max_value:.2f})"
    elif predicted_class == 'mehmetkocoglu':
        prediction_text = f"Tahmin edilen sanatçı: {predicted_class} (Tahmin Değeri: {max_value:.2f})"
    else:
        prediction_text = "Ses anlaşılamadı, tekrar deneyiniz."

    # Ses dosyasını metne çevirme işlemi
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='tr-TR')

    # Kelime sayısını bulma
    word_count = len(text.split())

    # Sonuçları gösterme
    prediction_textbox.config(state="normal")
    prediction_textbox.delete(1.0, tk.END)
    prediction_textbox.insert(tk.END, prediction_text)
    prediction_textbox.config(state="disabled")

    text_textbox.config(state="normal")
    text_textbox.delete(1.0, tk.END)
    text_textbox.insert(tk.END, text)
    text_textbox.config(state="disabled")

# Ana uygulama penceresi
root = tk.Tk()
root.title("Ses Kayıt ve Tanıma Uygulaması")

record_button = tk.Button(root, text="Kaydı Başlat", command=start_recording_in_thread)
record_button.pack()

status_label = tk.Label(root, text="Kayıt bekleniyor...", fg="black")
status_label.pack()

prediction_label = tk.Label(root, text="Tahmin Edilen Kişi ve Tahmin Değeri:")
prediction_label.pack()

prediction_textbox = tk.Text(root, height=2, width=50)
prediction_textbox.config(state="disabled")
prediction_textbox.pack()

text_label = tk.Label(root, text="Tanınan Metin:")
text_label.pack()

text_textbox = tk.Text(root, height=5, width=50)
text_textbox.config(state="disabled")
text_textbox.pack()

root.mainloop()
