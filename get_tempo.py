import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os

def main() :
    for i in os.listdir(r"F:\work\video_analyze\separated\htdemucs\test") :
        audio_path = os.path.join(r"F:\work\video_analyze\separated\htdemucs\test",i)
        y,sr = librosa.load(audio_path)
        tempo,beats = librosa.beat.beat_track(y=y,sr=sr)


        print(tempo,beats)

        print(librosa.frames_to_time(beats, sr=sr))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr,aggregate=np.median)

        hop_length = 512
        
        fig, ax = plt.subplots( figsize = (20,8) , nrows=2, sharex=True)
        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max),y_axis='mel', x_axis='time', hop_length=hop_length,ax=ax[0])
        ax[0].label_outer()
        ax[0].set(title='Mel spectrogram')
        ax[1].plot(times, librosa.util.normalize(onset_env),
                label='Onset strength')
        ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',linestyle='--', label='Beats')
        ax[1].legend()
        plt.savefig("%s.png"%(i.replace(".wav","")))

        plt.close('all')


if __name__ == "__main__" :
    main()