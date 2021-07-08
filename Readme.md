#Auto-accompaniment generation for vocal melody
##Usage
 * **offline** \
    Change wavfile_path here
    ```
    if is_offline:
    wavfile_path = "TEST/forgetme_2.wav"
      ```
 * **online** \
    Turn is_offline as False
    ```
    is_offline = False
    do_clean=True
    ```
## Hyper parameter
In **hparam.py** you can adjust tones, \
where 0 for C major 1 for D major and so on, and bpm \
    ```
    song_bpm = 60
    song_tone = 0
    ```
