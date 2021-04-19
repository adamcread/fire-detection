import os
import moviepy.editor as mp

root = "./split_dataset/"

for vid_name in os.listdir(root):
    clip = mp.VideoFileClip(root + vid_name)
    clip_resized = clip.resize(newsize=(100,100))

    clip_resized.write_videofile("./resized_dataset/" + vid_name)
