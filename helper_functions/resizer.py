import os
import moviepy.editor as mp

root = "./dataset/mivia/split/"
dest = "./dataset/mivia/split_resized/"

for vid_name in os.listdir(root):
    clip = mp.VideoFileClip(root + vid_name)
    clip_resized = clip.resize(newsize=(60,90))

    clip_resized.write_videofile(dest + vid_name)
