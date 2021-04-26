import os

root = "./dataset/test_dataset/resized"
dest = "./dataset/test_dataset/split"

for vid in os.listdir(root):
    name, extension = vid.split(".")

    os.system("ffmpeg -i {root}/{vid} -c:v libx264 -crf 22 -map 0 -segment_time 10 -reset_timestamps 1 -g 300 -sc_threshold 0 -force_key_frames 'expr:gte(t,n_forced*10)' -f segment {dest}/{name}-%03d.mp4".format(root=root,vid=vid,name=name,dest=dest))
