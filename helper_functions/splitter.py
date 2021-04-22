import os

root = "./dataset/original_datasets/kim-lee-2019/pos"
dest = "./dataset/split_dataset/kim-lee-2019/pos"

for vid in os.listdir(root):
    name, extension = vid.split(".")

    os.system(
        "ffmpeg -i '{root}/{vid}' -map 0 -codec copy -reset_timestamps 1 -f segment -segment_time 00:10 '{dest}/{name}-%03d.mp4'".format(
                                                                                                                                            root=root, 
                                                                                                                                            vid=vid,
                                                                                                                                            name=name,
                                                                                                                                            dest=dest
                                                                                                                                        )
    )