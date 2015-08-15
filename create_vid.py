from moviepy.editor import *

clip = (VideoFileClip("co2estimatefinal.avi")
        .subclip((0,1),(0,10))
        .resize(0.99))
clip.write_gif("use_your_head.gif")