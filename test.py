import subprocess
import processWhole as pg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

start_time = 73

# file = 'Bullets7Last5min.mp4'
# command = ('open', '-a', 'Quicktime Player 7', file)
# subprocess.Popen(command, shell=True)



# subprocess.Popen(["open", "/Users/zhuyuxi/PycharmProjects/viihdetest/Bullets7Last5min.mp4"])

# cast_names = pg.getCast('Bullets')
# text = ['juri borodin','juri borooodin','sherwan haji','zara','timo tuominen','juha puistola']

file_name = 'videos/Bullets1.mp4'

clipLen = 300
p = 929 - clipLen
ffmpeg_extract_subclip(file_name, p, p + clipLen, targetname='Bullets_EP7_5mins.mp4')