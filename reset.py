import os
import shutil

target_dir = 'static/tmp'
shutil.rmtree(target_dir)
os.mkdir(target_dir)

f = open('static/tmp/.gitkeep', 'w')
f.close()