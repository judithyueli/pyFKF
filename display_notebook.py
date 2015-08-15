from tempfile import NamedTemporaryFile
from IPython.display import HTML
import matplotlib.pyplot as plt
import json
import matplotlib.pylab as pylab
from IPython.core.display import HTML
import sys
import os
import numpy as np

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
	if not hasattr(anim, '_encoded_video'):
		with NamedTemporaryFile(suffix='.mp4') as f:
			anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
			video = open(f.name, "rb").read()
		anim._encoded_video = video.encode("base64")
	
	return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
	plt.close(anim._fig)
	return HTML(anim_to_html(anim))

def reset_axis():
	pylab.rcParams['figure.figsize'] = 10, 5
	
def css_styling():
	"""Load default custom.css file from ipython profile"""
	s = json.load(open("styles/538.json"))
	plt.rcParams.update(s)
	reset_axis ()
	np.set_printoptions(suppress=True)
	directory = '.'
	name='styles/custom.css'
	styles = "<style>\n%s\n</style>" % (open(os.path.join(directory, name),'r').read())
	return HTML(styles)