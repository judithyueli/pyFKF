from display_notebook import display_animation

#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

import param
CO2 = CO2simulation(param)
param.theta = (1.14,1e-5)
hikf, x_kf, cov_kf = simCO2.CO2_filter(CO2, param)
fig_setting = vco2.getImgParam(param)
# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(10):
#     x += np.pi / 15.
#     y += np.pi / 20.
    x_map = np.reshape(x_kf[i],(fig_setting['ny'],fig_setting['nx']), order = 'F')
    x_map = x_map[:,fig_setting['x_well1']:fig_setting['x_well2']]
    fig, (ax1,ax2) = plt.subplots(1,2)
    im = ax1.imshow(x_map)
    im1 = ax2.imshow(x_map)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
    repeat_delay=1000)

# call our new function to display the animation
display_animation(ani)