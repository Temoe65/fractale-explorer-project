# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:57:14 2023

@author: mateo
"""

import numpy as np
import matplotlib.pyplot as plt
import Ma324_BE_lib_1st_part as m


#%% TEST FRACTALES ###########################################################

#Test polynome 1    
#a = complex(0.198, 8/13)

# Lapin de Douady
#a = complex(-0.00508, 0.33136)

# Newton   
#a = complex(0, 3**(0.5)/2)

#Test polynome 3  
a = complex(0.001,0.7)

#Test polynome 4 
#a = complex(0.05,0.9)

m.fractale(700,a,-0.5, 0.5)

#%% TEST LeKlik

# On définit les paramètres initiaux
a = complex(0.00001, 0.65)
n = 500
xmin, xmax = -0.5, 0.5

A = m.fractale(n, a, xmin, xmax)              # On génère la fractale initiale, avec les paramètres qu'on vient de définir
m.plot_fractal(A, xmin, xmax, xmin, xmax)     # Et on affiche la fractale en utilisant la fonction plot_fractal

plt.gcf().canvas.mpl_connect('button_press_event', m.LeKlik) # On associe l'événement de clic de souris à la fonction onclick
plt.show() # On affiche la fenêtre contenant la fractale et on attend les interactions de l'utilisateur

#%% TEST Vidéo 

a = complex(0.5,0.5)
x_centre = 0.012767504145
y_centre = -0.024603975125
m.video(a, 700, -20, 20, 100, x_centre=x_centre, y_centre=y_centre)

