# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:36:21 2023

@author: mateo
"""
# Voici un projet réalisé durant mes études d'ingénieurs (en 1e année du cycle ingénieur)
# Programme librairie utilisé par le programme Ma324_main.py


import matplotlib.pyplot as plt
import numpy as np
import cv2

#%% PARTIE 1 - Des fractales avec la méthode de Newton

# On définit la fonction polynomiale P(x, a)
def P(x, a):
    return (x-1)*(x+1/2-a)*(x+1/2+a)

# On définit la fonction dérivée P'(x, a) du polynôme P(x, a), calculée à la main
def P_prime(x, a):
    return (x+1/2-a)*(x+1/2+a)+(x-1)*(x+1/2+a)+(x-1)*(x+1/2-a)

# On définit la méthode de Newton pour trouver les racines du polynôme
def Newton(a, x0, erreur, max_iter = 50):
    x = x0      # On initialise x avec x0
    c = 0       # On initialise le compteur d'itérations

    while c < max_iter :                # On réalise des boucles tant qu'on ne dépasse pas max_iter
        p_x = P(x, a)                   # On calcule P(x) 
        p_prime_x = P_prime(x, a)       # Puis P'(x) sa dérivée
        xk = x - p_x / p_prime_x        # On met à jour la valeur de xk en utilisant la méthode de Newton

        if abs(xk - x) < erreur:        # On vérifie si la convergence est atteinte
            break                       # Auquel cas on stoppe la boucle

        x = xk                          # On met à jour la valeur de x
        c += 1                          # Et on incrémente le compteur
    return x

def fractale(n, a, xmin, xmax, show=True):  # On crée la fonction qui permet l'affichage des fractales
    A = np.zeros((n + 1, n + 1))            # Création d'une matrice carré et vide pour l'affichage final

    x_real = np.linspace(xmin, xmax, n+1)   # On crée les valeurs des coordonnées réelles de chaque point
    x_imag = np.linspace(xmin, xmax, n+1)   #          //            //           imaginaires      //
    X, Y = np.meshgrid(x_real, x_imag)      # On combine les deux listes pour créer la grille de point 
    x0 = X + 1j * Y                         # Chaque nombre complexe s'ecrira donc avec un élément de la matrice X (partie réelle) et un élément de la matrice Y (imaginaire) 

    x = x0.copy()                           # On réalise une copie de x0 dans x
    for _ in range(50):                     # Et on effectue 50 itérations
        p_x = P(x, a)                       # Ensuite, on calcule P(x) pour chaque point x de la grille
        p_prime_x = P_prime(x, a)           # On fait pareil avec P'(x)
        x -= p_x / p_prime_x                # Finalement, on met à jour x en utilisant la méthode de Newton

    # On calcule les distances entre x et les trois racines possibles
    d1 = np.abs(x - 1) 
    d2 = np.abs(x + 0.5 - a)
    d3 = np.abs(x + 0.5 + a)

    # Et en fonction des distances aux racines, on affecte des valeurs (entre 0 et 2) dans A 
    A[d1 < 1e-6] = 0
    A[d2 < 1e-6] = 1
    A[d3 < 1e-6] = 2
    A[(d1 >= 1e-6) & (d2 >= 1e-6) & (d3 >= 1e-6)] = 3   # Si aucune racine ne correspond, on affecte la valeur 3

    if show :     # On affiche la fractale si "show" est True, on met True par défaut
        plt.imshow(A, cmap='seismic', extent=[xmin, xmax, xmin, xmax])
        plt.show()

    return A


def video(a, n, xmin, xmax, nbiter, facteur=1.1, x_centre=None, y_centre=None): # On définit une fonction pour créer une vidéo montrant un zoom progressif sur la fractale
    A = np.zeros((n + 1, n + 1))           # Ensuite, on crée un tableau A pour stocker les futures valeurs de la fractale

    # D'abord, on initialise un objet "video" pour enregistrer la vidéo de la fractale avec OpenCV
    video = cv2.VideoWriter(
        "C:/Users/mateo/Downloads/video.avi",      # On indique le chemin et le nom du fichier vidéo qu'on enregistre
        cv2.VideoWriter_fourcc(*"XVID"),           # Et on spécifie le codec vidéo utilisé pour l'enregistrement (ici c'est XVID)
        10,                                        # On indique le nombre d'images par seconde qu'on veut (on a mis 10 FPS, nos PC ne sont pas très rapides)
        (np.shape(A)[1], np.shape(A)[0]))          # Enfin, on définit la résolution de la vidéo en fonction de la taille de A

    for i in range(1, nbiter + 1):                 # On réalise une boucle pour mettre chaque image de la vidéo
        zoom = facteur ** i                        # On incrémente le nouveau zoom 
        new_xmin, new_xmax = maj_limites(xmin, xmax, zoom, x_centre)  # On met à jour les limites de l'image en centrant sur la zone demandée 
        new_ymin, new_ymax = maj_limites(xmin, xmax, zoom, y_centre)  # Le tout à l'aide de la fonction maj_limites
        X = fractale(n, a, new_xmin, new_xmax, show=False)            # On crée la nouvelle image zoomée

        image = cv2.applyColorMap(np.uint8(255 * X / np.max(X)), cv2.COLORMAP_OCEAN)  # On convertit les données de la fractale en image couleur
        video.write(image)                         # On ajoute la nouvelle image à la vidéo
        print(i)
    video.release() # Une fois que toutes les images ont été ajouté à video, on ferme le fichier vidéo


def maj_limites(val_min, val_max, zoom, val_centre=None): #  Création d'une fonction qui met à jour les limites de l'image
    if val_centre is None:                         # Si la valeur du centre du zoom n'a pas été spécifié, alors on la calcule
        val_centre = (val_min + val_max) / 2       # En prennant la moyenne des des valeurs des bords de l'image (on zoom au centre en fait)
    plage = (val_max - val_min) / 2                # On calcule la nouvelle la plage de valeurs
    new_min = val_centre - plage / zoom            # Et avec cela, on calcule la nouvelle valeur minimale
    new_max = val_centre + plage / zoom            # Et puis la valeur maximale  
    return new_min, new_max                        # On return les nouvelles valeurs max 

#%% BONUS aide vidéo 

# Nous avons créé une fonction qui nous permet de zoomer manuellement sur une fractale pour trouver des points précis
# Nous garderons ces points précis pour créer des vidéos avec de grands zoom !

def LeKlik (event):        # Tout d'abord, on crée une fonction qui sera appelée lorsqu'on clique sur la figure, et qui zoomera sur la position du click
    global xmin, xmax      # On déclare xmin et xmax comme variables globales pour pouvoir les modifier à l'intérieur de la fonction
    x_centre, y_centre = event.xdata, event.ydata  # Ici, on récupère les coordonnées x et y du point cliqué à partir de l'objet event
                                                   # Ce point sera le centre de la nouvelle image ! (d'où l'appelation x_centre et y_centre)
                                                   
    zoom = 1.5   # Et puis on définit le facteur de zoom (1,5 c'est pas trop mal après plusieurs tests)

    plage_x = (xmax - xmin) / 2       # On calcule la moitié de la plage des valeurs x
    plage_y = (xmax - xmin) / 2       # Et aussi y 

    xmin = x_centre - plage_x / zoom  # On met à jour les limites de l'affichage en fonction du point cliqué et du facteur de zoom
    xmax = x_centre + plage_x / zoom
    ymin = y_centre - plage_y / zoom  # Avec ça, on réduit l'affichage de la valeur du facteur (si facteur = 2, alors on fait un zoom de *2)
    ymax = y_centre + plage_y / zoom
    
    A = fractale(n, a, xmin, xmax, show = False)     # Enfin, on génère la fractale avec les nouvelles limites
    plot_fractal(A, xmin, xmax, ymin, ymax)          # Et on l'affiche avec la fonction plot_fractale


# On définit donc une fonction pour afficher la fractale avec les limites qu'on a redéfinit avec LeKlik
def plot_fractal(A, x_min, x_max, y_min, y_max):     
    plt.clf()                                        # Déjà, on efface l'affichage précédent   
    
    # On affiche la fractale en utilisant la matrice A, une colormap et les limites définies
    plt.imshow(A, cmap='seismic', extent=[x_min, x_max, y_min, y_max], origin='lower')   
    
    # On a eu des problèmes avec les échelles des axes, alors on fixe le ratio d'aspect pour que les axes x et y aient la même échelle
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()  # Et puis il ne nous reste plus qu'à mettre à jour l'affichage
 
a = complex(0.00001, 0.65)  # Du fait qu'on demande n et a dans LeKlic, on les définit ici également pour ne pas avoir d'erreur 
n = 500
xmin, xmax = -0.5, 0.5