# -*- coding: utf-8 -*-
import sys
import pygame
import numpy as np
import random as rnd
from pygame.locals import *

# Fenêtre = 9 x 7 cases
#
# B = Balle
# G = Gardien
#
#
# (1,1)
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |===|===| G |===|===|   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   |   |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#    |   |   |   |   | B |   |   |   |   |
#    +---+---+---+---+---+---+---+---+---+
#                                       (9,7)
#

#######################
# Classes et fonctions #
#######################

def position_case(x, y):
    new_x = x
    new_y = y
    # Ajuster aux limites
    if (x < 1):
        new_x = 1
    if (x > 9):
        new_x = 9
    if (y < 1):
        new_y = 1
    if (y > 7):
        new_y = 9
    return ((new_x-1) * PPC, (new_y-1) * PPC)

def deplacer_x(pos, deplacement):
    return (pos[0] + deplacement, pos[1])

def deplacer_y(pos, deplacement):
    return (pos[0], pos[1] + deplacement)


class Balle:
    def __init__(self):
        self.reinitialiser()

    def reinitialiser(self):
        self.case = rnd.randint(1,9)
        self.x = position_case(self.case, 7)[0]
        self.y = position_case(self.case, 7)[1]

    def position(self):
        return (self.x, self.y)

    def avancer(self, facteur):
        if (self.y > 0):
            nouveau_y = self.y - facteur
            if nouveau_y <= 0:
                self.y = 0
            else:
                self.y = nouveau_y

    def est_dehors(self):
        return (self.case <= 2 or self.case >= 8)

    def est_dans_le_filet(self):
        return (self.case >= 3 and self.case <= 7)


class Gardien:
    def __init__(self):
        self.reinitialiser()

    def reinitialiser(self):
        self.x = POSITION_INITIALE_GARDIEN[0]
        self.y = POSITION_INITIALE_GARDIEN[1]
        self.case = 5

    def position(self):
        return (self.x, self.y)

    def est_dans_le_filet(self):
        return self.case >= 3 and self.case <= 7

    def pas_gauche(self, pas):
        if (self.case > 1):
            if (self.case - pas >= 1):
                self.case -= pas
            else:
                self.case = 1
            self.x = position_case(self.case, 1)[0]

    def pas_droite(self, pas):
        if (self.case < 9):
            if (self.case + pas <= 9):
                self.case += pas
            else:
                self.case = 9
            self.x = position_case(self.case, 1)[0]

    # Appliquer une valeur numérique d'action (entre 0 et 18) comme un mouvement
    # Les actions sont des pas du gardien (19 actions possibles)
    # Les mouvements négatifs sont à gauche, et positifs sont à droite
    #  action     =  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18
    #  mouvement = -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1   2   3   4   5   6   7   8   9
    def action_vers_mouvement(self, action):
        action_calculee = action - 9
        if (action_calculee == 0): # Ne pas bouger
            return
        elif (action_calculee < 0): # Bouger à gauche
            self.pas_gauche(abs(action_calculee))
        elif (action_calculee > 0): # Bouger à droite
            self.pas_droite(abs(action_calculee))
class Etat:
    def __init__(self, gardien, balle):
        self.gardien = gardien.case
        self.balle = balle.case

    def get_representation(self):
        return str(self.gardien) + str(self.balle)


def reinitialiser():
    global balle, gardien, ARRETS, POINTS, BUTS, DEHORS, EPISODES
    ARRETS = POINTS = BUTS = DEHORS = EPISODES = 0
    gardien.reinitialiser()
    balle.reinitialiser()
    print("\n#  REINITIALISATION\n")


##############
# Constantes #
##############

LARGEUR_CASES = 9
HAUTEUR_CASES  = 7
PIXELS_PAR_CASE = PPC =  100

LARGEUR_FENETRE = PPC * LARGEUR_CASES
HAUTEUR_FENETRE  = PPC * HAUTEUR_CASES
COULEUR_FENETRE = (10,175,30)

IPS = 60
FACTEUR_BALLE = 5

POSITION_INITIALE_GARDIEN = position_case(5, 1)
POSITION_INITIALE_BALLE = position_case(5, 7)
POSITION_INITIALE_FILE = position_case(3, 1)


#################
# État Global #
#################

ARRETS      = 0
BUTS         = 0
DEHORS         = 0
POINTS        = 0
TAUX_APPRENTISSAGE = 1.0
GAMMA         = 0.0
EPISODES     = 0


# Matrice Q comme dictionnaire d'états à vecteurs d'actions
#
# - Les états sont des instances de la classe `Etat`
# - Les actions sont des pas du gardien (19 actions possibles)
#     -  0-8  = pas à gauche
#     -   9   = ne pas bouger
#     - 10-18 = pas à droite
Q = {}


# Choisir la meilleure action disponible pour un état particulier
def meilleure_action(etat):
    if not (etat in Q):        # Cet état n'existe pas encore
        Q[etat] = [0] * 19     # Initialiser avec 0 pour toutes les actions
        return rnd.randint(0,18) # Retourner une action aléatoire
    else:
        return Q[etat].index(max(Q[etat]))


########################################
# Fenêtre, images et objets de jeu #
########################################

horlogeIPS = pygame.time.Clock()
pygame.init()
ecran = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
pygame.display.set_caption("La Finale")

image_balle = pygame.image.load('./img/balon.png')
image_balle = pygame.transform.scale(image_balle, (PPC, PPC))
image_gardien = pygame.image.load('./img/arquero.png')
image_gardien = pygame.transform.scale(image_gardien, (PPC, PPC))
image_filet = pygame.image.load('./img/red.png')
image_filet = pygame.transform.scale(image_filet, (5*PPC, PPC))

police = pygame.font.Font(None, 30)
police_petite = pygame.font.Font(None, 20)

balle = Balle()
gardien = Gardien()


######################
# Boucle d'Épisodes #
######################

while True:
    balle.reinitialiser()
    EPISODES += 1

    print(str(EPISODES) + " " + str(POINTS))

    #################
    # Choisir action #
    #################
    etat = Etat(gardien, balle).get_representation()
    action = meilleure_action(etat)
    gardien.action_vers_mouvement(action)
    nouvel_etat = Etat(gardien, balle).get_representation()

    while (balle.y > 0):
        for event in pygame.event.get():

            if (event.type == QUIT):
                pygame.quit()
                sys.exit()
            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_r):
                    reinitialiser()
                if (event.key == pygame.K_SPACE): # Turbo
                    if (IPS == 60):
                        IPS = 1000
                        FACTEUR_BALLE = 100
                    else:
                        IPS = 60
                        FACTEUR_BALLE = 5
                if (event.key == pygame.K_RIGHT):
                    gardien.pas_droite(1)
                if (event.key == pygame.K_LEFT):
                    gardien.pas_gauche(1)
                if (event.key == pygame.K_q):
                    exit()

        #################
        # Avancer jeu #
        #################

        balle.avancer(FACTEUR_BALLE)
        ecran.fill(COULEUR_FENETRE)
        ecran.blit(image_filet,     POSITION_INITIALE_FILE)
        ecran.blit(image_gardien, gardien.position())
        ecran.blit(image_balle,   balle.position())

        texte_buts = police.render('Buts: ' + str(BUTS), True, (255, 255, 255))
        texte_dehors = police.render('Dehors: ' + str(DEHORS), True, (255, 255, 255))
        texte_arrets = police.render('Arrêts: ' + str(ARRETS), True, (255, 255, 255))
        texte_points = police.render('Points: ' + str(POINTS), True, (255, 255, 255))
        texte_episodes = police.render('Penaltys: ' + str(EPISODES), True, (255, 255, 255))
        ecran.blit(texte_buts, (20, HAUTEUR_FENETRE - 30))
        ecran.blit(texte_dehors, (180, HAUTEUR_FENETRE - 30))
        ecran.blit(texte_arrets, (LARGEUR_FENETRE - 350, HAUTEUR_FENETRE - 30))
        ecran.blit(texte_points, (LARGEUR_FENETRE - 150, HAUTEUR_FENETRE - 30))
        ecran.blit(texte_episodes, (LARGEUR_FENETRE - 550, HAUTEUR_FENETRE - 30))

        pygame.display.update()
        horlogeIPS.tick(IPS)

        ########################
        # Règles de récompense #
        ########################

        recompense = 0
        # Balle hors du filet
        if (balle.est_dehors()):
            if (gardien.est_dans_le_filet()):
                recompense += 1 # Règle 4
            else:
                recompense -= 1 # Règle 3

        # Balle arrêtée
        elif (balle.est_dans_le_filet() and gardien.est_dans_le_filet() \
                and (balle.case == gardien.case)):
            recompense += 2 # Règle 1

        # But
        elif (balle.est_dans_le_filet() and (balle.case != gardien.case)):
            recompense -= 2 # Règle 2

        ##############
        # Apprendre Q #
        ##############
        if not (nouvel_etat in Q):
            Q[nouvel_etat] = [0] * 19
        Q[etat][action] += TAUX_APPRENTISSAGE * (recompense + GAMMA * max(Q[nouvel_etat]))

    ## Mettre à jour l'information à l'écran
    # Balle hors du filet
    if (balle.est_dehors()):
        DEHORS += 1
        if (gardien.est_dans_le_filet()):
            POINTS += 1 # Règle 4
        else:
            POINTS -= 1 # Règle 3

    # Balle arrêtée
    elif (balle.est_dans_le_filet() and gardien.est_dans_le_filet() \
            and (balle.case == gardien.case)):
        ARRETS += 1
        POINTS += 2 # Règle 1

    # But
    elif (balle.est_dans_le_filet() and (balle.case != gardien.case)):
        BUTS += 1
        POINTS -= 2 # Règle 2