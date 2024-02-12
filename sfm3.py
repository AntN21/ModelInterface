import pyNetLogo
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t


def simulate(numbers_out, prop_personnes_polies, patience_gens_polis, nbr_runs, max_ticks):
    """
    numbers_out: liste dont les éléments sont le nombre de personnes sur le quai (qui veulent monter)
                qu'on doit tester
    prop_personnes_polies: proportions de personnes polies  (qui ne vont pas se coller initialement) à tester.
                entre 0 et 1
    patience_gens_polis: patience des gens polis initialement (quand les personnes polies restent immobile patience
                ticks, ils deviennent impolis)
    nbrs_runs: nombre de fois où la simulation tourne pour élément de numbers_out
    max_ticks: nombre max de ticks pour faire tourner les simulations

    return X, Y, R avec:
    X: liste qui vaut numbers_out
    Y: liste de liste avec la durée de la simulation (min (max_ticks, situation où tout le monde est monté))
    pour chacune des nbrs_runs simulations pour tous les éléments de numbers_out
    R: liste de liste avec le rayon-libre moyen pour chacune des nbrs_runs simulations pour tous les éléments
    de numbers_out

    """

    # netlogo = pyNetLogo.NetLogoLink(gui=True)  # Lance Netlogo.
    netlogo = pyNetLogo.NetLogoLink()

    netlogo_file = 'sfm2.nlogo'
    netlogo.load_model(netlogo_file)

    X, Y, R = [], [], []

    for number_out in numbers_out:
        print("\n", number_out)

        X.append(number_out)
        Y.append([])
        R.append([])

        for i in range(nbr_runs):
            seed = i
            print(f"run {i+1}/{nbr_runs}")
            t, rayon_libre_moyen = simulate_once(netlogo, number_out, prop_personnes_polies, patience_gens_polis, max_ticks, seed)
            Y[-1].append(t)
            R[-1].append(rayon_libre_moyen)

    netlogo.kill_workspace()

    X = np.array(X)
    Y = np.array(Y)
    R = np.array(R)
    return X, Y, R


def simulate_once(netlogo, number_out, number_in, A, D, delta, max_ticks):
    """
    Réalise une simulation avec un fichier netlogo.

    renvoie t, r avec :
    t: la durée de la simulation (min (max_ticks, situation où tout le monde est monté))
    r: rayon-moyen libre pendant la simulation
    """

    # On initialise.
    netlogo.command(f'set nb-peds {number_out}')
    netlogo.command(f'set nb-peds-in {number_in}')
    netlogo.command(f'set A {A}')
    netlogo.command(f'set D {D}')
    netlogo.command(f'set Tr {0.5}')
    netlogo.command(f'set delta {delta}')

    netlogo.command('setup')

    # On lance la simulation
    #netlogo.repeat_command("Setup", max_ticks)
    netlogo.repeat_command("Move", max_ticks)

    # t: le nombre de ticks pour que tout le monde soit monté.
    # pas toujours max_ticks car la simulation n'avance plus si tout le monde est monté.
    t = int(netlogo.report("time"))
    #rayon_libre_moyen = netlogo.report("rayon-libre-mean")

    return t





if __name__ == "__main__":
    nlogo = pyNetLogo.NetLogoLink(jvm_home = "C:/Program Files/NetLogo 6.3.0/runtime/lib" )
    lt = [simulate_once(nlogo, number_out=0, number_in=nbin, A=1, D=2.7) for nbin in range(10, 15)]
    plt.plot(range(10, 45), lt)
    plt.show()
    #show_graphs()
