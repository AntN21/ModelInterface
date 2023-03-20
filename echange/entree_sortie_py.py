import pyNetLogo
import numpy as np
import json
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import t


def simulate(numbers, prop_personnes_impolies, prop_personnes_a_respecter, prop_personnes_assises, nbr_runs, max_ticks):
    """number_out: le nombre de personnes sur le quai (qui veulent monter)"""
    """number_in: le nombre de personnes dans le train (qui veulent descendre)"""
    """number : le nombre de personnes de chaque côté du quai (supposé pareil au début)"""

    # Prend du temps de lancer et tuer Netlogo, si possible, ne le faire qu'une seule fois.

    # netlogo = pyNetLogo.NetLogoLink(gui=True)  # Lance Netlogo.
    netlogo = pyNetLogo.NetLogoLink()
    netlogo_file = 'echange.nlogo'
    netlogo.load_model(netlogo_file)

    X, Y = [], []

    for number in numbers:
        print("\n", number)

        X.append(number)
        Y.append([])

        for i in range(nbr_runs):
            seed = i
            print(f"run {i+1}/{nbr_runs}")
            t = simulate_once(netlogo, number,  prop_personnes_impolies, prop_personnes_a_respecter, prop_personnes_assises, max_ticks, seed)
            Y[-1].append(t)

    netlogo.kill_workspace()

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def simulate_once(netlogo, number,  prop_personnes_impolies, prop_personnes_a_respecter, prop_personnes_assises, max_ticks, seed):

    netlogo.command(f'set number-out {number}')
    netlogo.command(f'set number-in {number}')
    netlogo.command(f'set staying-in {number*prop_personnes_assises}')
    netlogo.command(f'set staying-out {number*prop_personnes_assises}')
    netlogo.command(f'set seed {seed}')
    netlogo.command(f'set prop_personnes_impolis {prop_personnes_impolies}')
    netlogo.command(f'set prop_personnes_a_respecter {prop_personnes_a_respecter}')
    netlogo.command('setup')
    netlogo.repeat_command("go", max_ticks)

    # le nombre de ticks pour que tout le monde soit monté.
    # pas toujours 150 car la simu n'avance plus si tout le monde est monté.
    t = int(netlogo.report("ticks"))

    return t


if __name__ == "__main__":

    """
    nbr_runs = 20
    prop_personnes_impolies = 0.0 # proportion de personnes polies
    prop_personnes_a_respecter = 0.1  # proportion de personnes à respecter 
    prop_personnes_assises = 0.1
    numbers = range(10, 81, 10)
    max_ticks = 2000

    X, Y = simulate(numbers, prop_personnes_impolies, prop_personnes_a_respecter, prop_personnes_assises, nbr_runs, max_ticks)

    # Problème de conversion des numpy type en json
    x = [int(e) for e in X]
    y = [[int(y_i) for y_i in Y[i]] for i in range(len(Y))]

    data = {"prop_personnes_impolies": prop_personnes_impolies,
            "prop_personnes_a_respecter": prop_personnes_a_respecter,
            "X": x,
            "Y": y}

    with open(f'simu_poli_{prop_personnes_impolies}_respect_{prop_personnes_a_respecter}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    """
    
    sn.set()
    _Y_ = []
    for k in range(10):
            
        prop_personnes_impolies = str(k/10)
        prop_personnes_a_respecter = "0.1"
        prop_personnes_assises = "0.1"
        with open(f'simu_poli_{prop_personnes_impolies}_respect_{prop_personnes_a_respecter}.json') as json_file:
            data = json.load(json_file)

        prop_personnes_impolies = data["prop_personnes_impolies"]
        prop_personnes_a_respecter = data["prop_personnes_a_respecter"]
        X = np.array(data["X"])
        Y = np.array(data["Y"])
        Y_= []

        intervals = []

        for i in range(Y.shape[0]):
            if k == 5:
                Y_i = np.array([Y[i,j] for j in range(Y.shape[1]) if Y[i,j] < 4*np.min(Y[i,:])])
            else:
                Y_i = np.array([Y[i,j] for j in range(Y.shape[1]) if Y[i,j] < 3*np.min(Y[i,:])])
            Y_.append(Y_i.mean())
            interval = t.interval(0.95, Y_i.shape[0] - 1, Y_i.mean(), Y_i.std())
            intervals.append(interval)
            print(Y_)

        _Y_.append(Y_)
        
        """Y_i = np.array([Y[-2,j] for j in range(Y.shape[1]) ])
        Y_.append(Y_i.mean())
        print("Y_ mean : ", Y_i.mean())
        interval = t.interval(0.95, Y_i.shape[0] - 1, Y_i.mean(), Y_i.std())
        intervals.append(interval)

        Y_i = np.array([Y[-1,j] for j in range(Y.shape[1]) if Y[-1,j] < 4*np.min(Y[-1,:])])
        Y_.append(Y_i.mean())
        print("Y_ mean : ", Y_i.mean())
        interval = t.interval(0.95, Y_i.shape[0] - 1, Y_i.mean(), Y_i.std())
        intervals.append(interval)"""

        borne_inf = np.array([interv[0] for interv in intervals])
        borne_sup = np.array([interv[1] for interv in intervals])

    plt.figure()
    plt.title(f"Nombre moyen de ticks avant que tout le monde soit installé pour \n"
              f"une proportion de personnes à respecter  = {prop_personnes_a_respecter} et \n"
              f"une proportion de personnes assises = {prop_personnes_assises} ")

    plt.xlabel("Nombre de personnes qui montent et descendent")
    plt.ylabel("Nombre de ticks avant que tout le monde soit installé")

    for j in range(10):
        plt.plot(X, _Y_[j], label = str(j/10))
        #plt.fill_between(X, borne_inf, borne_sup, color = 'b', alpha = .3)

    plt.legend()
    plt.savefig(f"prop_impolies_2.png", dpi=400)
    plt.show()

    print("done")
