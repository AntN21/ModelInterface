import pyNetLogo
import numpy as np
import json
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import t


def simulate(numbers_in, number_out, staying_in, staying_out, nbr_runs, max_ticks):
    """number_out: le nombre de personnes sur le quai (qui veulent monter)"""

    # Prend du temps de lancer et tuer Netlogo, si possible, ne le faire qu'une seule fois.

    # netlogo = pyNetLogo.NetLogoLink(gui=True)  # Lance Netlogo.
    netlogo = pyNetLogo.NetLogoLink()

    netlogo_file = 'descente.nlogo'
    netlogo.load_model(netlogo_file)

    X = []
    Y = []

    for number_in in numbers_in:
        print("\n", number_in)

        X.append(number_in)
        Y.append([])

        for i in range(nbr_runs):
            print(f"run {i+1}/{nbr_runs}")
            t = simulate_once(netlogo, number_in, number_out, staying_in, staying_out, max_ticks)
            Y[-1].append(t)

    netlogo.kill_workspace()

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def simulate_once(netlogo, number_in, number_out, staying_in, staying_out, max_ticks):

    netlogo.command(f'set number-in {number_in}')
    netlogo.command(f'set number-out {number_out}')
    netlogo.command(f'set staying-in {staying_in}')
    netlogo.command(f'set staying-out {staying_out}')
    netlogo.command('setup')

    netlogo.repeat_command("go", max_ticks)

    t = int(netlogo.report("end-time"))
    # nb_outside = netlogo.report("nb-outside")

    return t


if __name__ == "__main__":

    nbr_runs = 5
    numbers_in = range(10, 81, 10)
    number_out = 0
    staying_in = 0
    staying_out = 0
    max_ticks = 3000

    # t = simulate_once(netlogo, numbers_in, number_out, staying_in, staying_out, max_ticks)
    X, Y = simulate(numbers_in, number_out, staying_in, staying_out, nbr_runs, max_ticks)

    # Problème de conversion des numpy type en json
    x = [int(e) for e in X]
    y = [[int(y_i) for y_i in Y[i]] for i in range(len(Y))]

    data = {"X": x,
            "Y": y}

    with open(f'simu_descente.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    """
    sn.set()

    number_in = 50
    with open(f'simu_descente.json') as json_file:
        data = json.load(json_file)

    X = np.array(data["X"])
    Y = np.array(data["Y"])
    nbr_runs = Y.shape[1]

    intervals = []
    intervals_r = []
    for i in range(Y.shape[0]):
        interval = t.interval(0.95, nbr_runs - 1, Y.mean(axis=1)[i], Y.std(axis=1)[i])
        intervals.append(interval)

    borne_inf = np.array([interv[0] for interv in intervals])
    borne_sup = np.array([interv[1] for interv in intervals])

    plt.figure()
    plt.title(f"Nombre moyen de ticks avant que tout le monde soit monté pour \n"
              f"le nombre de personnes dedans au départ = {number_in})")

    plt.xlabel("Nombre de personnes qui sortent")
    plt.ylabel("Nombre de ticks avant que tout le monde soit monté")
    plt.plot(X, Y.mean(axis=1))
    plt.fill_between(X, borne_inf, borne_sup, color = 'b', alpha = .3, label = "intervalle de confiance à 95%")
    plt.legend()
    plt.savefig(f"number_in_{number_in}.png", dpi=400)
    plt.show()
    """
