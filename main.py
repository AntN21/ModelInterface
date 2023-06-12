import jpype
import pynetlogo as pyNetLogo
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

    netlogo_file = 'compression_montee.nlogo'
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
    netlogo_file = "C:/Users/ANT/Documents/Model interface RER/explo-master/sfm2.nlogo"
    netlogo.load_model(netlogo_file)
    # On initialise.
    netlogo.command('set Nb-peds ' + str(number_out))
    netlogo.command(f'set Nb-peds-in {number_in}')
    netlogo.command(f'set A {A}')
    netlogo.command(f'set D {D}')
    netlogo.command(f'set Tr {0.5}')
    netlogo.command(f'set delta {delta}')

    netlogo.command('setup')

    # On lance la simulation
    #netlogo.repeat_command("Setup", max_ticks)
    time = 0.
    time_1=-1.
    ticks =0
    netlogo.repeat_command('test',1000)
    while time != time_1 and ticks < max_ticks:
        netlogo.command("Move")
        time_1 = time
        time = netlogo.report("time")
        ticks += 1

    # t: le nombre de ticks pour que tout le monde soit monté.
    # pas toujours max_ticks car la simulation n'avance plus si tout le monde est monté.

    print(time)
    #rayon_libre_moyen = netlogo.report("rayon-libre-mean")

    return time

def save_simulations():
    """
    pas vraiment une fonction. paramètres à modifier à l'intérieur.
    enregistre des simulations en format json pour ne pas avoir à les refaire tourner.
    """

    # On fixe les paramètres qu'on veut enregistrer.
    nbr_runs = 10
    prop_personnes_polies = 1.0  # proportion de personnes polies
    patience_gens_polis = 10  # nombre de ticks où une personne peut rester sur place avant de devenir impolie.
    numbers_out = range(10, 81, 10)
    max_ticks = 300

    # On simule
    X, Y, R = simulate(numbers_out, prop_personnes_polies, patience_gens_polis, nbr_runs, max_ticks)

    # Problème de conversion des numpy type en json
    x = [int(e) for e in X]
    y = [[int(y_i) for y_i in Y[i]] for i in range(len(Y))]
    r = [[float(r_i) for r_i in R[i]] for i in range(len(R))]

    data = {"prop_personnes_polies": prop_personnes_polies,
            "patience_gens_polis": patience_gens_polis,
            "X": x,
            "Y": y,
            "R": r}

    # On enregistre les simulations.
    with open(f'simulations/simu_poli_{prop_personnes_polies}_patience_{patience_gens_polis}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def display_simulations():
    """
    pas vraiment une fonction non plus.
    Ouvre et affiche des données (ticks finaux et rayon moyen libre) selon le nombre de personnes qui veulent monter
    sur les simulations enregistrées.
    """

    sns.set()

    # On fixe la simulation qu'on veut afficher.
    prop_personnes_polies = "1.0"
    patience_gens_polis = "10"
    with open(f'simulations/simu_poli_{prop_personnes_polies}_patience_{patience_gens_polis}.json') as json_file:
        data = json.load(json_file)

    # On convertit les données
    prop_personnes_polies = data["prop_personnes_polies"]
    patience_gens_polis = data["patience_gens_polis"]
    X = np.array(data["X"])
    Y = np.array(data["Y"])
    R = np.array(data["R"])
    nbr_runs = Y.shape[1]

    # On crée des intervalles de confiance à l'aide d'une loi de Student.
    intervals = []
    intervals_r = []
    for i in range(Y.shape[0]):
        interval = t.interval(0.95, nbr_runs - 1, Y.mean(axis=1)[i], Y.std(axis=1)[i])
        intervals.append(interval)

        interval_r = t.interval(0.95, nbr_runs - 1, R.mean(axis=1)[i], R.std(axis=1)[i])
        intervals_r.append(interval_r)

    borne_inf = np.array([interv[0] for interv in intervals])
    borne_sup = np.array([interv[1] for interv in intervals])

    borne_inf_r = np.array([interv[0] for interv in intervals_r])
    borne_sup_r = np.array([interv[1] for interv in intervals_r])

    # On affiche et sauvegarde.
    plt.figure()
    plt.title(f"Nombre moyen de ticks avant que tout le monde soit monté pour \n"
              f"une proportion de personnes polies = {prop_personnes_polies} et \n"
              f"une patience des gens polis = {patience_gens_polis}")

    plt.xlabel("Nombre de personnes qui montent")
    plt.ylabel("Nombre de ticks avant que tout le monde soit monté")
    plt.plot(X, Y.mean(axis=1))
    plt.fill_between(X, borne_inf, borne_sup, color='b', alpha=.3, label="intervalle de confiance à 95%")
    plt.legend()
    # plt.savefig(f"images/prop_polie_{prop_personnes_polies}_patience_{patience_gens_polis}.png", dpi=400)
    plt.show()

    plt.figure()
    plt.title(f"Distance moyenne minimale avec une autre personne pour \n"
              f"une proportion de personnes polies = {prop_personnes_polies} et \n"
              f"une patience des gens polis = {patience_gens_polis}")

    plt.xlabel("Nombre de personnes qui montent")
    plt.ylabel("Distance moyenne minimale avec une autre personne")
    plt.plot(X, R.mean(axis=1))
    plt.fill_between(X, borne_inf_r, borne_sup_r, color='b', alpha=.3, label="intervalle de confiance à 95%")
    plt.legend()
    # plt.savefig(f"images/confort_prop_polie_{prop_personnes_polies}_patience_{patience_gens_polis}.png", dpi=400)
    plt.show()


def simulate_once_bis():
    """
    runs a simulation.
    Différent de simulate_once car affiche l'évolution selon les ticks du nombre de personnes montées
    et du rayon-libre moyen (entre les personnes) alors que l'autre renvoie juste le nombre de ticks final et
    le rayon-libre moyen moyen. (moyen selon le nombre de personnes et selon les ticks).
    """

    netlogo = pyNetLogo.NetLogoLink(gui=True)  # Lance Netlogo.
    # netlogo = pyNetLogo.NetLogoLink()

    netlogo_file = 'compression_montee.nlogo'
    netlogo.load_model(netlogo_file)

    netlogo.command(f'set number-out {80}')
    netlogo.command(f'set prop_personnes_polies {1}')
    netlogo.command(f'set patience_gens_polis {20}')
    netlogo.command(f'set seed {0}')
    netlogo.command('setup')

    X, T, P, R = [], [], [], []

    for t in range(281):
        X.append(t)

        T.append(netlogo.report("count turtles with [installed]"))
        P.append(netlogo.report("count turtles with [politesse]"))
        R.append(netlogo.report("mean [rayon-libre] of turtles"))  # modifie l'aléatoire.
        netlogo.command('go')

    netlogo.kill_workspace()

    sns.set()

    plt.figure()
    plt.plot(X, T)
    plt.xlabel("Nombre de ticks")
    plt.ylabel("Nombre de tortues dans le train")
    plt.title(
        "Nombre de tortues dans le train selon les ticks,\npour 80 personnes polies avec une patience de 20 ticks")
    plt.savefig("images/montee_ticks_80.png", dpi=400)
    plt.show()

    plt.figure()
    plt.plot(X, P)
    plt.xlabel("Nombre de ticks")
    plt.ylabel("Nombre de tortues polies")
    plt.title("Nombre de tortues polies selon les ticks,\npour 80 personnes polies avec une patience de 20 ticks")
    plt.savefig("images/poli_ticks_80.png", dpi=400)
    plt.show()

    plt.figure()
    plt.plot(X, R)
    plt.xlabel("Nombre de ticks")
    plt.ylabel("Moyenne du rayon-libre")
    plt.title("Moyenne de la distance minimale de la personne "
              "\nla plus proche des tortues selon les ticks,"
              "\npour 80 personnes polies avec une patience de 20 ticks")
    plt.savefig("images/rayon_libre_ticks_80.png", dpi=400)
    plt.show()


def show_graphs():
    """
    affiche les graphes avec les courbes superposées.
    """
    sns.set()

    # On ouvre.

    with open(f'simulations/simu_0.0_0.json') as json_file:
        data_0_0 = json.load(json_file)

    with open(f'simulations/simu_poli_0.5_patience_10.json') as json_file:
        data_05_10 = json.load(json_file)

    with open(f'simulations/simu_poli_1.0_patience_10.json') as json_file:
        data_1_10 = json.load(json_file)

    with open(f'simulations/simu_poli_0.5_patience_20.json') as json_file:
        data_05_20 = json.load(json_file)

    with open(f'simulations/simu_poli_1.0_patience_20.json') as json_file:
        data_1_20 = json.load(json_file)

    # On convertit.

    X_0_0 = np.array(data_0_0["X"])
    Y_0_0 = np.array(data_0_0["Y"])
    R_0_0 = np.array(data_0_0["R"])

    X_05_10 = np.array(data_05_10["X"])
    Y_05_10 = np.array(data_05_10["Y"])
    R_05_10 = np.array(data_05_10["R"])

    X_05_20 = np.array(data_05_20["X"])
    Y_05_20 = np.array(data_05_20["Y"])
    R_05_20 = np.array(data_05_20["R"])

    X_1_10 = np.array(data_1_10["X"])
    Y_1_10 = np.array(data_1_10["Y"])
    R_1_10 = np.array(data_1_10["R"])

    X_1_20 = np.array(data_1_20["X"])
    Y_1_20 = np.array(data_1_20["Y"])
    R_1_20 = np.array(data_1_20["R"])

    nbr_runs = Y_0_0.shape[1]

    def inter(Y, R):
        """
        ressort les bornes pour les intervalles de confiance
        """
        intervals = []
        intervals_r = []
        for i in range(Y.shape[0]):
            interval = t.interval(0.95, nbr_runs - 1, Y.mean(axis=1)[i], Y.std(axis=1)[i])
            intervals.append(interval)

            interval_r = t.interval(0.95, nbr_runs - 1, R.mean(axis=1)[i], R.std(axis=1)[i])
            intervals_r.append(interval_r)

        borne_inf = np.array([interv[0] for interv in intervals])
        borne_sup = np.array([interv[1] for interv in intervals])

        borne_inf_r = np.array([interv[0] for interv in intervals_r])
        borne_sup_r = np.array([interv[1] for interv in intervals_r])

        return borne_inf, borne_sup, borne_inf_r, borne_sup_r

    # On crée les bornes pour les intervalles de confiances.
    borne_inf_0_0, borne_sup_0_0, borne_inf_r_0_0, borne_sup_r_0_0 = inter(Y_0_0, R_0_0)
    borne_inf_05_10, borne_sup_05_10, borne_inf_r_05_10, borne_sup_r_05_10 = inter(Y_05_10, R_05_10)
    borne_inf_05_20, borne_sup_05_20, borne_inf_r_05_20, borne_sup_r_05_20 = inter(Y_05_20, R_05_20)
    borne_inf_10_10, borne_sup_10_10, borne_inf_r_10_10, borne_sup_r_10_10 = inter(Y_1_10, R_1_10)
    borne_inf_10_20, borne_sup_10_20, borne_inf_r_10_20, borne_sup_r_10_20 = inter(Y_1_20, R_1_20)

    ## Figure 1.

    plt.figure()
    plt.title(f"Nombre moyen de ticks avant que tout le monde soit monté")

    plt.xlabel("Nombre de personnes qui montent")
    plt.ylabel("Nombre de ticks avant que tout le monde soit monté")

    plt.plot(X_0_0, Y_0_0.mean(axis=1), color='b', label="prop polie = 0.0, patience = 0")
    plt.fill_between(X_0_0, borne_inf_0_0, borne_sup_0_0, color='b', alpha=.1)

    plt.plot(X_05_10, Y_05_10.mean(axis=1), color='orange', label="prop polie = 0.5, patience = 10")
    plt.fill_between(X_05_10, borne_inf_05_10, borne_sup_05_10, color='orange', alpha=.1)

    plt.plot(X_05_20, Y_05_20.mean(axis=1), color='green', label="prop polie = 0.5, patience = 20")
    plt.fill_between(X_05_20, borne_inf_05_20, borne_sup_05_20, color='green', alpha=.1)

    plt.plot(X_1_10, Y_1_10.mean(axis=1), color='red', label="prop polie = 1.0, patience = 10")
    plt.fill_between(X_1_10, borne_inf_10_10, borne_sup_10_10, color='red', alpha=.1)

    plt.plot(X_1_20, Y_1_20.mean(axis=1), color='purple', label="prop polie = 1.0, patience = 20")
    plt.fill_between(X_1_20, borne_inf_10_20, borne_sup_10_20, color='purple', alpha=.1)

    plt.legend()
    # plt.savefig(f"images/max_ticsk.png", dpi=400)
    plt.show()

    # Figure 2.

    plt.figure()
    plt.title(f"Distance moyenne minimale avec une autre personne")

    plt.xlabel("Nombre de personnes qui montent")
    plt.ylabel("Distance moyenne minimale avec une autre personne")

    plt.plot(X_0_0, R_0_0.mean(axis=1), color='b', label="prop polie = 0.0, patience = 0")
    plt.fill_between(X_0_0, borne_inf_r_0_0, borne_sup_r_0_0, color='b', alpha=.1)

    plt.plot(X_05_10, R_05_10.mean(axis=1), color='orange', label="prop polie = 0.5, patience = 10")
    plt.fill_between(X_05_10, borne_inf_r_05_10, borne_sup_r_05_10, color='orange', alpha=.1)

    plt.plot(X_05_20, R_05_20.mean(axis=1), color='green', label="prop polie = 0.5, patience = 20")
    plt.fill_between(X_05_20, borne_inf_r_05_20, borne_sup_r_05_20, color='green', alpha=.1)

    plt.plot(X_1_10, R_1_10.mean(axis=1), color='red', label="prop polie = 1.0, patience = 10")
    plt.fill_between(X_1_10, borne_inf_r_10_10, borne_sup_r_10_10, color='red', alpha=.1)

    plt.plot(X_1_20, R_1_20.mean(axis=1), color='purple', label="prop polie = 1.0, patience = 20")
    plt.fill_between(X_1_20, borne_inf_r_10_20, borne_sup_r_10_20, color='purple', alpha=.1)

    plt.legend()
    # plt.savefig(f"images/confort.png", dpi=400)
    plt.show()

def simulation(param_index, list_p,max_ticks= 30000, gui = False): #[nb-peds,nb-peds-in,delta,A,D,Tr]


    [nb_peds, nb_peds_in, delta, A, D, Tr] = [0,0,4,1,2,0.5]
    netlogo = pyNetLogo.NetLogoLink(gui=gui,jvm_path = "C:/Program Files/NetLogo 6.3.0/runtime/bin/server/jvm.dll")#"C:/Program Files/NetLogo 6.3.0/runtime/lib" )
    netlogo_file = "C:/Users/ANT/Documents/Model interface RER/explo-master/sfm2.nlogo"

    commands = ['set Nb-peds ','set nb-peds-in ', 'set delta ','set A ','set D ', 'set Tr ']
    #param_names = [""]
    netlogo.load_model(netlogo_file)
    # On initialise.
    rdoor=5
    netlogo.command(f'set rdoor {rdoor}')


    list_t = []
    for p in list_p:
        netlogo.command(commands[param_index]+str(p))
        netlogo.command('setup')

    # On lance la simulation
    # netlogo.repeat_command("Setup", max_ticks)
        time = 0.
        time_1 = -1.
        ticks = 0
        netlogo.repeat_command('test', 1000)
        while time != time_1 and ticks < max_ticks:
            netlogo.repeat_command("Move",1000)
            time_1 = time
            time = netlogo.report("time")
            ticks += 1000
        list_t.append(time)
    # t: le nombre de ticks pour que tout le monde soit monté.
    # pas toujours max_ticks car la simulation n'avance plus si tout le monde est monté.

    # rayon_libre_moyen = netlogo.report("rayon-libre-mean")
    plt.plot(list_p, list_t)
    plt.savefig(f"C:/Users/ANT/Desktop/simul/alea2_simul_rdoor={rdoor}_nbpeds={nb_peds}_nbpedsin={nb_peds_in}_A={A}_D={D}_Tr={Tr}_delta={delta}_{commands[param_index]}_{list_p[-1]}.pdf", dpi=400)
    plt.show()
    plt.close()


if __name__ == "__main__":
    #nlogo = pyNetLogo.NetLogoLink(gui=True,jvm_path = "C:/Program Files/NetLogo 6.3.0/runtime/bin/server/jvm.dll")#"C:/Program Files/NetLogo 6.3.0/runtime/lib" )
    #lt = [simulate_once(nlogo, number_out=0, number_in=nbin, A=1, D=2.7,delta = 5, max_ticks=10000) for nbin in range(10, 60)]
    #plt.plot(range(10, 60), lt)
    #plt.show()
    #show_graphs()
    #for rdoor in [3,4,6]:
    #    simulation(0, np.arange(50,301,25), gui=False)
    simulation(3,np.arange(0.8,3,0.1),gui=False)
