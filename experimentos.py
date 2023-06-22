import sys
import re
import os

import pandas as pd
import numpy as np
import random as r
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

CLEANUP_STR = "\[|\',|\]|'"
plt.rcParams['hatch.linewidth'] = 0.5
style = {'Low_Intensity': '///', 'Med_Intensity': 'xxx', 'High_Intensity': '...'}

DETAILS = {
    'edp': [['low j',
             'high j'],
            ['low n',
             'high n']],
    'energy': [['low j',
                'high j'],
               ['low n',
                'high n']],
    'heuristic': [["random j",
                   "first j",
                   "shortest j",
                   "smallest j",
                   "low_mem j",
                   "low_mem_ops j"],
                  ["random n",
                   "first n",
                   "high_gflops n",
                   "high_cores n",
                   "high_mem n",
                   "high_mem_bw n",
                   "low_power n"]]
}


def get_style(row: pd.DataFrame) -> str:
    return style[row['profile']]


def random_color(row: pd.DataFrame) -> list:
    seed = int(row['id'])
    r.seed(seed)
    red = (r.randint(0, 16) + 16) / 32
    green = (r.randint(0, 2) + 1) / 8
    blue = (r.randint(0, 3) + 1) / 8

    return [red, green, blue]


def rename_res(row: pd.DataFrame) -> str:
    name = row['resources'].split('.')
    node = name[2][-1]
    proc = name[3][-1]
    core = name[4][-1]
    return node + ',' + proc + ',' + core


def load_results(jobs_file: str, simulation_file: str) -> tuple:
    jobs = pd.read_csv(jobs_file, sep=',')
    jobs['color'] = jobs.apply(random_color, axis=1)
    jobs['resources'] = jobs.apply(rename_res, axis=1)
    jobs['style'] = jobs.apply(get_style, axis=1)
    simulation = pd.read_csv(simulation_file, sep=',')
    simulation['edp'] = simulation['time'] * simulation['energy']
    return jobs, simulation


def generate_job_graph(jobs: pd.DataFrame):
    jobs = jobs.sort_values('resources')

    plt.barh(jobs['resources'], jobs['execution_time'], left=jobs['start_time'], color=jobs['color'],
             edgecolor='#000000', hatch=jobs['style'], linewidth=0.7)
    plt.xlabel("Time")
    plt.ylabel("Node, Processor, Core")

    jobs = jobs.sort_values('id')
    ids = {int(i.id): i.color for i in jobs.itertuples()}
    id_handle = [Patch(facecolor=ids[i], label=i) for i in ids]
    id_legend = plt.legend(title="Job id", handles=id_handle, loc=1)

    profs = {i.profile: i.style for i in jobs.itertuples()}
    profile_handle = [Patch(hatch=profs[i], label=i) for i in profs]
    plt.legend(title="Job profile", handles=profile_handle, loc=4)
    plt.gca().add_artist(id_legend)

    save_graph("jobs")


def generate_energy(simulation: pd.DataFrame):
    plt.plot('time', 'energy', data=simulation)
    plt.xlabel('Time(s)')
    plt.ylabel('Energy(J)')
    save_graph("energy")

    plt.plot('time', 'edp', data=simulation)
    plt.xlabel('Time(s)')
    plt.ylabel('Energy Delay Product')
    save_graph("edp")


def generate_comparison(values: dict, legend: list):
    x = np.arange(len(legend))
    for k, v in values.items():
        plt.bar(x, v)
        plt.xticks(x, legend)
        plt.ylim([min(v) * 0.85, max(v) * 1.15])
        plt.title(k)
        save_graph(options[3][0] + "/" + k)


def save_graph(name: str):
    plt.savefig(name + '.png', dpi=300)
    plt.clf()


def compare_simulations(simulations: list):
    print(simulations)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Error: must specify a file containing a valid experiment configuration"
        )

    options = [[], [], [], []]
    states = {'experiment:': 3, 'options:': 2, 'workload:': 1, 'platform:': 0, '---': -1}
    state = -1

    experiment_file = open(sys.argv[1], 'r')
    for line in experiment_file.readlines():
        if states.get(line.strip()) is not None:
            state = states.get(line.strip())
        elif state != -1:
            options[state].append(line.strip())

    platform_args = re.sub(CLEANUP_STR, '', str(options[0])).strip()
    workload_args = re.sub(CLEANUP_STR, '', str(options[1])).strip()

    try:
        os.mkdir(options[3][0])
    except:
        print("Experiment name already exists in directory")
        exit(1)

    # os.system('python3 ./generate_platform ' + platform_args)
    os.system('python3 ./generate_workload ' + workload_args)

    results = {'Time (s)': [], 'Energy (J)': [], 'Energy Delay Product': []}
    results_legend = []

    for option in options[2]:
        try:
            os.mkdir(options[3][0] + '/' + option + ' results')

            os.system('python3 ./generate_options ' + option)
            os.system('irmasim options.json')

            j, s = load_results("jobs.log", "simulation.log")

            os.replace("options.json", options[3][0] + '/' + option + ' results/options.json')
            os.replace("irmasim.log", options[3][0] + '/' + option + ' results/irmasim.csv')
            os.replace("jobs.log", options[3][0] + '/' + option + ' results/jobs.csv')
            os.replace("resources.log", options[3][0] + '/' + option + ' results/resources.csv')
            os.replace("simulation.log", options[3][0] + '/' + option + ' results/simulation.csv')

            if len(j) < 100:
                generate_job_graph(j)
                os.replace("jobs.png", options[3][0] + '/' + option + ' results/jobs.png')

            re = s[['time', 'energy', 'edp']].tail(1).values.tolist()[0]

            results['Time (s)'].append(re[0])
            results['Energy (J)'].append(re[1])
            results['Energy Delay Product'].append(re[2])

            params = option.split()
            results_legend.append(params[0]+"\n" +
                                  DETAILS[params[0]][0][int(params[1])]+"\n"
                                  + DETAILS[params[0]][1][int(params[2])])

        except Exception as e:
            print("WM already tested, skipping", e)

    generate_comparison(results, results_legend)
    # os.replace("platform.json", options[3][0] + '/platform.json')
    os.replace("workload.json", options[3][0] + '/workload.json')
