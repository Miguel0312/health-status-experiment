from pathlib import Path
import ast
from io import StringIO
import csv
from dataclasses import dataclass
import dataclasses
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.rcsetup

# matplotlib.use("webagg")


@dataclass
class ExperimentOutput:
    config: dict = dataclasses.field(default_factory=dict)

    loss: list[float] = dataclasses.field(default_factory=list)
    fdr: list[float] = dataclasses.field(default_factory=list)
    far: list[float] = dataclasses.field(default_factory=list)


files = ["results/2025_06_09-12_55_50.txt", "results/2025_06_09-12_57_35.txt", "results/2025_06_09-12_59_21.txt", "results/2025_06_09-13_56_52.txt", "results/2025_06_09-13_58_30.txt", "results/2025_06_09-14_00_09.txt"]

x = []
y = []

colors = ["blue", "red", "green", "cyan", "orange"]
index = 0

fdr_results = {}
far_results = {}

plt.figure(1)

for file in files:
    content = Path(file).read_text().replace("\n", "").split("#")

    output = ExperimentOutput()

    output.config = ast.literal_eval(content[0])
    good_bad_ratio: float = output.config["good_bad_ratio"]
    hidden_nodes: float = output.config["hidden_nodes"]

    loss = list(csv.reader(StringIO(content[1])))[0]
    output.loss = list(map(float, loss))

    fdr = list(csv.reader(StringIO(content[2])))[0]
    output.fdr = list(map(lambda x: 100 * float(x), fdr))

    far = list(csv.reader(StringIO(content[3])))[0]
    output.far = list(map(lambda x: 100 * float(x), far))

    first = False

    if hidden_nodes not in fdr_results:
        fdr_results[hidden_nodes] = []
        far_results[hidden_nodes] = []
        first = True

    if first:
        plt.plot(output.loss, label = f'{hidden_nodes}')
    # TODO: save the interval when serializing
    # ax1.plot([100 * (i + 1) for i in range(len(fdr))], output.fdr, color=colors[index], linestyle="", marker="o" ,label=f'{output.config["hidden_nodes"]}')
    # ax2.plot([100 * (i + 1) for i in range(len(far))], output.far, color=colors[index], linestyle = "", marker="+", label=f'{output.config["hidden_nodes"]}')


    for idx, x in enumerate(fdr[0 : 100 if first else -1]):
        fdr_results[hidden_nodes].append(((1000 if first else 200) * (idx + 1), 100 * float(x)))

    for idx, x in enumerate(far[0 : 100 if first else -1]):
        far_results[hidden_nodes].append(((1000 if first else 200) * (idx + 1), 100 * float(x)))

    index += 1
    # break

plt.title("Evolution of Loss Function for Different Values of Hidden Nodes", pad=12.0)
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend(title = "Hidden Nodes")

plt.figure(2)
plt.suptitle("Evolution of FAR and FDR for Different Values of Hidden Nodes")

# plt.yticks([0,1,2,3,4])

plt.subplot(121)

for val in fdr_results:
    fdr_results[val].sort()
    x = [u[0] for u in fdr_results[val]]
    y = [u[1] for u in fdr_results[val]]

    plt.plot(x, y, label=f'{val}')

# plt.title("Evolution of FDR for Different Values of Good To Bad Ratio", pad=12.0)
plt.xlabel("Round")
plt.ylabel("FDR (%)")
plt.legend(title = "Hidden Nodes")

# plt.ylim(97, 99)

plt.subplot(122)

for val in far_results:
    far_results[val].sort()
    x = [u[0] for u in far_results[val]]
    y = [u[1] for u in far_results[val]]

    plt.plot(x, y, label=f'{val}')

plt.xlabel("Round")
plt.ylabel("FAR (%)")
plt.legend(title = "Hidden Nodes")

plt.show()