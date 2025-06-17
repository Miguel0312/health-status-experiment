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


files = [
    "results/2025_06_16-10_42_04.txt",
    "results/2025_06_16-10_49_56.txt",
    "results/2025_06_16-10_58_45.txt",
    "results/2025_06_16-11_08_29.txt",
]

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

    indicator = good_bad_ratio

    loss = list(csv.reader(StringIO(content[1])))[0]
    output.loss = list(map(float, loss))

    fdr = list(csv.reader(StringIO(content[2])))[0]
    output.fdr = list(map(lambda x: 100 * float(x), fdr))

    far = list(csv.reader(StringIO(content[3])))[0]
    output.far = list(map(lambda x: 100 * float(x), far))

    fdr_results[indicator] = []
    far_results[indicator] = []

    plt.plot(output.loss, label=f"{indicator}")

    for idx, x in enumerate(fdr):
        fdr_results[indicator].append((100 * (idx + 1), 100 * float(x)))

    for idx, x in enumerate(far):
        far_results[indicator].append((100 * (idx + 1), 100 * float(x)))

    index += 1
    # break

plt.title("Evolution of Loss Function for Different Values of Good Bad Ratio", pad=12.0)
plt.xlabel("Round")
plt.ylabel("Loss")
# plt.legend(title="Good Bad Ratio")
plt.legend(title="Hidden Nodes")

plt.savefig("tmp.png")

plt.figure(2, figsize=(10.8, 5.6))
plt.suptitle("Evolution of FAR and FDR for Different Values of Good Bad Ratio")

plt.subplot(121)

for val in fdr_results:
    fdr_results[val].sort()
    x = [u[0] for u in fdr_results[val]]
    y = [u[1] for u in fdr_results[val]]

    plt.plot(x, y, label=f"{val}")

plt.xlabel("Round")
plt.ylabel("FDR (%)")
plt.legend(title="Good Bad Ratio")
# plt.legend(title="Hidden Nodes")

# plt.ylim(97, 99)

plt.subplot(122)

for val in far_results:
    far_results[val].sort()
    x = [u[0] for u in far_results[val]]
    y = [u[1] for u in far_results[val]]

    plt.plot(x, y, label=f"{val}")

plt.xlabel("Round")
plt.ylabel("FAR (%)")
plt.legend(title="Good Bad Ratio")
# plt.legend(title="Hidden Nodes", loc="lower right")

plt.savefig("tmp2.png")
plt.show()
