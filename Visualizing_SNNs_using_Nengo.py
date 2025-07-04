import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(4 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t))
    ens = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(stim, ens)

    out_probe = nengo.Probe(ens, synapse = 0.01)

    spike_probe = nengo.Probe(ens.neurons)


sim = nengo.Simulator(model)
sim.run(1.0)

plt.figure()
plt.plot(sim.trange(), sim.data[out_probe])
plt.xlabel("Time(s)")
plt.ylabel("Decoded Value")
plt.grid(True)
plt.title("Decoded output from Ensemble")
plt.show()