import matplotlib.pyplot as plt

values = []
with open("nohup.out", "r") as file:
    for line in file:
        values.append(float(line.split()[0]))  # Extract first value

# Plot
plt.plot(values[:1000], marker=",", linestyle="-")
plt.xlabel("Iterations")
plt.ylabel("MMD Loss")
plt.title("Training Tensor Network Born Machine with MMD Loss \n - 1000 Iterations - 9 qubits")
plt.yscale("log")  # Set y-axis to log scale
plt.grid()
plt.savefig("output.png", dpi=300)