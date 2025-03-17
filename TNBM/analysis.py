from postprocessing import *
from functions import *
from main import main
import matplotlib.pyplot as plt

# dataset = get_ising(8)
# print(dataset.shape)

plot_BS()

# bond_dim = [2, 100, 600]
# filename = "variance_results.txt"
# plot_variance(filename, bond_dim)




# print(loss())




# plot_numbers_from_files("dag.txt", "nodag.txt")





# with open("./loss_values.dat", "a") as f:
#     for _ in range(100):
#         result = loss(main())
#         f.write(f"{result}\n")





# values = []
# with open("./results/BS/performace_stat/loss_values.dat", "r") as f:
#     data = f.readlines()
#     data = [float(i.strip()) for i in data]
#     values.append(data)
# values = np.array(values)
# print("Total values: ", values.shape[1])
# class1 = values[values < 0.001]
# class2 = values[(values >= 0.001) & (values < 0.0025)]
# class3 = values[(values >= 0.0025) & (values < 0.003)]
# class4 = values[(values >= 0.003) & (values < 0.004)]
# class5 = values[(values > 0.006)]

# print("Class 1: ", class1.shape[0])
# print("Mean: ", np.mean(class1))
# print("Variance: ", np.var(class1))
# print(class1)
# print()
# print("Class 2: ", class2.shape[0])
# print("Mean: ", np.mean(class2))
# print("Variance: ", np.var(class2))
# print(class2)
# print()
# print("Class 3: ", class3.shape[0])
# print("Mean: ", np.mean(class3))
# print("Variance: ", np.var(class3))
# print(class3)
# print()
# print("Class 4: ", class4.shape[0])
# print("Mean: ", np.mean(class4))
# print("Variance: ", np.var(class4))
# print(class4)
# print()
# print("Class 5: ", class5.shape[0])
# print("Mean: ", np.mean(class5))
# print("Variance: ", np.var(class5))
# print(class5)
# print()

# plt.figure()
# plt.title("Results 100 runs of training for 2000 iterations on BS dataset 9 qubits")
# plt.xlabel("Loss")
# plt.ylabel("Frequency")
# counts, bins, patches = plt.hist(values[0], bins=100)
# for count, bin in zip(counts, bins):
#     if count != 0:
#         plt.text(bin, count, str(int(count)), rotation=90, verticalalignment='bottom')

# plt.hist(values[0], bins=100)
# plt.savefig("stat.png")
