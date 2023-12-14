from prettytable import PrettyTable
import matplotlib.pyplot as plt

#function to plot table for CRR
def plot_CRR(originals, transforms1):
    ## convert the input data to 100% scale
    originals = originals*100
    transforms1 = transforms1*100
    
    ## construct the table
    table = PrettyTable()
    table.field_names = ["Similarity Measurement", "Original feature Set", "Reduced feature set (LDA)"]
    table.add_row(["L1 distance measure",originals[0],transforms1[0]])
    table.add_row(["L2 distance measure", originals[1], transforms1[1]])
    table.add_row(["Cosine similarity measure", originals[2], transforms1[2]])
    print("TABLE of Correct Recognition Rate (%) Using Different Similarity Measures")
    print(table)
    
#table to plot ROC curve
def plot_LDA_tunning(tunning_values, rates):
    fg = plt.figure(figsize=(5, 5))
    ax = fg.add_subplot(111)

    ax.set_xlabel("Dimensionality of the feature vector", size="large")
    ax.set_ylabel("Correct regonition rate", size="large")

    line_cv = ax.plot(tunning_values, rates, label="recognition rate", marker='o')

    ax.legend(loc="best", fontsize="large")
    plt.show()