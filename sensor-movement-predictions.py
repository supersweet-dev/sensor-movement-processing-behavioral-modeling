# Jaime Alvarez Perez

# The Pandas library helps parse CSV files, or files delimited by specific character.
# In this case we use pandas to process our training and test data so we can easily feed it into our program.
import pandas as pd

# From tree we take the DTC class, which will create a decision tree trained with our data.
from sklearn.tree import DecisionTreeClassifier as dtc

# From metrics we use acuracy_score to test our outcome against our expectations.
from sklearn import metrics


# Some constant global variables, labels for our CSV since it has no headers,
# the labels we'll use for training, and an array to hold the performance scores
featureNames = ["sd_front", "sd_left"]
colNames = featureNames + ["move"]
scores = []

# A function to parse our datafiles using Pandas, returns a data frame. It works for training and test sets.
def getDataSet(filename):
    # We specify that there's no index column and feed our header names so the values process
    # as we expect them to.
    dataset = pd.read_csv(filename, index_col=None, header=None, names=colNames)
    classes = [
        "Move-Forward",
        "Slight-Right-Turn",
        "Sharp-Right-Turn",
        "Slight-Left-Turn",
    ]
    # We simplify our "label" field, which is our goal, using a simple function that returns a number representing one of the 4 possible moves
    dataset["move"] = dataset["move"].map(lambda n: classes.index(n) + 1)
    return dataset


# We call our get set function to generate the sets we'll be using through the program.
trainingset = getDataSet("sensor_readings_2_training.csv")
testset = getDataSet("sensor_readings_2_test.csv")

# We use the DecisionTreeClassifier constructor to generate a DTC using entropy to decide on splits.
# We feed it our training set and the associated goals to train it.
moveDecider = dtc(criterion="entropy").fit(trainingset[featureNames], trainingset.move)
# To test it, we feed it out test set and it returns an array of predictions.
testPredictions = moveDecider.predict(testset[featureNames])
# We compare our result to our expectations (the test set goals) and store the percentage in our score array.
scores.append(metrics.accuracy_score(testset.move, testPredictions))

print("Accuracy for intial Decision Tree:", scores[0])
if scores[0] == 1.0:
    print("a score of 1 means our predictions were perfect.")
