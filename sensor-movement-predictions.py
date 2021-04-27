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
featureNames2sensor = ["sd_front", "sd_left"]
featureNames4sensor = ["sd_front", "sd_left", "sd_right", "sd_back"]
featureNames24sensor = [("US"+ str(x)) for x in range(1, 24)]
scores = []

# A function to parse our datafiles using Pandas, returns a data frame. It works for training and test sets.
def getDataSet(filename, featureNames):
    # We specify that there's no index column and feed our header names so the values process
    # as we expect them to.
    colNames = featureNames + ["move"]
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

def runTest(numSensors, featureNames):
  # We call our get set function to generate the sets we'll be using through the program.
  trainingset = getDataSet("sensor_readings_" + str(numSensors) + "_training.csv", featureNames)
  testset = getDataSet("sensor_readings_" + str(numSensors) + "_test.csv", featureNames)

  # We use the DecisionTreeClassifier constructor to generate a DTC using entropy to decide on splits.
  # We feed it our training set and the associated goals to train it.
  moveDecider = dtc(criterion="entropy").fit(trainingset[featureNames], trainingset.move)
  # To test it, we feed it out test set and it returns an array of predictions.
  testPredictions = moveDecider.predict(testset[featureNames])
  # We compare our result to our expectations (the test set goals) and store the percentage in our score array.
  scores.append(metrics.accuracy_score(testset.move, testPredictions))

  
runTest(2, featureNames2sensor)
print("Accuracy for intial Decision Tree for 2 sensor data:", scores[0])
if scores[0] == 1.0:
    print("a score of 1 means our predictions were perfect.")
runTest(4, featureNames4sensor)
print("Accuracy for intial Decision Tree for 4 sensor data:", scores[1])
if scores[1] == 1.0:
    print("a score of 1 means our predictions were perfect.")
runTest(24, featureNames24sensor)
print("Accuracy for intial Decision Tree for 24 sensor data:", scores[2])
if scores[2] == 1.0:
    print("a score of 1 means our predictions were perfect.")
