#
# Neural Network
#
#

# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Enable option to treat NAN and inf the same way
pd.options.mode.use_inf_as_na = True

# Read in data
examResults = pd.read_csv("../data/processed/results/ug_results.csv")
accreditationScore = pd.read_csv("../data/processed/college_accreditation_score.csv")

# Drop unnecessary columns
examResults = examResults.drop(['appeared_female', 'appeared_total', 'passed_female', 'passed_total', 'first_class_passed_female', 'first_class_passed_total', 'appeared_male', 'passed_male', 'first_class_passed_male'], axis=1)

# Merge exam results with accreditation score
df = pd.merge(examResults, accreditationScore, on=["id", "name"]).dropna()

# Set index
df = df.set_index(["id", "name"])

# Separate score
score = df["normalised_score"].to_frame()

# Remove score from data
df = df.drop(["normalised_score"], axis=1)

# Normalise all percentage based stats
for column in df.columns:
    df[column] = 0.01*df[column]

# Separate data for testing and training
testing, df = np.split(df, [500])
testingScore, score = np.split(score, [500])

# Initialise number of nodes in hidden layer
noOfNodes = 2

# Initialise random weights for first layer
weights1 = pd.DataFrame(np.random.uniform(0,1,size=(df.shape[1], noOfNodes)), columns=["node1", "node2"])
weights1["index"] = df.columns
weights1 = weights1.set_index(["index"])

# Initialise random weights for second layer
weights2 = pd.DataFrame(np.random.uniform(0,1,size=(noOfNodes, 1)), columns=["normalised_score"])
weights2["index"] = weights1.columns
weights2 = weights2.set_index(["index"])

# Initialise error and counter
err = 100
counter = 0

# Iterate till error is less than 10%
while abs(err) > 1:
    # Update counter
    counter = counter + 1

    # Calculate the output
    res1 = df@weights1
    res = res1@weights2

    # Calculate mean square error
    diff = score["normalised_score"].subtract(res["normalised_score"])
    diff = diff.to_frame()
    diff = diff.pow(2)
    err = diff.sum()
    err = err.values[0]/df.shape[0]
    err = math.sqrt(err)

    # Replace cells with infinite value with max value
    df.fillna(df.max(),inplace=True)
    diff.fillna(diff.max(),inplace=True)

    # Update weights in second layer
    weights2 = weights2 + 0.000000001*res1.T@diff

    # Calculate back propagation error
    backprop = diff@weights2.T

    # Update weights in first layer
    weights1 = weights1 + 0.000000001*df.T@backprop

    # Check if iteration limit is reached
    if counter > 5000:
        break

# Print number of iterations and training error
print("training iterations: ", counter)
print("training err: ", err)

# Print the weights
print(weights1)
print(weights2)

# Use leaned weights on testing data
testRes1 = testing@weights1
testRes = testRes1@weights2

# Calculate testing error
diff = testingScore["normalised_score"].subtract(testRes["normalised_score"])
diff = diff.to_frame()
diff = diff.pow(2)
err = diff.sum()
err = err.values[0]/testing.shape[0]
err = math.sqrt(err)

# Display testing error
print("testeerr; ", err)
