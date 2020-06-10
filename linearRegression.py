#
# Linear regression
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

# Initialise random weights
weights = pd.DataFrame(np.random.uniform(0,1,size=(df.shape[1], 1)), columns=["normalised_score"])
weights["index"] = df.columns
weights = weights.set_index(["index"])

# Initialise error and counter
err = 100
counter = 0

# Iterate till error is less than 10%
while abs(err) > 1:
    # Update counter
    counter = counter + 1

    # Calculate score using weights
    res = df@weights

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

    # Update weights
    weights = weights + 0.00000001*df.T@diff

    # Check if iteration limit is reached
    if counter > 5000:
        break

# Print number of iterations and training error
print("training iterations: ", counter)
print("training err: ", err)

# Print the weights
print(weights)

# Use leaned weights on testing data
testRes = testing@weights

# Calculate testing error
diff = testingScore["normalised_score"].subtract(testRes["normalised_score"])
diff = diff.to_frame()
diff = diff.pow(2)
err = diff.sum()
err = err.values[0]/testing.shape[0]
err = math.sqrt(err)

# Display testing error
print("testeerr; ", err)
