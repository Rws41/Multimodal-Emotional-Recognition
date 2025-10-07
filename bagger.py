import numpy as np
from collections import Counter
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
class bagger_model(object):
    def __init__(self, learner, kwargs={}, bags=20):
        """
        Constructor method
        """
        self.bags = bags
        self.learners = []
        self.trained_learners = []
        # This should create a list of learner instances.
        for i in range(bags):
            self.learners.append(learner(**kwargs))


        pass

    def add_data(self, x_train, y_train):
        filled_bags = []
        y_train = np.array(y_train)
        for _ in range(self.bags):
            targets = np.random.choice(x_train.shape[0], x_train.shape[0], replace=True)
            bag_x = x_train[targets]
            bag_y = y_train[targets]
            filled_bags.append((bag_x, bag_y))

        for index, learner in enumerate(self.learners):
            learner.fit(filled_bags[index][0], filled_bags[index][1])
        return
    def predict(self, x_test, y_test):
        predictions = np.zeros((x_test.shape[0], len(self.learners)), dtype=object)

        for index, learner in enumerate(self.learners):
            predictions[:, index] = learner.predict(x_test)

        most_common = []
        for row in predictions:
            most_common_class = Counter(row).most_common(1)[0][0]
            most_common.append(most_common_class)

        accuracy = self.scoring(most_common, y_test)
        return accuracy

    def scoring(self, predict, y):
        correct = 0
        total = len(y)
        for row in zip(predict,y):
            if row[0] == row[1]:
                correct +=1
        return correct/total






