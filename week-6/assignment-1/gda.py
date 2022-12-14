# PART E

import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):

    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    clf.predict(x_train)

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        m, n = x.shape
        # assign theta to be a vector of zeros
        theta = np.zeros((n, 1))
        phi = np.mean(y)
        # set mu_0 and mu_1 to be the mean of x for each class
        mu_0 = np.mean(x[y == 0], axis=0)
        mu_1 = np.mean(x[y == 1], axis=0)
        # set sigma to be the covariance matrix of x
        sigma = np.cov(x.T)

        self.theta = np.linalg.inv(sigma).dot(mu_1 - mu_0)
        self.theta_0 = 0.5 * (mu_0.T.dot(np.linalg.inv(sigma)).dot(mu_0) - mu_1.T.dot(np.linalg.inv(sigma)).dot(mu_1)) + np.log(phi / (1 - phi))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        #result
        res = np.zeros((m, 1))
        # calculate the probability of each class

        for i in range(m):
            indic = x[i].dot(self.theta) + self.theta_0
            if indic >= 0:
                p_1 = 1 / (1 + np.exp(-indic))
            else:
                p_1 = np.exp(indic) / (1 + np.exp(indic))
            if res[i] == 0.5:
                res[i] = 1
            else:
                res[i] = 0

        return res

        # *** END CODE HERE

if __name__ == '__main__':
    main(
        train_path = '/home/Machine Learning/week-6/assignment-1/dataset/dataset1_train.csv',
        eval_path = '/home/Machine Learning/week-6/assignment-1/dataset/dataset1_valid.csv',
        pred_path = ''
    )