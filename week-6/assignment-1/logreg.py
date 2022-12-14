#PART B


import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***

    #load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    print("sdfsdf")
    print(y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        #g = lambda x: 1 / (1 + np.exp(-x))


        x = np.column_stack((np.ones((x.shape[0], 1)), x))
        m, n = x.shape
        # assign theta to be a vector of zeros
        theta = np.zeros((n, 1))

        g = lambda x: 1 / (1 + np.exp(-x))

        # define the gradient of J(theta)
        def gradient(theta, x, y):
            return (1 / m) * x.T.dot(g(x.dot(theta)) - y)

        # define the hessian of J(theta)
        def hessian(theta, x, y):
            return (1 / m) * x.T.dot(np.diag(g(x.dot(theta)) * (1 - g(x.dot(theta))))).dot(x)
        
        # define the Newton's update
        def newton_update(theta, x, y):
            return theta - np.linalg.inv(hessian(theta, x, y)).dot(gradient(theta, x, y))
        
        # define the stopping criteria
        def stop_criterion(theta, x, y):
            return np.linalg.norm(gradient(theta, x, y)) < 1e-5
        
        # run Newton's method
        while not stop_criterion(theta, x, y):
            grad = gradient(theta, x, y)
            hess = hessian(theta, x, y)
            theta = newton_update(theta, x, y)

        # run newton's method
        while True:
            # compute gradient
            grad = (1 / m) * x.T.dot(g(x.dot(self.theta)) - y)

            # compute hessian
            hess = (1 / m) * x.T.dot(np.diag(g(x.dot(self.theta)).flatten()).dot(np.diag(1 - g(x.dot(self.theta)).flatten()))).dot(x)

            # compute update
            update = np.linalg.solve(hess, grad)

            # update theta
            theta = self.theta - update

            # check for convergence
            if np.linalg.norm(update) < 1e-5:
                break

        return theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        x = np.column_stack((np.ones((x.shape[0], 1)), x))
        #get prediction
        self.pred = g(x.dot(self.theta))
        

        # *** END CODE HERE ***

# call main function
if __name__ == '__main__':
    main(
        train_path = '/home/Machine Learning/week-6/assignment-1/dataset/dataset1_train.csv',
        eval_path = '/home/Machine Learning/week-6/assignment-1/dataset/dataset1_valid.csv',
        pred_path = ''
    )
