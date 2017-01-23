import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Classification:

    def __init__(self, X_tr, y_tr, X_te, y_te, classifier_name='' ):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.y_pred = np.array([])
        self.classifier_name = classifier_name

    def select_classifier(self, classifier_name, **kwargs):
        if classifier_name == "randomforest":
            classifier = RandomForestClassifier(**kwargs)
        elif classifier_name == "decisiontree":
            classifier = DecisionTreeClassifier(**kwargs)
        elif classifier_name == "logreg":
            classifier = LogisticRegression()
        else:
            raise NameError("Wrong input classifier_name")

        self.classifier = classifier
        return classifier

    def prediction(self, classifier_name='', **kwargs):
        self.classifier_name = classifier_name
        self.select_classifier(classifier_name, **kwargs)
        self.classifier.fit(self.X_tr, self.y_tr)
        self.y_pred = self.classifier.predict(self.X_te)


    def cross_validation(self, classifier_name="", scoring_method='accuracy', **kwargs):
        self.select_classifier(classifier_name, **kwargs)
        score = sklearn.cross_validation.cross_val_score(self.classifier, self.X_tr, self.y_tr, scoring = scoring_method)
        print("average ", scoring_method, "over three instances =", np.mean(score), "; standard deviation:", np.std(score))
        print(score)