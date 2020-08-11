from sklearn.model_selection import train_test_split
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd

from src.utils.context import Context


class RandomForest(RandomForestClassifier):
    def __init__(self, ctx, features, label, n_estimators=100):
        super(RandomForest, self).__init__(n_estimators=100)
        self.context = ctx
        self.logging = ctx.logger
        self.project_dir = ctx.project_dir
        self.data = ctx.data

        self.features = features
        self.label = label
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(self.data)

    def load_data(self, path):
        dataset = pd.read_csv(path, engine='c')
        X = dataset[self.features]
        y = dataset[self.label]
        return train_test_split(X, y, test_size=0.2)  # 80% training and 30% test

    def train(self):
        self.fit(self.X_train,self.y_train)

    def run(self):
        # Train the model using the training sets y_pred=self.predict()
        self.train()
        y_pred = self.predict(self.X_test)
        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
        # plt.barh(feat, clf.feature_importances_)
        # plt.yticks(fontsize=7)
        # plt.tight_layout()


if __name__ == "__main__":
    ctx = Context("test")

    if 'Employee' in ctx.data:
        features = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
                     'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                     'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                     'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                     'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                     'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                     'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                     'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                     'YearsWithCurrManager']
        label = ['Attrition']

    elif 'Civilian' in ctx.data:
        features = ['age', 'personnel_id', 'gender', 'race', 'marital_status', 'education',
                    'type_of_work', 'salary', 'distance_from_home',
                    'years_with_current_manager']
        label = ['suicide']

    else:
        ctx.logger.info("The input dataset does not find")
        exit(-1)

    engine = RandomForest(ctx, features, label)
    engine.run()