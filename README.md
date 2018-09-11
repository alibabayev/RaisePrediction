# RaisePrediction
----A more detailed report is available in the Documents.

Introduction

This project aims to investigate the relationship between the attributes given in the IBM HR Ana-
lytics Employee Attrition and Performance Sample Datasets. Using pandas, seaborn and matplotlib
libraries with Python, the project hopes to be able to classify the PercentSalaryHike with the
chosen attributes by the end of Data Exploration according to their individual effects into three
categories designated by the authors. Three classes defined are as the following

• Low: 11-15
• Medium:16-20
• High:21-25

Explaining the Dataset

2.1

The Data

The dataset consists of 1470 rows of fictional data prepared by IBM Data Scientists for utilizing
Machine Learning in order to predict Attrition. However in the scope of this project, Machine
Learning will be utilized in the same manner but the target value will be PercentSalaryHike and
its values vary between the values 11-25. For increasing the accuracy in the first place, the tar-
get data will be classified into three as Low(L),Middle(M),High(H) with the values given in the
Introduction section. All data will be used for visualization and understanding the relations,
but the attributes with no decent relationship with PercentSalaryHike will not be included in the
Prediction stage.

2.2

Attributes

There are 35 columns in the original dataset, three of them, Employee Number, Over18 and
PercentSalaryHike, will be discarded for the sake of this project since Percent Salary Hike is the
target value and Attrition is a set of experimental values. Attributes can be listed as the following:

• Age: Age of the employee

• Attrition: Did the employee decide to leave the company?

• BusinessTravel: How often does the employee travel for the company?

• DailyRate: Daily salary level of the employee

• Department: Department of the company that the employee is currently working in

• DistanceFromHome: How far is the home of the employee from the company?

• Education: Level of education of the employee based o graduate schools

• EducationField: Graduation Department

• EmployeeCount: How many employees does the employee work with?

• EmployeeNumber: ID of the employee

• EnvironmentSatisfaction: How satisfied is the employee from the company environment?

• Gender: Gender of the employee

• HourlyRate: Monthly salary level of the employee

• JobInvolvement: How involved is the employee with his/her job?

• JobLevel: Level of the job the emloyee is assigned

• JobRole: What is the employee working as within the job?

• JobSatisfaction: How satisfied is the employee with his/her job?

• MaritalStatus: Is the employee married?

• MonthlyIncome: Monthly salary of the employee

• MonthlyRate: Monthly salary rate of the employee

• NumCompaniesWorked: How many companies did the employee work with in the past?

• Over18: Is the employee over 18 years old?

• OverTime: Does the employee work overtime?

• PercentSalaryHike: Percentage of raise that the employee will get

• PerformanceRating: How does the employee perform?

• RelationshipSatisfaction: Is the employee satisfied with his/her relationship?

• StandardHours: Standard working hours of the employee

• StockOptionLevel: Stock options of the employee

• TotalWorkingYears: Experience of the employee

• TrainingTimesLastYear: How many times was the employee trained last year?

• WorkLifeBalance: Time spent between work and outside?

• YearsAtCompany: How long has the employee been working in the company?

• YearsInCurrentRole: How long has the employee been working in the current position?

• YearsSinceLastPromotion: How many years passed since the last promotion of the employee

• YearsWithCurrManager: How long has the employee been working with the current
manager?



