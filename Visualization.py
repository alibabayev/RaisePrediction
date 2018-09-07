import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as mp

dataset = pd.read_csv("data/HREmployeeDataset.csv")
sns.set_palette("Reds")
target = dataset.PercentSalaryHike

print("Dataset Size:",dataset.shape, end = "\n\n")

print("Column values are: ",list(dataset), end = "\n\n")

print("Showing some target values", target.head(),sep = "\n\n")

#PERCENT SALARY VS ATTRIBUTES

#First Attribute to evaluate with respect to PercentSalaryHike is Business Travel

print( "---First Attribute to evaluate is Business Travel with STRING values ---" ,dataset.BusinessTravel.head() ,sep = "\n")

first= sns.boxplot(x="BusinessTravel", y="PercentSalaryHike", data=dataset)
figure = first.get_figure()
#figure.savefig('BusinessTrav_plot.png', dpi=400)

#Second Attribute to evaluate with respect to PercentSalaryHike is Distance From Home


print( "---Second Attribute to evaluate is  Distance From Home with INT values ---" ,dataset.DistanceFromHome.head() ,sep = "\n")

mp.figure()

#second= dataset.plot.hexbin(x='DistanceFromHome', y='PercentSalaryHike',gridsize=15,cmap=plt.cm.Reds)
second = sns.lineplot(x='DistanceFromHome', y='PercentSalaryHike',data=dataset)
figure = second.get_figure()
figure.savefig('Distance2_plot.png', dpi=400)

#Third Attribute to evaluate with respect to PercentSalaryHike is Employee Count


print( "---Third Attribute to evaluate is  Employee Count with INT values ---" ,dataset.EmployeeCount.head() ,sep = "\n")

third = dataset.plot.line(x='EmployeeCount', y='PercentSalaryHike')

figure = third.get_figure()
#figure.savefig('EmployeeCount_plot.png', dpi=400)

#Fourth Attribute to evaluate with respect to PercentSalaryHike is Gender

print( "---Fourth Attribute to evaluate is Gender with STRING values ---" ,dataset. Gender.head() ,sep = "\n")

mp.figure()
fourth = sns.barplot(x="Gender", y="PercentSalaryHike", data=dataset)
figure = fourth.get_figure()
#figure.savefig('Gender_plot.png', dpi=400)
#Fifth Attribute to evaluate with respect to PercentSalaryHike is Job Level

print( "---Fifth Attribute to evaluate is Job Level with INT values ---" ,dataset. JobLevel.head() ,sep = "\n")

mp.figure()
fifth = sns.lineplot(x = "JobLevel", y = "PercentSalaryHike",data=dataset)
figure = fifth.get_figure()
#figure.savefig('JobLevel_plot.png', dpi=400)

#Sixth Attribute to evaluate with respect to PercentSalaryHike is Marital Status

print( "---Sixth Attribute to evaluate is  Marital Status with STRING values ---" ,dataset. MaritalStatus.head() ,sep = "\n")

mp.figure()

sixth = sns.barplot(x="MaritalStatus", y="PercentSalaryHike", data=dataset)
figure = sixth.get_figure()
#figure.savefig('Marital_plot.png', dpi=400)

#Seventh Attribute to evaluate with respect to PercentSalaryHike is NumCompaniesWorked

print( "---Seventh Attribute to evaluate is NumCompaniesWorked with INT values ---" ,dataset.NumCompaniesWorked.head() ,sep = "\n")

mp.figure()

seventh = sns.lineplot(x = "NumCompaniesWorked", y = "PercentSalaryHike",data=dataset)
figure = seventh.get_figure()
#figure.savefig('CompaniesWorked_plot.png', dpi=400)

#Eight Attribute to evaluate with respect to PercentSalaryHike is StandardHours


print( "---Eight Attribute to evaluate is  Standard Hours with STRING values ---" ,dataset.StandardHours.head() ,sep = "\n")

eight = dataset.plot.line(x='StandardHours', y='PercentSalaryHike')
figure = eight.get_figure()
#figure.savefig('StandardHours_plot.png', dpi=400)

#Ninth Attribute to evaluate with respect to PercentSalaryHike is TrainingTimesLastYear

print( "---Ninth Attribute to evaluate is  TrainingTimesLastYear with INT values ---" ,dataset.TrainingTimesLastYear.head() ,sep = "\n")


mp.figure()

ninth = sns.lineplot(x='TrainingTimesLastYear', y='PercentSalaryHike',data=dataset)
figure = ninth.get_figure()
#figure.savefig('TrainingTimes.png', dpi=400)

#Tenth Attribute to evaluate with respect to PercentSalaryHike is YearsInCurrentRole

print( "---Tenth Attribute to evaluate is YearsInCurrentRole with INT values ---" ,dataset.YearsInCurrentRole.head() ,sep = "\n")

mp.figure()

tenth = sns.lineplot(x='YearsInCurrentRole', y='PercentSalaryHike',data=dataset)
figure = tenth.get_figure()
#figure.savefig('YearsInCurrentRole_plot.png', dpi=400)

#Eleventh Attribute to evaluate with respect to PercentSalaryHike is YearsWithCurrManager

print( "---Eleventh Attribute to evaluate is YearsWithCurrManager with INT values ---" ,dataset.YearsWithCurrManager.head() ,sep = "\n")

mp.figure()

eleventh = sns.lineplot(x='YearsWithCurrManager', y='PercentSalaryHike',data=dataset)
figure = eleventh.get_figure()
#figure.savefig('YearsWithManager_plot.png', dpi=400)

mp.show()
