"""
Delete comments for showing plot which you want
Do not forget to change read_csv's folder location..

@autor: Yusuf Samsum
@version: 11/09/2018
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print( '-----------------------------Checked---------------------------------------')

watsons_employee = pd.read_csv( '/home/stajyer/Desktop/project/Research/data.csv')

#print( watsons_employee.dtypes )

#print(watsons_employee.isnull().any() )


one_hot_encoded_watsons_employee = pd.get_dummies( watsons_employee )

#print( one_hot_encoded_watsons_employee.columns )

#print( "Attrition_No  INDEX---------------------------------------------------------------")

#print( watsons_employee.columns.get_loc( "EnvironmentSatisfaction" ) )

print( '-----------------------------Checked---------------------------------------')

#sns.countplot('PercentSalaryHike',data=watsons_employee)
#sns.countplot('MonthlyRate',data=watsons_employee)
#plt.show()
"""

Facetgrid = sns.FacetGrid( watsons_employee,hue='PercentSalaryHike',height=6)
Facetgrid.map(sns.kdeplot,'JobSatisfaction',shade=False)
Facetgrid.set(xlim=(0,watsons_employee['JobSatisfaction'].max()))
Facetgrid.add_legend()

Facetgrid = sns.FacetGrid( watsons_employee,hue='JobSatisfaction',height=6)
Facetgrid.map(sns.kdeplot,'PercentSalaryHike',shade=False)
Facetgrid.set(xlim=(0,watsons_employee['PercentSalaryHike'].max()))
Facetgrid.add_legend()


Facetgrid = sns.FacetGrid( watsons_employee,hue='Department',height=6)
Facetgrid.map(sns.kdeplot,'PercentSalaryHike',shade=False)
Facetgrid.set(xlim=(0,watsons_employee['PercentSalaryHike'].max()))
Facetgrid.add_legend()


#### Attrition Effects
over_time = sns.boxplot( x = 'Attrition_Yes', y = 'PercentSalaryHike', data = one_hot_encoded_watsons_employee)
over_time = sns.swarmplot( x = 'Attrition_Yes', y = 'PercentSalaryHike', data = one_hot_encoded_watsons_employee, color = ".20" )
"""

"""
### Department
departments = sns.lineplot( x = 'PercentSalaryHike', y = 'Department_Human Resources', data = one_hot_encoded_watsons_employee)


departments = sns.lineplot( x = 'PercentSalaryHike', y = 'Department_Research & Development', data = one_hot_encoded_watsons_employee)


departments = sns.lineplot( x = 'PercentSalaryHike', y = 'Department_Sales', data = one_hot_encoded_watsons_employee)
"""


"""
##RelationshipSatisfaction
sns.jointplot( x = 'RelationshipSatisfaction', y = 'PercentSalaryHike', data = one_hot_encoded_watsons_employee, kind = "hex")


##OverTime
over_time = sns.boxplot( x = 'OverTime', y = 'PercentSalaryHike', data = watsons_employee)
over_time = sns.swarmplot( x = 'OverTime', y = 'PercentSalaryHike', data = watsons_employee )


##TotalWorkingYears
sns.jointplot( x = 'TotalWorkingYears', y = 'PercentSalaryHike', data = one_hot_encoded_watsons_employee, kind = "hex")



total_working_years = sns.boxplot( x = 'TotalWorkingYears', y = 'PercentSalaryHike', data = watsons_employee )
total_working_years = sns.swarmplot( x = 'TotalWorkingYears', y = 'PercentSalaryHike', data = watsons_employee, color = ".20" )
"""

#sns.lineplot( x = 'PercentSalaryHike', y = 'TotalWorkingYears' , data = watsons_employee )

"""
##YearsAtCompany
sns.jointplot( x = 'YearsAtCompany', y = 'PercentSalaryHike', data = one_hot_encoded_watsons_employee, kind = "hex")


##MonthlyRate
sns.jointplot( x = 'MonthlyRate', y = 'PercentSalaryHike', data = watsons_employee, kind = "hex")

##JobInvolvement
sns.jointplot( x = 'JobInvolvement', y = 'PercentSalaryHike', data = watsons_employee, kind = "hex")

##EnvironmentSatisfaction

sns.jointplot( x = 'EnvironmentSatisfaction', y = 'PercentSalaryHike', data = watsons_employee, kind = "kde")

"""
##EducationField

#education_field = sns.boxplot( x = 'EducationField', y = 'PercentSalaryHike', data = watsons_employee)
#education_field = sns.swarmplot( x = 'EducationField', y  = 'PercentSalaryHike', data = watsons_employee, color = ".25" )

## Demonstration
plt.show()