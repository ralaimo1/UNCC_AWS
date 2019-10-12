# UNCC_AWS
Submission for the AWS competition - INFORMS 2019.
Running this code requires Python IDE (any 3.7 version) and a license for Gurobi 8.1.0. A number of packages have been used to design this model and it is important to install each of these packages before running the code. The list of packages needed are: 
calendar
pandas
statsmodels.api
numpy
gurobipy 
math
random
Please follow these instructions to run the code:
1. Place the file named “TenYearDemand.csv” provided during submission in the working directory without changing the name. This data contains the monthly demand values from January 1996 to December 2005 and is used for forecasting the unknown demand values from January 2006 to December 2007.
2. In the same directory, place the input file that will be used for validation (demand data from January 2006 to December 2007). It is assumed that this demand data will be structured exactly the same as the data that was provided for the competition, as specified in the problem description.
3. Change the file name that is in line 19 to the name of the file that will be used for the testing (currently set as “Ten-Year-Demand.csv”).
4. Run the program.
5. Output will be displayed in the format shown in the report in Section 6 for the 24 months.

This work was in collaboration between Richard Alaimo and Heena Jain from The University of North Carolina at Charlotte.
