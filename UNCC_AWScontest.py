import calendar

#packages for forecasting
import pandas
from pandas import read_csv
from pandas import DataFrame
import statsmodels.api as sm

#packages for optimization
import numpy as np
from gurobipy import *
import math
import random

print('Program initializing...\n')

#Place both the files in the working directory from where code will be executed
series = read_csv('TenYearDemand.csv', header=0, index_col=0,usecols=[1, 2])
series1 = read_csv('Ten-Year-Demand.csv', header=0, index_col=0,usecols=[1, 2])

#add 0 demand value for t = 0
series1 = pandas.DataFrame(np.array([[0]]), columns=['x']).append(series1, ignore_index=True)

my_order = (2, 1, 0)
my_seasonal_order = (1, 1, 0, 12)
numMonths = len(series1) - 1

X = series.values
Y = series1.values

train, test = X, Y
history = [x for x in train]

planAhead = [1]
programOutput = np.zeros((len(Y),5))

#number of scenarios
K = 300

#define intial inventory and order quantity
inventoryZero = 73

#index for totalCost vector
count = 0
for n in planAhead:
    print(n,'- month Ahead Strategy is being considered for inventory optimization.\n')
    monthIndex = 12
    yearIndex = 2005
    
    random.seed(49)
    history = [x for x in train]
    
    count = count + 1
    predictions = [0 for x in range(n)]
    var = [0 for x in range(n)]

    #define n-month ahead strategy, planning horizon, and number of scenarios
    T = n + 1

    #arrays for storing input/output for optimization
    demandScenarioData = np.zeros((T + 1, K))
    inventoryScenarioData = np.zeros((len(Y), 1))
    inventoryScenarioData[0] = inventoryZero
    orderQuantityData = np.zeros((len(Y), 1))
    
    for j in range(len(Y)-1):

        #FORECASTING MODULE - BEGIN
        
        #index used for optimization
        t = 1 + j
        
        model = sm.tsa.statespace.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order)
        model_fit = model.fit(disp=0)
        preds = model_fit.get_prediction(start=len(history),end=len(history)+n-1,full_results=False)

        pred_df = preds.summary_frame()
    
        obs = test[j]
        history.append(test[j+1])

        for i in range(0,n):
            var[i] = pred_df.loc[i, "mean_se"]
            predictions[i] = pred_df.loc[i, "mean"]
                
        #FORECASTING MODULE - END

        #OPTIMIZATION MODULE - BEGIN
        if K == 1:
            for i in range(1,T+1):
                if i == 1:
                    demandScenarioData[1] = obs
                else:
                    demandScenarioData[i] = random.normalvariate(predictions[i-2], var[i-2]**(1/2))
        else:        
            for k in range(K):            
                for i in range(1,T+1):
                    if i == 1:
                        demandScenarioData[1][k] = obs
                    else:
                        demandScenarioData[i][k] = random.normalvariate(predictions[i-2], var[i-2]**(1/2))
                        
        #continuous variables
        NV_inventoryLevel = K*(n + 1)
        NV_positiveInventory = K*(n + 1)
        NV_positiveInventoryGT90 = K*(n + 1)
        NV_orderQuantity = (n + 1)
        NV_backorderLevel = K*(n + 1)

        NV1 = 0
        NV2 = NV_inventoryLevel + NV_positiveInventory + NV_positiveInventoryGT90 + NV_orderQuantity + NV_backorderLevel
        NV = NV1 + NV2

        ###DEFINE VARIABLES - END

        ###DEFINE CONSTRAINTS - BEGIN

        #inequality
        NC_backOrderLevel = K*T
        NC_positiveInventory = K*T
        NC_positiveInventoryGT90 = K*T

        #equality constraints
        NC_inventoryBalance = K*T

        NC1 = NC_backOrderLevel + NC_positiveInventory + NC_positiveInventoryGT90
        NC2 = NC_inventoryBalance

        NC =  NC1 + NC2

        ###DEFINE CONSTRAINTS - END

        ###DEFINE A, sense, rhs, lb, ub, vtype, x - BEGIN

        A = [[0 for i in range(NV)] for j in range(NC)]
        x = [0]*NV

        rhs = [0 for i in range(NC)]
        sense = ['' for i in range(NC)]
        for i in range(1,NC+1):
            if i <= NC1:
                sense[i-1] = GRB.LESS_EQUAL
            else:
                sense[i-1] = GRB.EQUAL

        lb = [0 for i in range(NV)]
        for i in range(1,NV+1):
            if i <= NV_inventoryLevel:
                lb[i-1] = -1*math.inf
            
        ub = ['' for i in range(NV)]
        vtype = ['' for i in range(NV)]
        for i in range(1,NV+1):
            if i <= NV1:
                ub[i-1] = 1
                vtype[i-1] = GRB.BINARY
            else:
                ub[i-1] = GRB.INFINITY
                vtype[i-1] = GRB.CONTINUOUS

        ###DEFINE A, sense, rhs, lb, ub, vtype, x - END

        ###OBJ. FUNCTION - BEGIN

        objFunction = [0 for i in range(NV)]
        for i in range(1,T+1):
            for k in range(K):
                objFunction[(NV1 + NV_inventoryLevel + T*k + i) - 1] = 1
                objFunction[(NV1 + NV_inventoryLevel + NV_positiveInventory + T*k + i) - 1] = 1
                objFunction[(NV1 + NV_inventoryLevel + NV_positiveInventory + NV_positiveInventoryGT90 + NV_orderQuantity + T*k + i) - 1] = 3

        ###OBJ. FUNCTION - END
                    
        ###CONSTRAINTS - BEGIN

        m = 0
        #NC_backOrderLevel
        for i in range(1,T+1):
            for k in range(K):
                A[m][(NV1 + NV_inventoryLevel + NV_positiveInventory + NV_positiveInventoryGT90 + NV_orderQuantity + T*k + i) - 1] = -1
                A[m][(NV1 + T*k + i) - 1] = -1
                rhs[m] = 0
                m += 1

        #NC_positiveInventory
        for i in range(1,T+1):
            for k in range(K):
                A[m][(NV1 + NV_inventoryLevel + T*k + i) - 1] = -1
                A[m][(NV1 + T*k + i) - 1] = 1
                rhs[m] = 0
                m += 1

        #NC_positiveInventoryGT90
        for i in range(1,T+1):
            for k in range(K):
                A[m][(NV1 + NV_inventoryLevel + NV_positiveInventory + T*k + i) - 1] = -1
                A[m][(NV1 + T*k + i) - 1] = 1
                rhs[m] = 90
                m += 1

        #NC_inventoryBalance
        for i in range(1,T+1):
            for k in range(K):
                A[m][(NV1 + T*k + i) - 1] = 1
                if i == 1:
                    if K == 1:
                        rhs[m] = -1*demandScenarioData[1] + inventoryScenarioData[t-1] + orderQuantityData[t-1]
                    else:
                        rhs[m] = -1*demandScenarioData[1][k] + inventoryScenarioData[t-1] + orderQuantityData[t-1]
                else:
                    #inventory t-1
                    A[m][(NV1 + T*k + i - 1) - 1] = -1
                    # order quantity t-1
                    A[m][(NV1 + NV_inventoryLevel + NV_positiveInventory + NV_positiveInventoryGT90 + i - 1) - 1] = -1
                    if K == 1:
                        rhs[m] = -1*demandScenarioData[i]
                    else:
                        rhs[m] = -1*demandScenarioData[i][k]
                m += 1

        ###CONSTRAINTS - END

        ###CALL GUROBI SOLVER - BEGIN
            
        def dense_optimize(rows, cols, c, A, sense, rhs, lb, ub, vtype,
                           solution):

          model = Model()

          # Add variables to model
          vars = []
          for j in range(cols):
            vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

          # Populate A matrix
          for i in range(rows):
            expr = LinExpr()
            for j in range(cols):
              if A[i][j] != 0:
                expr += A[i][j]*vars[j]
            model.addConstr(expr, sense[i], rhs[i])

          # Populate objective
          obj = LinExpr()
          for i in range(cols):
            if c[i] != 0:
              obj += c[i]*vars[i]
          model.setObjective(obj)

          model.setParam('OutputFlag', 0)
          
          # Solve
          model.optimize()
            
          if model.status != GRB.INFEASIBLE:
            x = model.getAttr('x', vars)
            for i in range(cols):
              solution[i] = x[i]
            return True
          else:
              return False

        success = dense_optimize(NC, NV, objFunction, A, sense, rhs, lb, ub, vtype, x)
        
        print('Measuring order quantity for ',calendar.month_abbr[monthIndex],' ',yearIndex,'( t = ',t-1,') ...')
        
        ###CALL GUROBI SOLVER - END

        #store order quantity level for first month in the planning horizon
        orderQuantityData[t] = x[(NV1 + NV_inventoryLevel + NV_positiveInventory + NV_positiveInventoryGT90 + 1) - 1]

        #store end of month inventory level for first month in the planning horizon
        inventoryScenarioData[t] = x[(NV1 + 1) - 1]

        #OPTIMIZATION MODULE - END
        
        #generate output for beginning inventory level, order quantity, ending inventory level, holding costs, and backorder costs for each month
        programOutput[t-1][1] = orderQuantityData[t]
        if t == 1:
            programOutput[0][0] = 0
            programOutput[0][2] = inventoryZero
        else:
            programOutput[t-1][0] = programOutput[t-2][1] + programOutput[t-2][2]
            programOutput[t-1][2] = programOutput[t-1][0] - test[t-1]
            
        programOutput[t-1][3] = abs(programOutput[t-1][2]*(programOutput[t-1][2] > 0) + programOutput[t-1][2]*(programOutput[t-1][2] > 90))
        programOutput[t-1][4] = abs(3*programOutput[t-1][2]*(programOutput[t-1][2] < 0))

        monthIndex += 1
        if monthIndex == 13:
            monthIndex = 1
            yearIndex += 1
            
    print('t = ', t, 'is in reference to ', calendar.month_abbr[monthIndex],' ',yearIndex)        
    programOutput[t][0] = programOutput[t-1][1] + programOutput[t-1][2]
    programOutput[t][1] = 0
    programOutput[t][2] = programOutput[t][0] - test[t]
    programOutput[t][3] = abs(programOutput[t][2]*(programOutput[t][2] > 0) + programOutput[t][2]*(programOutput[t][2] > 90))
    programOutput[t][4] = abs(3*programOutput[t][2]*(programOutput[t][2] < 0))

    df = pandas.DataFrame(data=programOutput)
    df.columns = ['Initial Inv.', 'Order Quant.', 'End Inv.', 'Holding Costs', 'Backorder Costs']
    df = df.round(2)
    inventoryPositive = df.loc[df['End Inv.'] > 0, 'End Inv.'].sum() - inventoryZero*(inventoryZero > 0)
    inventoryNegative = abs(df.loc[df['End Inv.'] < 0, 'End Inv.'].sum()) - inventoryZero*(inventoryZero < 0)
    print('\nInventory optimization complete. Month-by-month reporting is displayed below:\n')
    print(df)
    print('\nTotal holding costs: $', round(df['Holding Costs'].sum(),2) - round(df.iat[0,3],2))
    print('Total backorder costs: $', round(df['Backorder Costs'].sum(),2) - round(df.iat[0,4],2))
    print('\nAverage monthly holding costs: $', (round(df['Holding Costs'].sum(),2)- round(df.iat[0,3],2))/numMonths)
    print('Average monthly backorder costs: $', (round(df['Backorder Costs'].sum(),2)- round(df.iat[0,4],2))/numMonths)
    print('\nTotal cost: $', round(df['Holding Costs'].sum(),2) + round(df['Backorder Costs'].sum(),2) - round(df.iat[0,3],2) - round(df.iat[0,4],2),'\n') 

print('Program terminated.')
