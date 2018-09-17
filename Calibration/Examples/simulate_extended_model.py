import os
import numpy as np
from modelicares import SimRes
from dymola.dymola_interface import DymolaInterface

# Start the interface
dymola = DymolaInterface()

# AixLib directory
dir_Aix = os.path.join(r'D:\04_Git\AixLib_development\AixLib\AixLib\package.mo')

# Use this string to construct the directory where to store the simulation result
dir_res = r'D:\01_Dymola_WorkDir\00_Testzone\pyInterface'
dir_res_stor = dir_res + r'\dsres'
dir_res_open = dir_res_stor + r'.mat'

# Use this path to access the model to be used in this study
path = 'AixLib.Fluid.HeatPumps.Examples.HeatPumpDetailed'
# Initial names
init_names = ["heatPump.dataTable.tableQdot_con[2, 2]", "heatPump.dataTable.tableQdot_con[2, 3]"]
init_values = [4800, 6300]
# Name of the controlled variable
sim_outputs_to_check = ['heatPump.T_conOut.T', 'heatPump.P_eleOut']
# Change working directory for python and dymola instance
os.chdir(dir_res)
dymola.cd(dir_res)
# Open AixLib and store the returned Boolean indicating successful opening
check1 = dymola.openModel(dir_Aix, changeDirectory=False)
# Translate the model
dymola.experimentSetupOutput(events=False)
check2 = dymola.translateModel(path)

sim_success, sim_final_results = dymola.simulateExtendedModel(
    problem=path,
    startTime=0.0,
    stopTime=20000,
    outputInterval=1,
    method="Dassl",
    tolerance=0.0001,
    resultFile=dir_res_stor,
    initialNames=init_names,
    initialValues=init_values,
    finalNames=sim_outputs_to_check
)

print('Was simulation successfully? --> ' + str(sim_success))
print(sim_final_results)

# Get the simulation result directly from file
sim = SimRes(dir_res_open)

# Extract trajectory... HIER NOCH LOOPEN LASSEN ODER AEHNLICHES
# OLD: values_res = np.column_stack((sim[sim_outputs_to_check[0]].values(), sim[sim_outputs_to_check[1]].values()))
values_res = sim.to_pandas(sim_outputs_to_check)


if dymola is not None:
    dymola.close()
    dymola = None
