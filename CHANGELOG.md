**v.0.1**:

- **v0.1.0**: Implemented.
- **v0.1.1**: Split into different frameworks and adjust changes from based on new version of ebcpy
- **v0.1.2**: Move CalibrationClass from ebcpy and add it to the general module aixcalibuha. Adjust Goals etc. based on changes in ebcpy.
- **v0.1.3**: Remove Continuous Calibration methods and introduce new, better methods for calibration of multiple classes.

   - Issue 43: Same class now optimizes to one optimum instead of multiple. If an intersection in tuner parameters occurs, the statistics are logged and plotted so the user can better decide with what values to go on.
   - Issue 42: Visualizer is adjusted to better print the results more readable
   - Issue 39: Several kwargs are added for better user-interaction and plotting of multiple classes
   - Issue 46: Current best iterate is stored to ensure an interruption of a calibration won't yield in a lost optimized value. Keyboard interrupt is now possible.

- **v0.1.4**
   - Add Goals from ebcpy
   - Add new tutorial for a better start with the framework. (See Issue 49)
   - Make changes based on new version 0.1.5 in ebcpy

- **v0.1.5**
   - Add new scripts in bin folder to ease the setup of the calibration for new users
   - Add configuration files and save/load classes
   - Issue 54: Skip failed simulations using two new kwargs in Calibrator class
   - Issue 53: Save final plots despite abortion of calibration process via STRG+C
   - Issue 51: Refactor reference_start_time to fix_start_time
   - Issue 23: Model Wrapper for MoCaTe files.

- **v0.1.6**
   - Add Re-Calibration code from master thesis of Sebastian Borges
   - Add fixed_parameters to calibration
   - Re-add tunerParas from ebcpy
   - Make changes based on ebcpy v.0.1.7
   - Split SensivitiyAnalyzer class and use object oriented programming
  
- **v0.2.0**
   - Adjust based on ebcpy v 0.2.0
   - Add examples and fix tutorial
   - Improve validation output
   - Fix version of SALib as 1.4 is not working
  
- **v0.2.1**
   - Unfix version of SALib as 1.4.0.2 and 1.4.4 are not working

- **v0.2.2**
   - Issue 21: Fix setup.py by removing the tests packages 

- **v0.2.3**
   - Add workflows

- **v0.3.0**
   - Issue 20: Add parallelization for calibration and sensitivity analysis
   - Issue 32: Add example converter to CI/CD

- **v0.3.1**
   - Issue 41: Fix logging and add kwarg
  
-  **v1.0.0**
    - Issue 43: Improvement of sensitivity analysis
    - Enables verbose sensitivity analysis and the reuse of simulations
    - It is now possible to use verbose sensitivity analysis for an automatic selection of tuner parameters
    - Enables multiprocessing for the entire sensitivity process
    - Sensitivity analysis is now usable for large models and data
    - Ends support vor python 3.7
