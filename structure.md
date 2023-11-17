# Plan for structure:

This is currently envisaged as the code for both running and analysing the NECOM experiment. It's a bit mixed.

* **experiment_run**
    * **control**
        * actually control the experiment, save data etc
    * **acquisition**
        * Acquiring chunked trace data
    * **processing** 
        * processing data as it comes in (realtime)
    * **visualise**
        * real time plotting
    * **GNOME**
        * sending processed data to GNOME
    * **scripts**
        * scripts for doing experiment tasks
    * **lib**
        * extra code to be installed (if it doesn't exist elsehere)
        * **shared_parameters**
* **MCU**
    * **MAIN**
        * arduino code for the main MCU that controls the experiment
        * possibly also wiring?
    * **STEPPER_MCU**
        * code for custom ESPXXX based stepper drivers (if required)
* **analysis**
    * **notebooks**
        * notebooks etc for analysis of existing data, or at least a link to where these can be found.
    * **necom_analysis_lib**
        * installable library of analysis routines.

    
