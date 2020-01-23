Prerequisites
-----------------------

ANABLEPS runs on Unix. It requires the following softwares:

* Fuzzing engine (e.g., Driller https://github.com/shellphish/driller.git)
* Intel PT
* Perf: 
  Can find the source code of corresponding kernel, and compile by your own.
  Or Can use the following command to install it: sudo apt install linux-tools-[kernel_version]-common)



Run
-----------------------
1. Fuzzing

We use the Driller as the fuzzing engine to generate inputs for programs.
Users can also choose different input generation tools (e.g., fuzzing (AFL), symbolic execution, manually inputs collection) to generate different inputs.
After inputs generated, please put all generated inputs of the tested program in a specific folder.

2. Trace Collection

Command: python pt-data-collect.py -b ls -a "-l %s" -o [path to store pt data] -i [path of generated inputs] -f [true/false]

usage: pt-data-collect.py [-h] -b BIN_PATH -p PERF_PATH[-a BIN_ARGS] [-o OUTPUT_PATH]
                          [-i INPUT_FILE_PATH] [-f IS_FILE]
optional arguments:
  -h, --help          show this help message and exit
  -b BIN_PATH         The test program
  -a BIN_ARGS         The arguments for the binary 
  -o OUTPUT_PATH      The path of output files, "./pt_data" is used if not specified.
  -p PERF_PATH        The path of the perf program
  -i INPUT_FILE_PATH  The input file path (the generated inputs from the first step)
  -f IS_FILE          true:		the input is the file, 
                      false: 	the input is the content in the file
In the -a option, please replace the place of generated input as "%s"

Example: python pt-data-collect.py -p /path_to_perf -b ls -a "-l %s" -f false -i ./ls_inputs/


3. Side-channel Detection
usage: side-channel-detection.py [-h] -b BIN_PATH -p PERF_PATH [-i INPUT_PATH]
                                 [-o OUTPUT_PATH] [-l LOAD_PATH]

Detect side-channel vulnerabilities
optional arguments:
  -h, --help      show this help message and exit
  -b BIN_PATH     The binary to be run
  -i INPUT_PATH   The pt files folder path
  -o OUTPUT_PATH  The path to store output files
  -p PERF_PATH    The path of the perf program
  -l LOAD_PATH    Load the genreated CFGs for detection, and skip the graph
                  generation phase

Example: python side-channel-detection.py -b /bin/ls -p ~/Tools/linux/tools/perf/perf -i ./pt_data -o './results'


Results
-----------------------
In the "results" folder, the detection statistic is stored in the "[binary].performance" file. 
The detailed information about these results are stored in separate files as following:
[binary].pageorder.result:	The page level order-based results
[binary].cacheorder.result:	The cache level order-based results
[binary].page.result.big:	The page level nodes, which is vulnerable to time-based attack, not vulnerable to order-based attack
[binary].cache.result.big:	The cache level nodes, which is vulnerable to time-based attack, not vulnerable to order-based attack