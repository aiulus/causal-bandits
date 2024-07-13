# causal-bandits
Performance analysis of Causal Bandit algorithms 

## Example usages
### 1. Generate SCM with parallel topology, 2 non-reward nodes and standard Gaussian noises:
    > Navigate to src/utils
    > python SCM.py --graph_type parallel --n 2 --noise_types N(0,1) 
    >>> saves the generated SCM to outputs/SCMs/.. as well as the causal graph generated as part of the SCM to outputs/graphs/.. as JSON files
### 2. Genereate a visualization for the same SCM:
    > python SCM.py --graph_type parallel --n 2 --noise_types N(0,1) --plot
    >>> saves the generated plot to outputs/plots as a .png file
### 3. Get the observational distribution for the SCM:
    > python sampling.py --file_path <../../outputs/file_name.json>
    > python sampling.py --file_path <../../outputs/file_name.json> --plot
### 4. Get interventional distribution for SCM | do(X_i=c):
    >>> Example: Intervening on three variables simulataneously
    > python sampling.py --mode l2 --do '(X1,0)' '(X2,3)' '(X3,7)' --file_name 'SCM_n3_parallel_graph_linear_functions_N(2,5).json'
    >>> --mode {'l1', 'l2', 'l3'} required
    >>> --file_name <filename.json> is required and points to the JSON file that contains the SCM's specification
    >>> --do '(X1,0)' to speciy one or more interventions
    >>> OUTPUTS: 
    >>>>> '../outputs/data/SCM_n3_parallel_graph_linear_functions_N(2,5).csv'
    >>>>> '../outputs/plots/SCM_n3_parallel_graph_linear_functions_N(2,5).png'
    >>> TO PLOT (AND SAVE) SAMPLING RESULTS, USE:
    >  python sampling.py --mode l2 --file_name 'SCM_n3_parallel_graph_linear_functions_N(2,5)_doX1-5X2-3X3-7.json' --plot --save 

