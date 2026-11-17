# Synthetic sample-panel prototype

Python 3.11, CPU. Install requirements in a virtual environment and run `python demo.py`.

`set_model.py` implements the original independent Bernoulli model with complete-set observations. `panel.py` ranks independent scores. The files are the working starting point; the requested correlated set model and constrained joint selection are not implemented. You may refactor these entry points or add appropriate modules. No external datasets, model downloads or services are required.
