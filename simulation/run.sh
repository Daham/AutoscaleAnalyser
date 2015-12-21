R --vanilla -f ../prediction-model/ensemble.R
python AWS.py
python PlotVM.py
python PlotCost.py
