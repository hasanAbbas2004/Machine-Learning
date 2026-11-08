# Loan approval prediction: hybrid MPI + OpenMP ML pipeline

CXX      ?= g++
MPICXX   ?= mpic++
CXXFLAGS ?= -O2 -fopenmp

MODELS = random_forest.cpp mlp.cpp logistic_regression.cpp

all: loan_preprocessor hybrid_ml_trainer ml_predictor

loan_preprocessor: main.cpp loan_data_preprocessor.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

hybrid_ml_trainer: main_model.cpp $(MODELS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

ml_predictor: prediction.cpp $(MODELS)
	$(CXX) $(CXXFLAGS) $^ -o $@

preprocess: loan_preprocessor
	./loan_preprocessor loan_data.csv processed_data.csv

train: hybrid_ml_trainer
	mpirun -np 3 ./hybrid_ml_trainer processed_data.csv

predict: ml_predictor
	./ml_predictor

clean:
	rm -f loan_preprocessor hybrid_ml_trainer ml_predictor *.o *.exe

.PHONY: all preprocess train predict clean
