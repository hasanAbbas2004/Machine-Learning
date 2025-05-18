# Machine Learning Pipeline for Loan Approval Prediction  

## **Repository Details**  
- A **machine learning pipeline** for predicting loan approvals using:  
  - **Logistic Regression** (interpretable probabilities)  
  - **Random Forest** (robust ensemble model)  
  - **Multilayer Perceptron (MLP)** (neural network for complex patterns)  
- **Data preprocessing** to handle:  
  - Missing values (mean imputation, categorical defaults)  
  - Invalid data (negative/zero values corrected)  
  - Categorical encoding (employed=1/unemployed=0, approved=1/rejected=0)  
- **Parallel training & evaluation** (MPI + OpenMP for speed, but ML is the focus here)  
- **Performance metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix  

## **Problem Statement**  
- **Problem**: Loan approvals require fast, consistent, and data-driven decisions.  
- **Challenge**: Manual review is slow, subjective, and lacks scalability.  
- **Solution**:  
  - Automate risk assessment using ML models.  
  - Compare multiple algorithms to find the best performer.  
  - Handle real-world data issues (missing values, outliers).  
  - Provide interpretable predictions for decision-making.  

**Use Case**: Banks, fintech, or lenders can deploy this to:  
✔ Speed up approval processes  
✔ Reduce human bias  
✔ Maintain accuracy at scale  

*(For technical details on parallel implementation, see the full report.)*
