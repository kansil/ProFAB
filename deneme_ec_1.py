
from profab.model_preprocess import extract_protein_feature


extract_protein_feature('edp', 0, 
                       'use_case/ecNo_1-2-1-88', 
                       'negative_data')



from profab.import_dataset.data_loader import ECNO
data_model = ECNO(ratio = [0.1, 0.2], protein_feature = 'paac', pre_determined = False, set_type = 'random')
X_train,X_test,X_validation,y_train,y_test,y_validation = data_model.get_data(data_name = 'ecNo_1-2-4')

from profab.model_preprocess.scaler import scale_methods
X_train,scaler = scale_methods(X_train,scale_type = 'standard')
X_test = scaler.transform(X_test)

from profab.model_learn.classifications import classification_methods
model_path = 'model_path.txt'
model = classification_methods(ml_type = 'logistic_reg', X_train = X_train, y_train = y_train, path = model_path)

from profab.model_evaluate.evaluation_metrics import evaluate_score
score_train,f_train = evaluate_score(model,X_train,y_train,preds = True)
score_test,f_test = evaluate_score(model,X_test,y_test,preds = True)
score_validation,f_validation = evaluate_score(model,X_validation,y_validation,preds = True)

from profab.model_evaluate.form_table import form_table
score_path = 'score_path.csv'
scores = {'train':score_train,'test':score_test,'validation':score_validation}
form_table(scores = scores, path = score_path)

