from catboost import CatBoostClassifier
from utils.improve_tool.mutation_funs import approved_mutated_sequences

model = CatBoostClassifier()
model.load_model("models/catboost_final_model.cbm")



