import pandas as pd
import data_preprocess
import prediction
from glob import glob
label_data = pd.read_csv('./label.csv', encoding='cp949')
file_list = glob('./data/*.mp4')
classes = ['Oropharynx', 'Tonguebase', 'Epiglottis']
data_preprocess.Preprocessing(file_list, label_data)
path_list, total_y, total_prob, cm, cm_box, s_result_box = prediction.predict(label_data,
                                                                              classes,
                                                                              file_list
)
df = pd.DataFrame(columns=['ID', 'true_label', 'predict'])
for i in range(len(path_list)):
    df.loc[i] = [path_list[i][0], classes[total_y[i].item()],
                 classes[total_prob[i].item()]]
df.to_csv('./data/predict.csv')
df_cm = pd.DataFrame(cm, 
                     columns=classes, 
                     index=classes
)
df_cm.to_csv('./data/cm.csv')

for i in range(len(classes)):
    df_cm_i = pd.DataFrame(cm_box[i], 
                           columns=[classes[i], "others"], 
                           index=[classes[i], "others"]
    )
    df_cm_i.to_csv(f'./data/cm_{classes[i]}.csv')
    idx_box = [i for i in range(len(file_list))]
    df_s_result_i = pd.DataFrame(s_result_box[i],
                               index=idx_box,
    )
    df_s_result_i.to_csv(f'./data/stacked_result_{classes[i]}.csv')