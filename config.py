#各种环境配置
from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1) #ExtraTreesClassifier(n_estimators=50, n_jobs=-1)极端随机树
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
from sklearn.svm import SVC
# clf = SVC(kernel='rbf', probability=True)
from xgboost import XGBClassifier
# clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1)
import torch
from sklearn.neighbors import KNeighborsClassifier

'''
定义当前使用的分类器
'''
CLASSFIERS = {
        # 'ETC':ExtraTreesClassifier(n_estimators=50, n_jobs=-1),
        'RF':RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1), 
        # 'KNN':KNeighborsClassifier(n_neighbors=5,   # 最近邻数量
        # weights='uniform',  # uniform/distance
        # algorithm='auto'   ),
        # 'XGBoost': XGBClassifier(objective='multi:softmax', num_class=10, n_jobs=-1),
        #  'SVC':SVC(kernel='rbf', probability=True)}
}

'''
定义数据集路径
'''
DATA_DIR = '/home/xiaowenyuan/Python/MGFS/data_csv'

'''
定义数据集名称
'''
DATASET_NAMES = [
# "kaggle.csv",
# "naticusdroid.csv",
# "madelon.csv",
# "UCI.csv",
"hepmass_train.csv",
]

'''
定义保存结果的文件路径  
'''
RESULT_FILE = '/home/xiaowenyuan/Python/MGFS/picture/result/experiment_time.csv'

'''
定义测试结果的函数
'''
RESULT_TEST_FILE = 'experiment_test.csv'

'''
定义所使用的cuda
'''
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")