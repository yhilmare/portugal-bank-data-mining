<h1>Data Mining Of Bank Info</h1>
This repository is about Bank Info data mining.<br/>
<h2>Data Set</h2>
The original data set is from <a href="http://archive.ics.uci.edu/ml/datasets/Bank+Marketing" target="_blank">UCI Machine Learning Repository</a>.The writer uses four methods to solve this problem, RandomForest, DecisitionTree, Support Vectory Machine and Logistic Regression respectively.<br/>
<h2>Code</h2>
This repository consists of three folders, Unit, lib and Util respectively. The lib package consists of the original code of the RandomForest, SVM, LogisticRegression and DecisionTree. The Unit consists of the test code of this project. The Util package consists of some tool code of this project.<br/>
The project contains the trained model, so that you can run this project immediately. Firstly, you should clone this project to your pc. Secondly, you should download the data set and put them to an specific position. The data set includes two main files, <code>bank-additional-full.csv</code> and <code>bank-additional.csv</code>, and these two files need be put together. Thirdly, you should modify the original code to let the program find the data set. You should open the file <code>./Util/DataUtil.py</code>, and modify the variable <code>filepath</code> to the folder that the data set locate in. Finally, you can open your cmd and type <code>python ./Unit/FinalUnit.py</code> to run the program. The program will show you the result.<br/>
If you do not want to use the model that the author gives, you can also train the model by yourself. The python file <code>./Unit/GeneralUnit.py</code> consists four functions, <code>serializeDTModel</code>, <code>serializeRFModel</code>, <code>serializeLRModel</code> and <code>serializeSVMModel</code> respectively. You can run anyone of them to train the model.<br/>
<h2>Result</h2>
The writer did some experiments on this data set. The result shows that the LogisticRegression model has the best performance and the DecisionTree performs worst.<br/>
<h2>More</h2>
If you want to learn more, you can visit my personal blog <a href="http://www.ilmareblog.com" target="_blank">ILMARE</a> and click <a href="http://www.ilmareblog.com/blog/GenArticleController?article_id=f60a3ead-df93-4fbf-a726-efe731ac9539&visitor_id=notlogin">here</a> to know more.
