# Data Mining Of Bank Info
## Data Set

The data set is from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).The original data set contains two editions, one of them is older one that the data contains less dimensions, the other edition is the latest. The author uses the latest data set to train the model. The writer uses four methods to solve this problem, RandomForest, DecisitionTree, Support Vectory Machine and Logistic Regression respectively.

## Code

This repository consists of three folders, Unit, lib and Util respectively. The lib package consists of the original code of the RandomForest, SVM, LogisticRegression and DecisionTree. The Unit consists of the test code of this project. The Util package consists of some tool code of this project.

The project contains the trained model, so that you can run this project immediately. Firstly, you should clone this project to your pc. Secondly, you should download the data set and put them to an specific position. The data set includes two main files, `bank-additional-full.csv` and `bank-additional.csv`, and these two files need be put together. Thirdly, you should modify the original code to let the program find the data set. You should open the file `./Util/DataUtil.py`, and modify the variable `filepath` to the folder that the data set locate in. Finally, you can open your cmd and type `python ./Unit/FinalUnit.py` to run the program. The program will show you the result.

<p>If you do not want to use the model that the author gives, you can also train the model by yourself. The python file <code>./Unit/GeneralUnit.py</code> consists four functions, <code>serializeDTModel</code>, <code>serializeRFModel</code>, <code>serializeLRModel</code> and <code>serializeSVMModel</code> respectively. You can run anyone of them to train the model.</p>
<h2>Result</h2>
<p>The writer did some experiments on this data set. The result shows that the Logistic regression model has the best performance and the Decision tree performs worst on the train set, but the SVM model performs best when the author tests the model by using the full data set. The results just like this:</p><img src="https://github.com/yhswjtuILMARE/DataMiningOfBank/blob/master/Images/result.png"/>
<p>The Decision tree model performs well as well, and this model can classify the 84.6% data of the data set correctly. The author did hundreds of experiments, and find out that the accuracy of the decision tree model is very stable. The average accuracy of this model converges to 0.62. The author can illustrate this tendency like this:</p><img src="https://github.com/yhswjtuILMARE/DataMiningOfBank/blob/master/Images/dt.png"/>
<p>The Random forest model can be understood as an intergration of many decision tree models. The author did some experiments to find out how the number of basic learner of the random forest affects the accuracy of the model. The result is just like this:</p><img src="https://github.com/yhswjtuILMARE/DataMiningOfBank/blob/master/Images/rf.png"/><p>We can conclude that the accuracy of the model increases as the number of model-based learners increases. </p>
<h2>More</h2>
<p>If you want to know more, you can visit my personal blog <a href="http://www.ilmareblog.com" target="_blank">ILMARE</a> and click <a href="http://www.ilmareblog.com/blog/GenArticleController?article_id=f60a3ead-df93-4fbf-a726-efe731ac9539&visitor_id=notlogin">here</a> to know more.</p>
