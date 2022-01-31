
author's contact information:
faruch ismaeilov-319258737
idan ben nahum-312552995

delete notebooks line from picture project
the project is in python. to run the project you need to use in command line


Installation instruction:
pip install numpy
pip install opencv-python 
pip install glob 
pip install pandas
pip install petl
pip install sklearn.metrics
pip install tabulate
pip install skimage



development environment:.
the python code work in any environment that support python from 2.5 version


How to Run Your Program:

to run the code go to the command line and make sure that you are on the project file path:
 python classifier.py path_train path_val path_test

Example:
python classifier.py gender_split/train gender_split/valid gender_split/test
python classifier.py Desktop/images/train Desktop/images/valid Desktop/images/test

About the app:
App take all images from the path (train, valid, test) and start to build dataset(use DF) for al image  like {label, feature}
After his train to SVC model  by best parameters for get the must high accuracy.
Finally app create results.txt file to show  best parameters, accuracy for test and confusion matrix

