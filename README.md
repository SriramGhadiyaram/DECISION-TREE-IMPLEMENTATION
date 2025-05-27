# DECISION-TREE-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : GHADIYARAM JAYA SAI SREE RAMA KUMAR

*INTERN ID* : CT06DM1188

*DOMAIN* : MACHINE LEARNING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

The process of building a Decision Tree classifier in Google Colab involves several well-defined steps that begin with setting up the coding environment and end with the visualization of the decision tree structure. Decision Trees are among the most widely used algorithms in supervised machine learning, especially for classification tasks. They work by splitting the dataset into subsets based on the value of input features, forming a tree structure where each internal node represents a decision on a feature, and each leaf node represents an output label. The steps detailed below were carried out using the Python programming language in Google Colab, and I referred to reliable online sources such as GeeksforGeeks, Scikit-learn documentation, and various tutorial blogs available via Google Search for conceptual clarity and practical implementation guidance.

To begin with, the necessary Python libraries are imported. These include pandas and numpy for data handling and numerical operations, matplotlib.pyplot and seaborn for data visualization, and several components from the sklearn library, such as DecisionTreeClassifier, plot_tree, and tools for model evaluation and dataset handling. Google Colab conveniently supports all these libraries without the need for manual installation, making it an ideal platform for machine learning experimentation.

Next, the Iris dataset, which is a classic dataset used in pattern recognition, is loaded using the load_iris() function from sklearn.datasets. The dataset consists of 150 instances of iris flowers classified into three species: setosa, versicolor, and virginica. Each instance has four numeric features: sepal length, sepal width, petal length, and petal width. The features are stored in a Pandas DataFrame, and the corresponding target labels are stored in a separate Pandas Series. At this stage, a preview of the dataset is displayed, and the class names (target labels) are printed to give a basic understanding of the data structure.

The dataset is then split into training and testing sets using the train_test_split() function. This is an essential step in any machine learning pipeline as it ensures that the model is trained on one portion of the data and tested on another, thereby providing a fair estimate of how the model will perform on unseen data. In this case, 80% of the data is used for training, and 20% is reserved for testing.

Following the data split, a DecisionTreeClassifier model is instantiated. The criterion parameter is set to 'entropy', meaning that the decision tree will use information gain as the splitting criterion. The model is then trained using the .fit() method on the training data. Once the model is trained, it is used to predict the classes for the test set. The predictions are then evaluated using several metrics, including accuracy score, classification report, and confusion matrix. These metrics help in understanding the performance of the model in terms of how accurately it is predicting each class. The confusion matrix is visualized using a heatmap via seaborn, which clearly shows how well the model differentiates between the different classes.

Finally, the decision tree itself is visualized using plot_tree() from the sklearn.tree module. This visualization provides a clear graphical representation of the decision-making process followed by the model. Each node in the tree shows the condition based on which the data is split, and the leaf nodes represent the final class predictions. The tree is colored to indicate different classes, making it easier to interpret the model. The visualization helps in understanding which features are most important in the classification and how the model arrives at a particular decision.

In summary, building a decision tree classifier in Google Colab is a structured process involving data preparation, model training, evaluation, and visualization. Referring to trusted sources such as GeeksforGeeks, Scikit-learn tutorials, and other technical blogs found via Google greatly helps in gaining both theoretical understanding and hands-on skills. This approach is not only educational but also sets a foundation for implementing more advanced machine learning algorithms in the future.

![Image](https://github.com/user-attachments/assets/3f193eae-9bb2-4d5f-ba8d-254983e6c3b1)

![Image](https://github.com/user-attachments/assets/ffc7920c-c771-45b3-97da-0065cc29f265)

![Image](https://github.com/user-attachments/assets/bb5d7278-a33a-4a29-89d9-d69db41915dc)

![Image](https://github.com/user-attachments/assets/00ce2c19-60da-4f8e-a84f-c4b6969db006)

![Image](https://github.com/user-attachments/assets/5ce92736-e375-44b6-a897-4620d1a037cd)

![Image](https://github.com/user-attachments/assets/69a4b0c3-dd58-4679-a03e-82830e76a916)
