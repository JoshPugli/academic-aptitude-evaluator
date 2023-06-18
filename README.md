# Academic Aptitude Evaluator

## ML classifier for a personalized education platform.

**Problem:** Online education services, such as Khan Academy and Eedi, provide a broader audience with access to high-quality education. 
On these platforms, students can learn new materials by watching a lecture, reading course material, and talking to instructors in a forum. 
However, one disadvantage of the online platform is that it is challenging to measure students’ understanding of the course material. 
To deal with this issue, many online education platforms include an assessment component to ensure that students understand the core topics. 
The assessment component is often composed of diagnostic questions, each a multiple choice question with one correct answer. 
The diagnostic question is designed so that each of the incorrect answers highlights a common misconception.

This project uses a dataset provided by Eedi , an online education platform that is currently being used in many schools. Using this data, the we can predict whether 
a student can correctly answer a specific diagnostic question based on the student’s previous answers to other questions and other students’ responses. 
Predicting the correctness of students’ answers to as yet unseen diagnostic questions helps estimate the student’s ability level in a personalized education platform. 
Moreover, these predictions form the groundwork for many advanced customized tasks. 
For instance, using the predicted correctness, the we can automatically recommend a set of diagnostic questions of appropriate difficulty that fit the student’s background and learning status. 
Source code can be accessed [here](https://github.com/JoshPugli/academic-aptitude-evaluator).

I used kNN, Item Response Theory, and Neural Networks to and Random Forest classifiers to predict the correctness of students’ answers to diagnostic questions. 
I also used the classifiers to predict the difficulty of diagnostic questions, and evauated the performance of the classifiers using the accuracy evaluation metric.

k Nearest Neighbors (kNN)

For kNN, I used the kNN classifier from the scikit-learn library. I implemented user-based collaborative filtering: given a user, 
kNN finds the closest user that similarly answered other questions and predicts the correctness based on the closest student’s correctness, 
and item-based collaborative filtering: given a question, kNN finds the closest question that was similarly answered by other students and predicts 
the correctness based on the closest question’s correctness.

<div style="display:flex;">
  <img src="https://github.com/JoshPugli/academic-aptitude-evaluator/assets/86436788/a65a82e9-366d-422e-9adb-8d7b89114901" alt="added_feature_item" style="width:40%;">
  <img src="https://github.com/JoshPugli/academic-aptitude-evaluator/assets/86436788/71407193-7809-44da-9401-6070204580dc" alt="added_feature_user" style="width:40%;">
</div>



Item Response Theory (IRT)

Neural Networks (NN)

Random Forest (RF)

Joshua Puglielli 2023
