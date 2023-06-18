from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import ast
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances

# -------------------------- Part B Attempts -------------------------- #


def load_data():
    question_meta_df = pd.read_csv("../311-project/data/question_meta.csv")
    subject_meta_df = pd.read_csv("../311-project/data/subject_meta.csv")
    student_meta_df = pd.read_csv("../311-project/data/student_meta.csv")
    question_data = question_meta_df.to_dict()
    question_data['subject_id'] = {k: ast.literal_eval(
        v) for k, v in question_data['subject_id'].items()}
    subject_data = subject_meta_df.to_dict()
    student_data = student_meta_df.to_dict()
    return question_data, subject_data, student_data


def create_user_prevent_overfitting_matrix(sparse_matrix, student_data, question_data, subject_data):
    questions = {}
    for key in question_data["question_id"]:
        questions[question_data["question_id"][key]
                  ] = question_data["subject_id"][key]

    for key in questions:
        lst = questions[key]
        replacement = []
        for subject_index in lst:
            replacement.append(subject_data["name"][subject_index])
        questions[key] = replacement

    topics = list(set([topic for topics in questions.values()
                  for topic in topics]))

    # Create a matrix of questions and their subject topics
    X = np.zeros((len(questions), len(topics)))
    for i, topics in enumerate(questions.values()):
        for topic in topics:
            X[i, topics.index(topic)] = 1

    # Cluster the questions using K-Means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    grouped_questions = {n: set() for n in range(n_clusters)}
    for i, cluster_num in enumerate(kmeans.labels_):
        grouped_questions[cluster_num].add(i)

    new_rows = []
    for student_row in sparse_matrix:
        new_vals = [[0, 0] for _ in range(n_clusters)]
        for i, q in enumerate(student_row):
            if not np.isnan(q):
                cluster_num = kmeans.labels_[i]
                new_vals[cluster_num][0] += q
                new_vals[cluster_num][1] += 1
        new_rows.append(new_vals)

    for row in new_rows:
        for i, val in enumerate(row):
            if val[1] == 0:
                row[i] = np.nan
                continue
            percentage = val[0] / val[1]
            if percentage > 0.5:
                row[i] = 1
            else:
                row[i] = 0

    matrix = sparse_matrix.copy()
    for row_num in range(sparse_matrix.shape[0]):
        for q_index in range(sparse_matrix.shape[1]):
            cluster_vals = new_rows[row_num]
            cluster_num = kmeans.labels_[q_index]
            val = cluster_vals[cluster_num]
            matrix[row_num][q_index] = val

    return matrix


def create_by_user_matrix(sparse_matrix, student_data):
    """
    Adds new feautes to the students in sparse matrix

    :param sparse_matrix: 2D sparse matrix
    :return: by user matrix
    """
    by_user_matrix = np.concatenate(
        (sparse_matrix.copy(), np.full((sparse_matrix.shape[0], 2), np.nan)), axis=1)

    for index in student_data['user_id']:
        id = student_data['user_id'][index]

        if student_data['gender'][index] != 0:
            by_user_matrix[id][-1] = student_data['gender'][index]

        if not np.isnan(student_data['premium_pupil'][index]):
            by_user_matrix[id][-2] = student_data['premium_pupil'][index]

        # if isinstance(student_data['data_of_birth'][index], str):
        #     by_user_matrix[id][-3] = int(student_data['data_of_birth'][index][:4])

    return by_user_matrix


def create_by_item_matrix(sparse_matrix, question_data, subject_data, clusters):
    """
    Adds new features to the questions in sparse matrix

    :param sparse_matrix: 2d sparse matrix
    :return: by item matrix
    """
    by_item_matrix = np.pad(sparse_matrix.copy(), ((
        0, len(clusters)), (0, 0)), mode='constant', constant_values=np.nan)

    # Adds subject tags as features for each question
    for index in question_data['question_id']:
        question_id = question_data['question_id'][index]
        subject_ids = question_data['subject_id'][index]

        for subject_id in subject_ids:
            subject_name = subject_data['name'][subject_id]
            for i in range(len(clusters)):
                if subject_name in clusters[i]:
                    by_item_matrix[-1 * (i + 1)][question_id] = 1

    return by_item_matrix


def manhattan_distance(X, Y, missing_values=np.nan):
    # Replace missing values with column mean before computing distance.
    X_mean = np.nanmean(X, axis=0)
    Y_mean = np.nanmean(Y, axis=0)
    X = np.nan_to_num(X, nan=X_mean)
    Y = np.nan_to_num(Y, nan=Y_mean)
    return manhattan_distances(X[np.newaxis, :], Y[np.newaxis, :])[0][0]

# ---------------------------- Part A ---------------------------- #


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    acc = None
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    # changed from ../data to ../311-project/data
    sparse_matrix = load_train_sparse("../311-project/data").toarray()
    val_data = load_valid_csv("../311-project/data")
    test_data = load_public_test_csv("../311-project/data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    # --------------------- extensions --------------------- #
    # question_data, subject_data, student_data = load_data()

    # topics = []
    # for name in subject_data['name'].values():
    #     topics.append(name)

    # # Exclude "Number" and "Math", as they are too frequent to be significant
    # topics = topics[2:]

    # # Convert the list of topics into a matrix of TF-IDF features
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(topics)

    # # Compute the cosine similarity matrix
    # cosine_sim = cosine_similarity(tfidf_matrix)

    # # Cluster the topics into similar groups using KMeans
    # num_clusters = 5
    # km = KMeans(n_clusters=num_clusters)
    # km.fit(cosine_sim)
    # clusters = []

    # # Print the clusters
    # for i in range(num_clusters):
    #     print("Cluster ", i+1)
    #     cluster = set()
    #     for j, topic in enumerate(topics):
    #         if km.labels_[j] == i:
    #             cluster.add(topic)
    #     clusters.append(cluster)
    #     print(cluster)
    #     print()

    # by_user_matrix = create_by_user_matrix(sparse_matrix, student_data)
    # by_user_matrix_2 = create_user_prevent_overfitting_matrix(sparse_matrix, student_data, question_data, subject_data)
    # by_item_matrix = create_by_item_matrix(sparse_matrix, question_data, subject_data, clusters)

    # --------------------- run KNN --------------------- #
    # ------------------- KNN by user ------------------- #
    x_val = [1, 6, 11, 16, 21, 26]
    y_val = []
    highest_acc = 0
    k_star = 0
    for k in x_val:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        if acc > highest_acc:
            highest_acc = acc
            k_star = k
        y_val.append(acc)

    k_star_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(f"Chosen k*: {k_star}, Final Test Accuracy: {k_star_acc}")

    # -------------------- Plotting -------------------- #
    # plt.xlabel('k')
    # plt.ylabel('Validation Accuracy')
    # plt.title('KNN Impute by User')
    # plt.plot(x_val, y_val, '-o')
    # plt.savefig('public/knn_user.png')
    # plt.clf()

    # ------------------- KNN by Item ------------------- #
    x_val = [1, 6, 11, 16, 21, 26]
    y_val = []
    highest_acc = 0
    k_star = 0
    for k in x_val:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        if acc > highest_acc:
            highest_acc = acc
            k_star = k
        y_val.append(acc)

    k_star_acc = knn_impute_by_item(sparse_matrix, test_data, k_star)
    print(f"Chosen k*: {k_star}, Final Test Accuracy: {k_star_acc}")
    # -------------------- Plotting -------------------- #
    # plt.xlabel('k')
    # plt.ylabel('Validation Accuracy')
    # plt.title('KNN Impute by Item')
    # plt.plot(x_val, y_val, '-o')
    # plt.savefig('public/knn_item.png')


if __name__ == "__main__":
    main()
