from utils import *


def main():
    """ Returns the majority vote.

    :return: None
    """
    # Load the train data.
    train_data = load_train_csv("data")

    correct_question_map = {}
    total_question_map = {}

    # Count how many questions were correct.
    for i, q in enumerate(train_data["question_id"]):
        if q in correct_question_map:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] += 1
            total_question_map[q] += 1
        else:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] = 1
            total_question_map[q] = 1

    valid_data = load_valid_csv("data")
    predictions = []
    for i, q in enumerate(valid_data["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    acc = evaluate(valid_data, predictions)
    print("Validation Accuracy: {}".format(acc))

    test_data = load_public_test_csv("data")
    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    acc = evaluate(test_data, predictions)
    print("Test Accuracy: {}".format(acc))

    private_test = load_private_test_csv("data")
    predictions = []
    for i, q in enumerate(private_test["question_id"]):
        ratio = correct_question_map[q] / float(total_question_map[q])
        if ratio >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    private_test["is_correct"] = predictions
    save_private_test_csv(private_test)
    return


if __name__ == "__main__":
    main()
