from Assignment2.utils import load_and_prepare_data
from Assignment2.models import LDAModel
from sklearn.metrics import accuracy_score


def rgb_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data()
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    print("****************************************")
    print("*        LDA Benchmark Solution        *")
    print("****************************************")

    # Create and fit the LDA model
    lda = LDAModel()
    print("Fitting LDA model...", end="")
    lda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = lda.predict(test_data)

    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test accuracy: {test_acc}")


def grayscale_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data(True)
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    print("****************************************")
    print("*        LDA Benchmark Solution        *")
    print("****************************************")

    # Create and fit the LDA model
    lda = LDAModel()
    print("Fitting LDA model...", end="")
    lda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = lda.predict(test_data)

    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test accuracy: {test_acc}")





def main():
    print("****************************************")
    print("*        RGB Benchmark Solution        *")
    print("****************************************")
    rgb_benchmark()

    print("\n****************************************")
    print("*     Grayscale Benchmark Solution     *")
    print("****************************************")
    grayscale_benchmark()


if __name__ == "__main__":
    main()

