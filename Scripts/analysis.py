import matplotlib.pyplot as plt


def plot_word_freq(questions):
    # compute no. of words in each question
    word_cnt = [len(quest.split()) for quest in questions]
    # Plot the distribution
    plt.figure(figsize=[8,5])
    plt.hist(word_cnt, bins = 40)
    plt.xlabel('Word Count/Question')
    plt.ylabel('# of Occurrences')
    plt.title("Frequency of Word Counts/sentence")
    plt.show()