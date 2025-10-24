import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class EOSClassifier:
    def train(self, trainX, trainY):

        # HINT!!!!!
        # (The following word lists might be very helpful.)
        self.abbrevs = load_wordlist('classes/abbrevs')
        self.sentence_internal = load_wordlist("classes/sentence_internal")
        self.timeterms = load_wordlist("classes/timeterms")
        self.titles = load_wordlist("classes/titles")
        self.unlikely_proper_nouns = load_wordlist("classes/unlikely_proper_nouns")


        # In this part of the code, we're loading a Scikit-Learn model.
        # We're using a DecisionTreeClassifier... it's simple and lets you
        # focus on building good features.
        # Don't start experimenting with other models until you are confident
        # you have reached the scoring upper bound.
        #self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42) #higher than base
        #self.clf = SVC() #lower than base
        #self.clf = LogisticRegression() #lower than base
        #self.clf = GradientBoostingClassifier() #higher than random forest
        #self.clf = DecisionTreeClassifier(max_depth=11, min_samples_split=8)
        #self.clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=5) #v slow




        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)




    def extract_features(self, array):

        # Our model requires some kind of numerical input.
        # It can't handle the sentence as-is, so we need to quantify our them
        # somehow.
        # We've made an array below to help you consider meaningful
        # components of a sentence, for this task.
        # Make sure to use them!
        id, word_m3, word_m2, word_m1, period, word_p1, word_p2, word_3, left_reliable, right_reliable, num_spaces = array

        # The "features" array holds a list of
        # values that should act as predictors.
        # We want to take some component(s) above and "translate" them to a numerical value.
        # For example, our 4th feature has a value of 1 if word_m1 is an abbreviation,
        # and 0 if not.

        features = [  # TODO: add features here
            left_reliable,
            right_reliable,
            num_spaces,
            1 if word_m1 in self.abbrevs else 0,

            # ==========TODO==========
            # Make a note of the score you'll get with
            # only the features above (it should be around
            # 0.9). Use this as your baseline.
            # Now, experiment with adding your features.
            # What is a sign that period marks the end of a
            # sentence?
            # Hint: Simpler features will get you further than complex ones, at first.
            # We've given you some features you might want to experiment with below.
            # You should be able to quickly get a score above 0.95!

            # sentence structure features
            1 if word_m1.istitle() and word_p1[0].isupper() else 0,
            len(word_m1),
            len(word_p1),
            1 if word_p1[0].isupper() and word_p1 not in self.unlikely_proper_nouns else 0,
            1 if word_m1.istitle() and word_p1[0].isupper() else 0,

            # punctuation features
            1 if word_p2 in [",", ";", "-"] else 0,
            1 if word_p1 in ["<P>"] else 0,
            1 if word_p1 == "," else 0,
            1 if word_p1 in [",", ";", "-"] else 0,

            # word list features
            1 if word_m1 in self.sentence_internal else 0,
            1 if word_m1 in self.titles else 0,
            1 if word_m1.lower() in ["mr", "mrs", "dr", "sr", "jr"] else 0,
            1 if word_m1 in self.abbrevs and word_p1[0].islower() else 0,  
            1 if word_m1 in self.abbrevs and word_p1.isdigit() else 0,  


            # capitalization checks
            1 if word_p1[0].isupper() else 0,
            1 if word_m1[0].isupper() else 0,
            1 if word_m1.lower in self.abbrevs else 0,

            # temporal and sentence-internal features
            1 if word_m1 in self.timeterms else 0,
            1 if word_p1 in self.sentence_internal else 0,
            1 if word_m1.lower() in ["a.m", "p.m"] and word_p1.isdigit() else 0,
            1 if len(word_m1) == 1 and word_p1[0].isupper() else 0,


            # period and contextual punctuation
            1 if word_p1 in ['.', '..', '...'] else 0,
            1 if word_p1 in ['"', ')'] else 0,
            1 if word_p1.lower() in ['and', 'but', 'or', 'however'] else 0,
            1 if word_m1[-1] == "." and word_m1 not in self.abbrevs else 0,
            len(word_m1) - len(word_m1.rstrip(".")),

            # number features
            1 if word_m1.isdigit() and word_p1.lower() in ["a.m", "p.m"] else 0,
            1 if word_m1 in ["!", "?", "”", "’"] else 0,
            1 if word_p1.lower() in ["the", "it", "this", "that", "he", "she", "they", "we", "but", "so", "and", "in"] else 0,
            1 if word_m1.isdigit() else 0,

            # specific words markers
            1 if word_m1.lower() in ["said", "asked", "concluded", "replied", "noted", "wrote"] else 0,
            1 if word_m1[0] in ['"', "’"] else 0,

            # special markers
            1 if word_m1.lower() in ["etc", "cf", "e.g", "i.e", "vs"] else 0,
            1 if word_m1 in [")", "]"] or word_p1 in [")", "]"] else 0,

            # common verbs
            1 if word_p1.lower() in ["is", "was", "has", "have", "do", "does"] else 0,
            1 if word_p1.lower() in [",", ":", ")", "which", "although", "because", "since"] else 0,

            # numbers
            1 if word_m1.isdigit() and word_p1.isdigit() else 0,
            sum([1 for w in [word_m3, word_m2, word_m1, period, word_p1, word_p2, word_3] if w == "."]),

            # abrreviations
            1 if word_m1 in self.abbrevs and word_p1[0].islower() else 0,
            1 if word_m1.isdigit() and word_p1.isdigit() else 0,
            1 if len(word_m1) <= 2 and word_p1[0].isupper() else 0,
            1 if word_m1 in ['"', "'"] and word_p1[0].isupper() else 0,

            # sentence-starters
            1 if word_p1.lower() in ["the", "this", "he", "she", "it", "i", "but", "and"] else 0,

            # punctuation and digits
            1 if word_m1 == "." and period == "." and word_p1 == "." else 0,
            1 if word_m1.isdigit() and word_p1.isdigit() and word_p2 == "." else 0,
            1 if word_m1 == "." and period == "." and word_p1 == "." else 0,

            # abbreviations and number formatting
            1 if word_m1.lower() in ["fig", "eq", "p", "sec", "vol", "chap", "gov"] else 0,
            1 if word_m1.isdigit() and word_p1 in ["mm", "cm", "kg", "%"] else 0,
            1 if word_m1.isdigit() and word_p1 == "." and word_p2[0].isupper() else 0,
            1 if word_m1.lower() == "sec" and word_p1.isdigit() else 0, 

            # titles, abbrevs, other edges
            1 if word_m1.lower() in ["mr", "mrs", "dr", "gov", "gen", "sr", "jr", "co", "capt", "rep"] and word_p1[0].isupper() else 0,
            1 if len(word_m1) == 2 and word_m1[1] == "." and word_p1[0].isupper() else 0,
            1 if word_m1.lower() in ["figs", "eq", "sec", "vol"] and word_p1.isdigit() else 0,
            1 if word_m1 in self.titles and word_p1[0].isupper() else 0,
            1 if word_m1 in self.abbrevs and word_p1[0].islower() else 0, 


            # transition words
            1 if word_p1.lower() in ["however", "thus", "therefore", "nevertheless", "consequently", "furthermore", "moreover", "never"] else 0,

            # Parenthesis
            1 if word_m1 == "." and word_p1 in [")", "]"] else 0,
            1 if word_p1 in ["(", "Sec", "Ch", "Pt"] else 0,
            1 if word_m1 == "." and word_p2 in [")", "]"] else 0, 
            1 if word_p2 == ")" else 0, 


            # ellipses
            1 if word_m1 == "." and word_p1 == "." and word_p2 == "." else 0,

            # Quotes
            1 if word_p1 in ['"', "'", "”", ')'] else 0,
            1 if word_p1 in ['``', "''"] and word_p2[0].isupper() else 0,

            # special abbreviations
            1 if word_m1 in self.abbrevs and word_p1[0].isupper() else 0,
            1 if word_m1 in self.abbrevs and word_p1.lower() in ["and", "or", "but", "however"] else 0,
            1 if word_m1 == "D.J" else 0,

            # Numbers/scientific
            1 if word_m1.isdigit() and word_p1.isdigit() else 0,
            1 if word_m1 in ["mm", "cm", "kg", "%", "p.m", "a.m", "24-hr"] else 0,
            1 if word_m1 == "p.m" else 0,
            1 if word_m1 == "a.m" else 0,
            1 if word_m1 == "p.m" and  word_m1.lower in self.timeterms else 0,
            1 if word_m1 == "a.m" and word_m1.lower in self.timeterms else 0,

            1 if word_p1[0].isupper() and word_p1 not in self.unlikely_proper_nouns else 0,
            1 if word_m1 in self.abbrevs and word_p1[0].islower() else 0,
            1 if word_p1.lower() in ["and", "but", "or", "which", "although", "because", "since"] else 0, 
            1 if word_m1 == "Aj" else 0,
            1 if word_m1 == "Af" else 0,
            1 if word_m1 == "2" else 0,
            1 if word_m1 == "U.S" else 0,
            1 if len(word_m1) == 1 and word_m2 == "." else 0,
            1 if word_m1.lower() in ["u.s", "u.n", "st", "mass", "calif", "gen"] else 0


         
        ]

        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_wordlist(file):
    with open(file) as fin:
        return set([x.strip() for x in fin.readlines()])


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split()
            X.append(arr[1:])
            y.append(arr[0])
        return X, y


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    classifier = EOSClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()