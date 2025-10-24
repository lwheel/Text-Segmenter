import argparse
import re
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


class SegmentClassifier:
    def train(self, trainX, trainY):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)  # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        words = text.split()
        num_chars = len(text)
        num_words = len(words)
        num_spaces = text.count(' ')
        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            text.count(' '),
            text.count(' ') / max(1, len(text)),  
            sum(1 if w.isupper() else 0 for w in words),
               
            # NNHEAD
            1 if text.startswith(("From:", "Subject:", "Date:", "Message-ID:", "Organization:", "Keywords:")) else 0,
            1 if "@" in text and "." in text.split("@")[-1] else 0,
            1 if any(w.lower() in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"} for w in words) else 0, 
            1 if "<" in text and ">" in text and "@" in text else 0,  
            1 if ":" in text and any(c.isdigit() for c in text) else 0,  
            1 if "Organization:" in text else 0,  
            1 if text.startswith("Subject: Re:") else 0, 
            1 if "(" in text and ")" in text else 0,  
            1 if any(c.isdigit() for c in text) else 0, 
            min(num_chars // 10, 10),  

            # QUOTED
            1 if any(text.endswith(ending) for ending in ["wrote:", "writes:", "write:", "said:"]) else 0,
            1 if text.lstrip().startswith((">", "@", ":")) else 0,  
            text.lstrip().count(">"), 
            1 if any(w in text for w in ["wrote:", "writes:", "said:", "noted:"]) else 0,
            1 if "@" in text and "." in text.split("@")[-1] else 0, 
            1 if "<" in text and ">" in text and "@" in text else 0, 
            1 if any(w in text for w in ["In article", "On ", "replied:"]) else 0,  
            1 if "(" in text and ")" in text and "@" in text else 0, 
            1 if text.endswith(":") else 0,  
            1 if words and words[0][0].isupper() else 0, 
            1 if text.lstrip().startswith(">>") else 0,  
            1 if text.count(">") >= 3 else 0,  
            1 if text.isupper() else 0, 
            1 if text.startswith("KC>") else 0, 
            1 if text.strip().endswith(":") else 0,  
            1 if text.lstrip().startswith(": ") else 0,
            1 if text.lstrip().startswith("[") and text.rstrip().endswith("]") else 0, 
            1 if any(w in text.lower() for w in ["previous message", "original post", "replied:"]) else 0,  
            
            # SIG
            1 if text.startswith("-- ") else 0, 
            1 if text.startswith(("-", "=", "*", "#", "_")) and len(set(text.strip())) <= 3 else 0,  
            1 if text.count("|") > 2 else 0,  
            1 if any(w in text.lower() for w in ["tel:", "phone:", "fax:", "e-mail:", "mailto:", "contact:"]) else 0,  
            1 if "@" in text and "." in text.split("@")[-1] else 0,  
            1 if any(w in text.lower() for w in ["http://", "https://", "www."]) else 0,  
            1 if text.strip().endswith("--") else 0,  
            1 if sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())) > 0.6 else 0, 
            1 if "(" in text and ")" in text else 0,  
            1 if sum(1 for c in text if c.isdigit()) > 4 else 0, 
            1 if text.lstrip().startswith(("-", "=")) and len(set(text.strip())) <= 3 else 0, 
            1 if len(words) < 5 and "@" in text else 0, 
            1 if len(words) < 8 and any(w.lower() in ["ceo", "founder", "engineer", "professor", "developer", "student", "researcher"] for w in words) else 0,  
            1 if len(words) < 8 and any(w.lower() in ["university", "institute", "lab", "dept", "corporation", "company", "division"] for w in words) else 0,  
            1 if sum(1 for c in text if c in ["*", "=", "-", "_", "~"]) / max(1, num_chars) > 0.3 else 0, 
            1 if any(w in text for w in ["Best,", "Regards,", "Sincerely,", "Thanks,", "Cheers,", "Yours,"]) else 0,  
            1 if any(w in text for w in ["P.S.", "Sent from", "Please consider", "Think before you print"]) else 0, 

            # BLANK
            1 if text.strip() == "" else 0, 

            # TABLE
            1 if text.lstrip().startswith(tuple("0123456789")) else 0, 
            1 if num_spaces / max(1, num_words) < 1.3 else 0,  
            1 if text.lstrip().startswith(tuple("0123456789")) else 0, 
            1 if re.match(r'^\s*\d+[\s\t]+', text) else 0,
            1 if re.match(r'^\s*[-=]+$', text) else 0, 
            1 if len(set(text.strip())) < 6 and any(c in text for c in '-=|') else 0, 
            1 if sum(1 for c in text if c.isdigit()) > (0.2 * len(text)) else 0,  

            text.count('\t') + text.count('  '), 
            1 if text.count('\t') > 2 else 0,  
            1 if sum(1 for c in text if c.isspace()) / max(1, len(text)) > 0.3 else 0, 

            1 if re.search(r'^\s*\d+\.\s+', text) else 0,  
            1 if len(re.findall(r'\d+', text)) > 3 else 0,  
            1 if re.search(r'\b[A-Z][a-z]+\s+\d+\b', text) else 0,  
            1 if re.search(r'^\s*[A-Za-z]+(\s+[A-Za-z]+)*\s+\d+', text) else 0,  

            1 if sum(1 for c in text if c == ' ') / max(1, len(text)) > 0.4 else 0, 
            1 if any(c in text for c in ['-', '=', '|']) and len(set(text.strip())) < 6 else 0, 
            1 if len(words) > 4 and num_spaces / max(1, num_words) < 1.5 else 0, 

            # GRAPHICS
            1 if sum(1 for c in text if not c.isalnum()) / max(1, num_chars) > 0.5 else 0,  
            1 if sum(1 for c in text if not c.isalnum()) / max(1, len(text)) > 0.6 else 0, 
            1 if any(c in text for c in ['_', '-', '*', '|', '=', '/', '\\', '+', 'o']) else 0,  
            1 if len(set(text.strip())) < 5 and any(c in text for c in "-=_|/\\+o*") else 0, 
            1 if text.count("|") > 5 else 0,  
            1 if text.count("*") > 5 else 0, 
            1 if text.count("=") > 5 else 0,  
            1 if text.count("_") > 5 else 0,  
            1 if text.count("\\") + text.count("/") > 3 else 0, 
            1 if sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())) > 0.7 else 0, 
            1 if text.count(" ") / max(1, len(text)) > 0.3 else 0,  
            1 if any(w in text for w in ["PPPPPPP", "IIIIIIII", "ZZZZZZZZ", "AAAA"]) else 0,  
            1 if any(text.startswith(c * 3) for c in "-=_|/\\+") else 0,  
            1 if len(text) > 50 and sum(1 for c in text if c in "-=_|/\\*o") / max(1, len(text)) > 0.5 else 0,  
            1 if text.strip() in {"__", "--", "==", "**", "||", "//", "\\\\"} else 0,  
            1 if re.match(r'^\s*\d+\.\d+\s*\|+\*+', text) else 0,  
            1 if sum(1 for c in text if not c.isalnum()) / max(1, len(text)) > 0.7 else 0, 
            1 if text.count("*") > 5 or text.count("=") > 5 else 0, 
            1 if re.search(r'^\s*\d+\.\d+\s*\|+\*+', text) else 0,  


            # HEADL
            sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, num_chars), 
            1 if 45 <= sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, num_chars) <= 55 else 0, 
            1 if abs(sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, num_chars) - 50) < 5 else 0, 
            sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isupper()) + sum(1 for c in text if c.islower())),  
            1 if text.isupper() else 0, 
            1 if text.strip().endswith(('.', '?', '!')) == False else 0, 
            1 if sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())) > 0.7 else 0,  
            1 if len(text) < 50 else 0,  
            1 if any(text.startswith(w) for w in ["ANNOUNCING", "CALL FOR", "NOW HIRING", "FINAL EXAM RESULTS", "CONFERENCE"]) else 0,  
            1 if text.strip() in {"***", "=====", "-----", "~~~~~"} else 0,  
            1 if text.count("*") > 3 or text.count("=") > 3 or text.count("-") > 3 else 0,  
            1 if len(set(text.strip())) < 6 and any(c in text for c in "-=*~") else 0,  
            1 if re.search(r'\b(FELLOWSHIP|SCHOLARSHIP|FACULTY POSITION|GRADUATE PROGRAM|CALL FOR PAPERS)\b', text) else 0, 
            1 if re.search(r'\b(MEETING|EVENT|LUNCH|SEMINAR|WORKSHOP|LECTURE|CONFERENCE)\b', text) else 0, 
            1 if re.search(r'\b(DEPARTMENT OF|UNIVERSITY OF|FACULTY OF|SCHOOL OF)\b', text) else 0,  
            1 if sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, len(text)) > 0.45 and sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, len(text)) < 0.55 else 0,  
            1 if len(text.split("\n")) < 3 else 0, 
            1 if text.strip().endswith((" 1993", " 1994", " 1995")) else 0, 

            # ADDRESS
            1 if any(kw in text for kw in ["tel", "fax", "e-mail", "route", "P.O. Box", "street", "avenue", "drive"]) else 0,
            1 if any(w in text.lower() for w in ["university", "department", "institute", "college", "school", "faculty"]) else 0,  
            1 if any(w in text.lower() for w in ["street", "avenue", "blvd", "road", "drive", "lane", "route", "highway"]) else 0, 
            1 if "P.O. Box" in text or "Suite" in text else 0,  
            1 if re.search(r'\d{1,5} [A-Za-z]+ (Street|Avenue|Blvd|Drive|Road|Lane|Route|Highway)', text) else 0,
            1 if re.search(r'\b[A-Z][a-z]+,? [A-Z]{2} \d{5}(-\d{4})?\b', text) else 0, 
            1 if re.search(r'\(\d{3}\) \d{3}-\d{4}', text) else 0,  
            1 if re.search(r'Fax:?\s*\(?\d{3}\)?\s*\d{3}-\d{4}', text, re.I) else 0,  
            1 if re.search(r'Email:?\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text, re.I) else 0,  
            1 if len(re.findall(r'\d{5}(-\d{4})?', text)) > 0 else 0, 
            1 if text.strip().startswith(("ATTN:", "Chair", "Faculty Search Committee", "Professor", "Director")) else 0,  
            1 if sum(1 for c in text if c.isdigit()) > 4 else 0, 
            1 if len(words) > 3 and num_spaces / max(1, num_words) > 1.5 else 0,  

            # ITEM
            1 if text.lstrip().startswith(("1.", "(1)", "- ")) else 0, 
            1 if text.lstrip().startswith(("1.", "(1)", "- ", "* ")) else 0, 
            1 if re.match(r'^\(?\d{1,2}\)?[\.\-)]\s', text) else 0,  
            1 if re.match(r'^\s*[\-\*]\s', text) else 0,  
            1 if len(text.split("\n")) > 1 and all(line.strip().startswith(("-", "*", "(")) or line.strip()[0].isdigit() for line in text.split("\n")) else 0, 
            1 if len(text) < 100 and text.count(",") >= 2 else 0, 
            1 if sum(1 for c in text if c.isdigit()) / max(1, len(text)) > 0.2 else 0, 
            1 if re.search(r'\b(?:step|point|rule|factor|reason|feature|aspect|method)\b', text.lower()) else 0,  
            1 if text.count("\t") > 1 else 0,  
            1 if len(set(text.strip())) < 10 and any(c in text for c in "0123456789-*)") else 0,  
            1 if len(text.split()) < 15 else 0, 


            # PTEXT
            1 if any(w in {"the", "this", "he", "she", "it", "i", "but", "and"} for w in words[:2]) else 0, 
            1 if any(w.lower() in {"said", "asked", "concluded", "replied", "noted", "wrote"} for w in words) else 0, 
            len(text), 
            len(words), 
            text.count('.') + text.count('!') + text.count('?'),  
            len(words) / max(1, text.count('.') + text.count('!') + text.count('?')), 
            sum(1 for w in words if w.lower() in {"i", "you", "he", "she", "it", "we", "they"}) / max(1, len(words)),  
            sum(1 for w in words if w.lower() in {"this", "that", "these", "those"}) / max(1, len(words)),  
            sum(1 for w in words if w.lower() in {"but", "and", "or", "because"}) / max(1, len(words)),  
            sum(1 for w in words if w.lower() in {"can", "could", "will", "would", "should"}) / max(1, len(words)), 
            sum(1 for w in words if w.lower() in {"in", "on", "at", "by", "with", "about", "as", "of"}) / max(1, len(words)),  
            text.count("\n"),  
            sum(len(line) for line in text.split("\n")) / max(1, len(text.split("\n"))),  
            1 if text.lstrip().startswith(("- ", "* ", "•", "1.")) else 0,  
            1 if text.startswith(" ") else 0,  
            1 if text.endswith(('.', '!', '?')) else 0,  
            sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())),  
            1 if any(w.isupper() for w in words) else 0,  
            sum(1 for w in words if '-' in w) / max(1, len(words)),  
            1 if any(w in text.lower() for w in {"opportunity", "analysis", "development"}) else 0, 
            1 if any(w in text.lower() for w in {"research", "university", "fellowship", "study", "professor"}) else 0, 
            1 if re.search(r'\bwas \w+ed\b', text) else 0,  
            1 if any(w in text.lower() for w in {"see below", "refer to", "contact"}) else 0,  
            1 if any(w in text.lower() for w in {"we invite", "applications are open"}) else 0,  
            1 if any(w in text.lower() for w in {"exciting opportunity", "highly competitive"}) else 0,  
            1 if any(w in text.lower() for w in {"requirements", "qualifications", "position available"}) else 0,  
            1 if re.search(r'\b\d{4}\b', text) else 0,  
            text.count(','),  
            text.count(':') + text.count(';'),  
            1 if '"' in text or "'" in text else 0,  
            1 if "(" in text and ")" in text else 0,  
            1 if text.count(".") / max(1, len(words)) > 0.1 else 0, 


            1 if num_chars < 3 else 0,  

            # **General Sentence Structure
            len(text.split("\n")),  
            1 if len(text.split("\n")) < 2 else 0,
            1 if len(text.split("\n")) > 3 else 0, 
            len(text) - len(text.lstrip()),
            1 if (len(text) - len(text.lstrip())) in [0, 1, 2] else 0, 
            sum(len(line) for line in text.split("\n")[:-1]) / max(1, len(text.split("\n")) - 1),  
            1 if sum(len(line) for line in text.split("\n")[:-1]) / max(1, len(text.split("\n")) - 1) > 70 else 0, 
            num_spaces / max(1, num_words),  
            1 if num_spaces / max(1, num_words) < 1.3 else 0,  

            # General 
            1 if text.endswith(('.', '!', '?')) else 0, 
            1 if len(words) > 1 and words[-1] in ['.', '..', '...'] else 0, 
            min(num_chars // 10, 10),  


            # QUOTED cont
            1 if text.lstrip().startswith("KC>") else 0,  
            text.count(">") if text.lstrip().startswith(">") else 0,  
            1 if any(w in text for w in ["wrote:", "writes:", "said:", "noted:"]) else 0, 
            1 if text.lstrip().startswith(">") else 0, 
            1 if text.lstrip().startswith("KC>") else 0,  
            1 if text.count(">") > 3 else 0,  
            1 if ">" in text and text.strip().endswith(":") else 0, 
            1 if any(w in text.lower() for w in ["previous message", "original post", "replied:"]) else 0, 


            #HEADL
            1 if text.strip().endswith(":") and not text.startswith(">") else 0,  
            1 if sum(1 for w in words if w.isupper()) / max(1, len(words)) > 0.6 else 0,  


            # SIG cont
            1 if text.startswith("-- ") else 0,  
            1 if len(words) < 8 and any(w.lower() in ["ceo", "founder", "engineer", "professor", "developer"] for w in words) else 0,  
            1 if len(words) < 8 and any(w.lower() in ["university", "institute", "lab", "dept", "corporation", "company", "division"] for w in words) else 0,  
            1 if text.isupper() and len(words) < 10 else 0, 

            1 if text.startswith("-- ") else 0, 
            1 if "@" in text and len(words) < 6 else 0,  
            1 if any(w.lower() in text for w in ["phone:", "fax:", "contact:", "email:", "www."]) else 0,  
            1 if text.count("|") > 2 else 0,  
            1 if sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())) > 0.6 else 0,  


            # ITEM cont
            1 if text.lstrip().startswith(("1.", "(1)", "- ", "* ")) else 0,  
            1 if re.match(r'^\s*[\-\*]\s', text) else 0,  
            1 if len(text.split("\n")) > 1 and all(line.strip().startswith(("-", "*", "(")) or line.strip()[0].isdigit() for line in text.split("\n")) else 0, 
            1 if text.lstrip().startswith(("1.", "2.", "3.", "(1)", "- ", "* ")) else 0,  
            1 if re.match(r'^\s*\d+\.\s+', text) else 0, 
            1 if len(re.findall(r'\d+', text)) > 3 else 0,  


            # GRAPHIC vs. TABLE 
            1 if text.count("|") > 5 else 0,  
            1 if sum(1 for c in text if not c.isalnum()) / max(1, len(text)) > 0.5 else 0,  
            1 if len(set(text.strip())) < 6 and any(c in text for c in "-=*|~") else 0,  

            # HEADL cont
            1 if text.isupper() else 0, 
            1 if text.strip().endswith(('.', '?', '!')) == False else 0, 
            1 if any(text.startswith(w) for w in ["ANNOUNCING", "CALL FOR", "NOW HIRING", "FINAL EXAM RESULTS", "CONFERENCE"]) else 0, 

        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


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
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
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