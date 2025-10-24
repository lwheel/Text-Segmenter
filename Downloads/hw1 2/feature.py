import re

# Define text and words as placeholders for illustration
text = "Example text"
words = text.split()
num_words = len(words)


features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            text.count(' '),
            text.count(' ') / max(1, len(text)),  
            sum(1 if w.isupper() else 0 for w in words),

            # **NNHEAD (Email Header)**
            1 if text.startswith(("From:", "Subject:", "Date:", "Message-ID:", "Organization:", "Keywords:")) else 0,
            1 if "@" in text and "." in text.split("@")[-1] else 0,
            1 if any(w.lower() in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"} for w in words) else 0,
            1 if "<" in text and ">" in text and "@" in text else 0,
            1 if ":" in text and any(c.isdigit() for c in text) else 0,
            1 if "Organization:" in text else 0,
            1 if text.startswith("Subject: Re:") else 0,
            1 if "(" in text and ")" in text else 0,
            1 if any(c.isdigit() for c in text) else 0,
            min(len(text) // 10, 10),

            # **QUOTED Features**
            1 if any(text.endswith(ending) for ending in ["wrote:", "writes:", "write:", "said:"]) else 0,
            1 if text.lstrip().startswith((">", "@", ":")) else 0,
            text.lstrip().count(">"),
            1 if any(w in text for w in ["wrote:", "writes:", "said:", "noted:"]) else 0,
            1 if any(w in text for w in ["In article", "On ", "replied:"]) else 0,
            1 if "(" in text and ")" in text and "@" in text else 0,
            1 if text.endswith(":") else 0,
            1 if words and words[0][0].isupper() else 0,
            1 if text.lstrip().startswith(">>") else 0,
            1 if text.count(">") >= 3 else 0,
            1 if text.startswith("KC>") else 0,
            1 if text.strip().endswith(":") else 0,
            1 if text.lstrip().startswith(": ") else 0,
            1 if text.lstrip().startswith("[") and text.rstrip().endswith("]") else 0,
            1 if any(w in text.lower() for w in ["previous message", "original post", "replied:"]) else 0,

            # **SIG (Signature Block)**
            1 if text.startswith(("-", "=", "*", "#", "_")) and len(set(text.strip())) <= 3 else 0,
            1 if text.count("|") > 2 else 0,
            1 if any(w in text.lower() for w in ["tel:", "phone:", "fax:", "e-mail:", "mailto:", "contact:"]) else 0,
            1 if any(w in text.lower() for w in ["http://", "https://", "www."]) else 0,
            1 if text.strip().endswith("--") else 0,
            1 if sum(1 for c in text if c.isdigit()) > 4 else 0,
            1 if text.lstrip().startswith(("-", "=")) and len(set(text.strip())) <= 3 else 0,
            1 if len(words) < 5 and "@" in text else 0,
            1 if len(words) < 8 and any(w.lower() in ["ceo", "founder", "engineer", "professor", "developer", "student", "researcher"] for w in words) else 0,
            1 if len(words) < 8 and any(w.lower() in ["university", "institute", "lab", "dept", "corporation", "company", "division"] for w in words) else 0,
            1 if sum(1 for c in text if c in ["*", "=", "-", "_", "~"]) / max(1, len(text)) > 0.3 else 0,
            1 if any(w in text for w in ["Best,", "Regards,", "Sincerely,", "Thanks,", "Cheers,", "Yours,"]) else 0,
            1 if any(w in text for w in ["P.S.", "Sent from", "Please consider", "Think before you print"]) else 0,

            # **BLANK (Whitespace Line)**
            1 if text.strip() == "" else 0,

            # **TABLE (Tabular Data)**
            1 if text.lstrip().startswith(tuple("0123456789")) else 0,
            1 if text.count(" ") / max(1, len(words)) < 1.3 else 0,
            1 if re.match(r'^\s*\d+[\s\t]+', text) else 0,
            1 if re.match(r'^\s*[-=]+$', text) else 0,
            1 if len(set(text.strip())) < 6 and any(c in text for c in '-=|') else 0,
            1 if sum(1 for c in text if c.isdigit()) > (0.2 * len(text)) else 0,
            text.count('\t') + text.count('  '),
            1 if text.count('\t') > 2 else 0,
            1 if sum(1 for c in text if c.isspace()) / max(1, len(text)) > 0.3 else 0,

            # **HEADL (Headlines)**
            sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, len(text)),
            1 if 45 <= sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, len(text)) <= 55 else 0,
            1 if text.isupper() else 0,
            1 if text.strip().endswith(('.', '?', '!')) == False else 0,
            1 if len(text) < 50 else 0,
            1 if any(text.startswith(w) for w in ["ANNOUNCING", "CALL FOR", "NOW HIRING", "FINAL EXAM RESULTS", "CONFERENCE"]) else 0,

            # **PTEXT (Plain Text)**
            1 if any(w in {"the", "this", "he", "she", "it", "i", "but", "and"} for w in words[:2]) else 0,
            1 if any(w.lower() in {"said", "asked", "concluded", "replied", "noted", "wrote"} for w in words) else 0,
            text.count('.') + text.count('!') + text.count('?'),
            len(words) / max(1, text.count('.') + text.count('!') + text.count('?')),
            sum(1 for w in words if w.lower() in {"i", "you", "he", "she", "it", "we", "they"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"this", "that", "these", "those"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"but", "and", "or", "because"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"can", "could", "will", "would", "should"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"in", "on", "at", "by", "with", "about", "as", "of"}) / max(1, len(words)),

            # **UNCERTAIN (#???#)**
            1 if len(text) < 3 else 0,

            # **General Structural Features**
            len(text.split("\n")),
            1 if len(text.split("\n")) < 2 else 0,
            1 if len(text.split("\n")) > 3 else 0,
            len(text) - len(text.lstrip()),
            1 if (len(text) - len(text.lstrip())) in [0, 1, 2] else 0,

            # **Final General Indicators**
            1 if len(words) > 1 and words[-1] in ['.', '..', '...'] else 0,
            min(len(text) // 10, 10),

            # **QUOTED Detection**
            1 if text.lstrip().startswith("KC>") else 0,
            text.count(">") if text.lstrip().startswith(">") else 0,
            1 if text.lstrip().startswith(">") else 0,

            # **HEADL Refinement**
            1 if text.strip().endswith(":") and not text.startswith(">") else 0,
            1 if sum(1 for w in words if w.isupper()) / max(1, len(words)) > 0.6 else 0,

            # **SIG Refinement**
            1 if text.startswith("-- ") else 0,
            1 if "@" in text and len(words) < 6 else 0,
            1 if any(w.lower() in text for w in ["phone:", "fax:", "contact:", "email:", "www."]) else 0,


            1 if text.isupper() and len(words) < 8 else 0,  # Short all-uppercase phrases
            1 if text.strip().endswith(":") else 0,
            1 if len(text.split()) < 5 and text.istitle() else 0,
            1 if re.search(r'\b\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?', text) else 0,

            # HEADL (Headlines)
            1 if text.isupper() and num_words < 10 else 0,
            1 if re.match(r'^\s*[A-Z0-9].*[A-Z0-9]\s*$', text) and num_words < 12 else 0,
            1 if text.strip().endswith(':') else 0,
            1 if any(text.startswith(w) for w in ["ANNOUNCING", "CALL FOR", "NOW HIRING", "FINAL EXAM RESULTS", "CONFERENCE"]) else 0,
            1 if 45 <= sum(i for i, c in enumerate(text) if not c.isspace()) / max(1, len(text)) <= 55 else 0,

            # ITEM (List/Numbered items)
            1 if re.match(r'^\s*(\d+[\).]|[-*])\s+', text) else 0,
            1 if text.strip().startswith(("•", "- ", "* ", "1. ", "2. ", "3. ")) else 0,
            1 if len(words) < 12 and re.search(r'\d+\.', text) else 0,

            # TABLE (Tabular Data)
            1 if text.count(" ") / max(1, num_words) < 1.3 else 0,
            1 if re.match(r'^\s*\d+[\s\t]+', text) else 0,
            1 if re.match(r'^\s*[-=]+$', text) else 0,
            1 if len(set(text.strip())) < 6 and any(c in text for c in '-=|') else 0,
            1 if sum(1 for c in text if c.isdigit()) > (0.2 * len(text)) else 0,
            text.count('\t') + text.count('  '),
            1 if text.count('\t') > 2 else 0,
            1 if sum(1 for c in text if c.isspace()) / max(1, len(text)) > 0.3 else 0,

            # SIG (Signature Block)
            1 if text.startswith(("-", "=", "*", "#", "_")) and len(set(text.strip())) <= 3 else 0,
            1 if text.count("|") > 2 else 0,
            1 if any(w in text.lower() for w in ["tel:", "phone:", "fax:", "e-mail:", "mailto:", "contact:"]) else 0,
            1 if any(w in text.lower() for w in ["http://", "https://", "www."]) else 0,
            1 if text.strip().endswith("--") else 0,
            1 if sum(1 for c in text if c.isdigit()) > 4 else 0,
            1 if text.startswith("-- ") else 0,
            1 if "@" in text and len(words) < 6 else 0,
            1 if any(w.lower() in text for w in ["phone:", "fax:", "contact:", "email:", "www."]) else 0,
            1 if text.isupper() and len(words) < 8 else 0,
            1 if text.strip().endswith(":") else 0,
            1 if len(text.split()) < 5 and text.istitle() else 0,

            # QUOTED (Quoted Text)
            1 if text.lstrip().startswith(">") else 0,
            text.count(">") if text.lstrip().startswith(">") else 0,
            1 if any(text.endswith(ending) for ending in ["wrote:", "writes:", "write:", "said:"]) else 0,
            1 if any(w in text for w in ["wrote:", "writes:", "said:", "noted:"]) else 0,
            1 if text.lstrip().startswith((">", "@", ":")) else 0,
            1 if text.lstrip().startswith(">>") else 0,
            1 if text.count(">") >= 3 else 0,

            # PTEXT (Plain Text)
            text.count('.') + text.count('!') + text.count('?'),
            len(words) / max(1, text.count('.') + text.count('!') + text.count('?')),
            sum(1 for w in words if w.lower() in {"i", "you", "he", "she", "it", "we", "they"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"this", "that", "these", "those"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"but", "and", "or", "because"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"can", "could", "will", "would", "should"}) / max(1, len(words)),
            sum(1 for w in words if w.lower() in {"in", "on", "at", "by", "with", "about", "as", "of"}) / max(1, len(words)),

            # BLANK (Whitespace Line)
            1 if text.strip() == "" else 0,

            # Address Detection
            1 if "@" in text and "." in text.split("@")[-1] else 0,
            1 if any(w.lower() in ["university", "institute", "lab", "dept", "corporation", "company", "division"] for w in words) else 0,

            # HEADL Refinement
            1 if text.strip().endswith(":") and not text.startswith(">") else 0,
            1 if sum(1 for w in words if w.isupper()) / max(1, len(words)) > 0.6 else 0,

            # Uncertain Cases
            1 if len(text) < 3 else 0,
                   
                   
            1 if text.isupper() and num_words < 10 else 0,
            1 if re.match(r'^\s*[A-Z0-9].*[A-Z0-9]\s*$', text) and num_words < 12 else 0,
            1 if text.strip().endswith(':') else 0,
            1 if any(text.startswith(w) for w in ["ANNOUNCING", "CALL FOR", "FINAL EXAM RESULTS", "CONFERENCE"]) else 0,
            1 if sum(1 for w in words if w.isupper()) / max(1, len(words)) > 0.6 else 0,
            1 if len(text) < 50 else 0,  # Headlines are usually short

            # **ITEM (List/Numbered items)**
            1 if re.match(r'^\s*(\d+[\).]|[-*])\s+', text) else 0,
            1 if text.strip().startswith(("•", "- ", "* ", "1. ", "2. ", "3. ")) else 0,
            1 if len(words) < 12 and re.search(r'\d+\.', text) else 0,
            
            # **TABLE (Tabular Data)**
            1 if text.count(" ") / max(1, num_words) < 1.3 else 0,
            1 if re.match(r'^\s*\d+[\s\t]+', text) else 0,
            1 if re.match(r'^\s*[-=]+$', text) else 0,
            1 if len(set(text.strip())) < 6 and any(c in text for c in '-=|') else 0,
            1 if sum(1 for c in text if c.isdigit()) > (0.2 * len(text)) else 0,
            text.count('\t') + text.count('  '),
            1 if text.count('\t') > 2 else 0,


            # **SIG (Signature Block)**
            1 if text.startswith("-- ") else 0,
            1 if "@" in text and len(words) < 6 else 0,
            1 if text.strip().endswith("--") else 0,
            1 if text.count("|") > 2 else 0,
            1 if any(w in text.lower() for w in ["phone:", "fax:", "contact:", "email:", "www."]) else 0,
            1 if len(text) < 50 and text.strip().startswith("-") else 0,
            
            



        ]

print("features = [")
for feature in features:
    print(f'    "{feature}",')
print("]")