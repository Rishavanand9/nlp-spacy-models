import spacy

# Load a spaCy language model
nlp = spacy.load("en_core_web_sm")

def get_parsed_output(line):
    # Use the spaCy model to parse the line
    doc = nlp(line)
    for token in doc:
        print (token)
    return ''

if __name__ == '__main__':
    with open('test_dataset.txt', 'r') as file:
        for line in file:
            line = line.strip('\n') + ' '
            parsed_output = get_parsed_output(line)
            # Process the parsed output as needed
            print(parsed_output)
