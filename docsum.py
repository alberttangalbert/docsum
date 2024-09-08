import argparse
import chardet 

from groq_wrapper import Groq_Wrapper

def split_text(text, max_token_size):
    """
    Split the text into chunks with each chunk having no more than max_token_size tokens.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for word in words:
        word_length = len(word)  # Approximate tokens by word length
        if current_chunk_size + word_length > max_token_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_size = word_length
        else:
            current_chunk.append(word)
            current_chunk_size += word_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

if __name__ == '__main__':
    # parse input file
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    # read text input file 
    with open('your_file', 'rb') as f:
        text = chardet.detect(f.read())

    # split text into chunks 
    max_token_size = 18000
    chunks = split_text(text, max_token_size)

    # summarize chunks into one ful
    groq_wrapper = Groq_Wrapper()
    summary = groq_wrapper.summarize_chunks(chunks)
    
    print("Summary:")
    print("--------\n")
    print(summary)
