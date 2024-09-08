import os
from groq import Groq
import argparse

# https://console.groq.com/docs/rate-limits
request_limit = 14400  # Requests per day limit
token_limit = 18000  # Tokens per minute limit
requests_remaining = 14370  # Remaining requests per day
tokens_remaining = 17997  # Remaining tokens per minute
request_reset_time = 179.56  # Time to reset requests (in seconds)
token_reset_time = 7.66  # Time to reset tokens (in seconds)

def split_document_into_chunks(text, max_token_size=token_limit):
    """
    Split the text into chunks with each chunk having no more than max_token_size tokens.
    """
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for word in text.split():
        # Approximate tokens by word length
        if current_chunk_size + len(word) > max_token_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_size = len(word)
        else:
            current_chunk.append(word)
            current_chunk_size += len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize(client, text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': (
                    'Summarize the input text below. '
                    'Limit the summary to 1 paragraph.'
                    'Just output the summary do not return any commentary or other remarks.'
                )
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model="llama3-8b-8192",
    )
    summary = chat_completion.choices[0].message.content
    return summary

if __name__ == '__main__':
    # parse input file
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    with open(args.filename) as f:
        text = f.read()

    summary = summarize(client, text)
    print(summary)
