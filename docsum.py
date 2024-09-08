import os
from groq import Groq
import argparse

def split_document_into_chunks(text):
    r'''
    Split the input text into smaller chunks so that an LLM can process those chunks individually.

    >>> split_document_into_chunks('This is a sentence.\n\nThis is another paragraph.')
    ['This is a sentence.', 'This is another paragraph.']
    >>> split_document_into_chunks('This is a sentence.\n\nThis is another paragraph.\n\nThis is a third paragraph.')
    ['This is a sentence.', 'This is another paragraph.', 'This is a third paragraph.']
    >>> split_document_into_chunks('This is a sentence.')
    ['This is a sentence.']
    >>> split_document_into_chunks('')
    []
    >>> split_document_into_chunks('This is a sentence.\n')
    ['This is a sentence.']
    >>> split_document_into_chunks('This is a sentence.\n\n')
    []
    >>> split_document_into_chunks('This is a sentence.\n\nThis is another paragraph.\n\n')
    ['This is a sentence.', 'This is another paragraph.']''

    '''
    return text.split('\n\n')

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
