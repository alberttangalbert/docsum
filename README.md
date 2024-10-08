# docsum ![](https://github.com/alberttangalbert/docsum/workflows/tests/badge.svg)

Use an LLM to summarize a document on the command line.

The following example summarizes the declaration of independence.

Steps to run:
1) Create .env file (example given) and enter GROQ api key
2) pip install -r requirements.txt
3) python3 docsum/docsum.py docsum/docs/declaration.txt

Output:
A long time ago, some people in America decided they wanted to be free. They were tired of being ruled by a king from a faraway land. The king was being mean and taking away their rights. The people wrote a special paper to say they were free and independent. They said they didn't want to be ruled by the king anymore. They promised to work together and take care of each other to make sure they stayed safe and happy. This paper is called the Declaration of Independence. It's like a big declaration that says, "We are free and we're going to make our own decisions!"

Notes:
You can put any filepath as an argument 
If the file is too large, script will break it down into chuncks summarize the smaller chunks.
The script will then output all final summary of all the smaller chunks.
