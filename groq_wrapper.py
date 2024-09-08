import os
from groq import Groq
import time 

# https://console.groq.com/docs/rate-limits
request_limit = 14400  # Requests per day limit
token_limit = 18000  # Tokens per minute limit
requests_remaining = 14370  # Remaining requests per day
tokens_remaining = 17997  # Remaining tokens per minute
request_reset_time = 179.56  # Time to reset requests (in seconds)
token_reset_time = 7.66  # Time to reset tokens (in seconds)

class Groq_Wrapper:
    def __init__(self):
        # initilize groq client 
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        # used to keep track of the number of requests remaining in at current rate
        self.requests_remaining = requests_remaining
        self.tokens_remaining = tokens_remaining

    def enforce_rate_limits(self):
        """
        Enforce rate limits based on remaining requests and tokens.
        """
        if self.requests_remaining <= 0:
            print(f"Requests exhausted. Sleeping for {request_reset_time} seconds.")
            time.sleep(request_reset_time)
            self.requests_remaining = request_limit
        
        if self.tokens_remaining <= 0:
            print(f"Tokens exhausted. Sleeping for {token_reset_time} seconds.")
            time.sleep(token_reset_time)
            self.tokens_remaining = token_limit
    
    
    def summarize_chunk(self, chunk, max_retries=3):
        """
        Summarizes an individual chunk, ensuring rate limits are respected.
        Retries the request if rate limits are exceeded, waiting 10 seconds before retrying.
        """
        retries = 0
        while retries < max_retries:
            try:
                self.enforce_rate_limits()

                # Make the API call to summarize the chunk
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                'Summarize the input text below. '
                                'Limit the summary to 1 paragraph.'
                                'Just output the summary, do not return any commentary or other remarks.'
                            )
                        },
                        {
                            "role": "user",
                            "content": chunk,
                        }
                    ],
                    model="llama3-8b-8192",
                )
                summary = chat_completion.choices[0].message.content
                
                # Update remaining requests and tokens
                self.requests_remaining -= 1
                self.tokens_remaining -= len(chunk.split())
                
                return summary  # Return the summary if successful
            
            except Exception as e:
                # Check if the error is due to rate limiting
                retries += 1
                print(f"Rate limit exceeded. Retrying in 10 seconds... ({retries}/{max_retries})")
                time.sleep(token_reset_time + 3)  
    
    def summarize_chunks(self, chunks):
        """
        Break down the large text into chunks and summarize them.
        Then summarize the chunk summaries into a final summary.
        """
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk)
            chunk_summaries.append(summary)
        
        # Summarize the summaries
        final_summary = self.summarize_chunk(' '.join(chunk_summaries))
        
        return final_summary