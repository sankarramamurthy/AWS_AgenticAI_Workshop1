# Streaming responses are useful when you want to start returning content immediately to the end user. 
# You can display the output a few words at a time, instead of waiting for the entire response to be created.

import boto3

# Define the callback handler for streaming results.
# This function allows us to print response chunks as they are returned from the streaming api.

def chunk_handler(chunk):
    print(chunk, end='')

# Define the function to call the Amazon Bedrock streaming API
# We use Amazon Bedrock's converse_stream function to make the call to the streaming API endpoint
# As response chunks are returned, this code extracts the chunk's text from the returned JSON and passes it to the provided callback function

def get_streaming_response(prompt, streaming_callback):
    
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    message = {
        "role": "user",
        "content": [ { "text": prompt } ]
    }
    
    response = bedrock.converse_stream(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.0
        }
    )
    
    stream = response.get('stream')
    for event in stream:
        if "contentBlockDelta" in event:
            streaming_callback(event['contentBlockDelta']['delta']['text'])

# Display the response

prompt = "Tell me a story about two puppies and two kittens who became best friends:"
                
get_streaming_response(prompt, chunk_handler)
print("\n")
