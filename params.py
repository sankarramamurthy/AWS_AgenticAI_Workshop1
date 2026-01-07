# Inference Parameters are used to configure the response behavior of the foundation model.
# Inference parameters vary from model to model
# they can be used to influence the variability (Temperature, Top P), token length of the response etc.

import sys
import boto3
#

def get_text_response(model, input_content):

    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    message = {
        "role": "user",
        "content": [ { "text": input_content } ]
    }
    
    response = bedrock.converse(
        modelId=model,
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
    )
    
    return response['output']['message']['content'][0]['text']
    

response = get_text_response(sys.argv[1], sys.argv[2])

print(response)

# You can run this script by passing various models as parameter
# python3 params.py "us.amazon.nova-pro-v1:0" "Please write a haiku:"
# python3 params.py "mistral.mixtral-8x7b-instruct-v0:1" "Write a haiku:" 
# python3 params.py "cohere.command-r-plus-v1:0" "Write a haiku:"
