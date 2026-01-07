# Tool use is a capability that allows a large language model to tell the calling application to invoke a function with parameters supplied by the model
# The available functions and supported parameters are passed to the model along with a prompt
# The LLM does not call a function itself; it just returns JSON and lets the calling application do the rest.
# Tool Use with the Amazon Bedrock Converse API follows these steps:
# 1. The calling application passes (A) tool definitions and (B) a triggering message to the LLM
# 2. If the request matches a tool definition, the model generates a tool use request, including the parameters to pass to the tool.
# 3. The calling application extracts the parameters from the model’s tool use request and passes them to the corresponding local function for the tool.
# 4. The calling application can then either use the tool result directly, or pass the tool result back to the model to get a follow-on response.
# 5. The model either returns a final response, or requests another tool.

import boto3, json, math

print("\n----Defining a tool and sending a message that will make Claude ask for tool use----\n")

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

# create a simple message to trigger the tool use request and add it to an empty list of messages.
# We’re creating a message from the “user” role.
# Within that message, we can include a list of content blocks.
# In this example, we have a single text content block where we ask the model "What is the cosine of 7?"
# We specify Anthropic’s Claude 3 Sonnet as the target model.
# We can limit the number of tokens in the model’s response by setting the maxTokens value.
# We also set the temperature to zero to minimize the variability of responses.
# we also set a system message here so that Claude won’t attempt to do any math itself; The current generation of LLMs cannot reliably do math

tool_list = [
    {
        "toolSpec": {
            "name": "cosine",
            "description": "Calculate the cosine of x.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to pass to the function."
                        }
                    },
                    "required": ["x"]
                }
            }
        }
    }
]

message_list = []

initial_message = {
    "role": "user",
    "content": [
        { "text": "What is the cosine of 7?" } 
    ],
}

message_list.append(initial_message)

response = bedrock.converse(
    modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=message_list,
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0
    },
    toolConfig={
        "tools": tool_list
    },
    system=[{"text":"You must only do math by using a tool."}]
)

response_message = response['output']['message']
print(json.dumps(response_message, indent=4))
message_list.append(response_message)

# In the output generated, you can observe that Claude also generated some text prefacing its tool use request. Claude will only do this some of the time. Sometimes it just generates a tool use request with no text accompanying it.
# The toolUse block includes a toolUseId, in this case cosine
# The input property contains the JSON structure of arguments to pass to the tool. You can also use this JSON directly. In this case, Claude is asking the calling application to pass the cosine function an argument x with value 7


# We’ll now loop through the response message’s content blocks. We’ll use the cosine tool if requested, and print any text content blocks from the LLM’s message

print("\n----Calling a function based on the toolUse content block.----\n")

response_content_blocks = response_message['content']

for content_block in response_content_blocks:
    if 'toolUse' in content_block:
        tool_use_block = content_block['toolUse']
        tool_use_name = tool_use_block['name']
        
        print(f"Using tool {tool_use_name}")
        
        if tool_use_name == 'cosine':
            tool_result_value = math.cos(tool_use_block['input']['x'])
            print(tool_result_value)
            
    elif 'text' in content_block:
        print(content_block['text'])

# Now we will send a follow-up request to Claude to get a final response.
# we’ll loop through the content blocks from the response message, and check for a tool use request. 
# If there’s a tool use request, we’ll call the named tool and pass it the input parameters provided by Claude.
# We’ll then build a message with a toolResult content block to send back to Claude for a final response.

print("\n----Passing the tool result back to Claude----\n")

follow_up_content_blocks = []

for content_block in response_content_blocks:
    if 'toolUse' in content_block:
        tool_use_block = content_block['toolUse']
        tool_use_name = tool_use_block['name']
        
        
        if tool_use_name == 'cosine':
            tool_result_value = math.cos(tool_use_block['input']['x'])
            
            follow_up_content_blocks.append({
                "toolResult": {
                    "toolUseId": tool_use_block['toolUseId'],
                    "content": [
                        {
                            "json": {
                                "result": tool_result_value
                            }
                        }
                    ]
                }
            })

if len(follow_up_content_blocks) > 0:
    
    follow_up_message = {
        "role": "user",
        "content": follow_up_content_blocks,
    }
    
    message_list.append(follow_up_message)

    response = bedrock.converse(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=message_list,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0
        },
        toolConfig={
            "tools": tool_list
        },
        system=[{"text":"You must only do math by using a tool."}]
    )
    
    response_message = response['output']['message']
    
    message_list.append(response_message)
    print(json.dumps(message_list, indent=4))

# Now we’re going to take a step back and manufacture an error to send back to the LLM. 
# We set the status attribute to error so that Claude can decide what to do next

print("\n----Error handling - letting Claude know that tool use failed----\n")

del message_list[-2:] #Remove the last request and response messages

content_block = next((block for block in response_content_blocks if 'toolUse' in block), None)

if content_block:
    tool_use_block = content_block['toolUse']
    
    error_tool_result = {
        "toolResult": {
            "toolUseId": tool_use_block['toolUseId'],
            "content": [
                {
                    "text": "invalid function: cosine"
                }
            ],
            "status": "error"
        }
    }
    
    follow_up_message = {
        "role": "user",
        "content": [error_tool_result],
    }
    
    message_list.append(follow_up_message)
    
    response = bedrock.converse(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=message_list,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0
        },
        toolConfig={
            "tools": tool_list
        },
        system=[{"text":"You must only do math by using a tool."}]
    )
    
    response_message = response['output']['message']
    print(json.dumps(response_message, indent=4))
    message_list.append(response_message)

