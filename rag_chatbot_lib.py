# In this script, we shall build a chatbot supported by Retrieval-Augmented Generation (RAG). 
# We'll use Anthropic Claude, Amazon Titan Embeddings, and Streamlit. 
# We will use Amazon Bedrock's built-in tool use capabilities to allow Anthropic Claude to decide when to use the RAG pattern.
# We will use a local Chroma  database to demonstrate the RAG pattern

# The chatbot with RAG pattern is good for: Simple interactive user conversation, supported by specialized knowledge or data
# Past interactions are tracked in the chat memory object
# When the user enters a new message, The chat history is retrieved from the memory object and added before the new message
# The question is converted to a vector using Amazon Titan Embeddings, then matched to the closest vectors in the vector database
# The combined history, knowledge, and new message are sent to the model
# The model's response is displayed to the user.

import itertools
import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

MAX_MESSAGES = 20 # sets the upper limit for previous chat messages kept in memory

class ChatMessage(): #create a class that can store image and text messages
    def __init__(self, role, text):
        self.role = role
        self.text = text

# a function to connect to the ChromaDB collection
# This will allow us to access the previously created Chroma vector database

def get_collection(path, collection_name):
    session = boto3.Session()
    embedding_function = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2:0")
    
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name, embedding_function=embedding_function)
    
    return collection

# A function to retrieve results from the vector store

def get_vector_search_results(collection, question):
    
    results = collection.query(
        query_texts=[question],
        n_results=4
    )
    
    return results

# a function to create the tool definitions we will use to define the format for the generated JSON

def get_tools():
    tools = [
        {
            "toolSpec": {
                "name": "get_amazon_bedrock_information",
                "description": "Retrieve information about Amazon Bedrock, a managed service for hosting generative AI models.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The retrieval-augmented generation query used to look up information in a repository of FAQs about Amazon Bedrock."
                            }
                        },
                        "required": [
                            "query"
                        ]
                    }
                }
            }
        }
    ]

    return tools

# a function to convert ChatMessages to the Converse API format.
# This format allows us to send a list of current and past messages to Amazon Bedrock for processing

def convert_chat_messages_to_converse_api(chat_messages):
    messages = []
    
    for chat_msg in chat_messages:
        messages.append({
            "role": chat_msg.role,
            "content": [
                {
                    "text": chat_msg.text
                }
            ]
        })
            
    return messages

# a function to handle any tool use requests.
# This format lets us check the model's response to see if the get_amazon_bedrock_information tool was requested. 
# If it was, we will retrieve relevant content from the vector database and submit an additional request to Anthropic Claude to generate a final response based on the retrieved content

def process_tool(response_message, messages, bedrock, tool_list):
    
    messages.append(response_message)
    
    response_content_blocks = response_message['content']

    follow_up_content_blocks = []
    
    for content_block in response_content_blocks:
        if 'toolUse' in content_block:
            tool_use_block = content_block['toolUse']
            
            if tool_use_block['name'] == 'get_amazon_bedrock_information':
                
                collection = get_collection("../../data/chroma", "bedrock_faqs_collection")
                
                query = tool_use_block['input']['query']
                
                print("----QUERY:----")
                print(query)
                
                search_results = get_vector_search_results(collection, query)
    
                flattened_results_list = list(itertools.chain(*search_results['documents'])) #flatten the list of lists returned by chromadb
                
                rag_content = "\n\n".join(flattened_results_list)
                
                print("----RAG CONTENT----")
                print(rag_content)
                
                follow_up_content_blocks.append({
                    "toolResult": {
                        "toolUseId": tool_use_block['toolUseId'],
                        "content": [
                            { "text": rag_content }
                        ]
                    }
                })
                
                
    if len(follow_up_content_blocks) > 0:
        
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
    
        messages.append(follow_up_message)
        
        response = bedrock.converse(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0,
                "topP": 0.9,
                "stopSequences": []
            },
            toolConfig={
                "tools": tool_list
            }
        )
        
    
        return True, response['output']['message']['content'][0]['text'] #tool used, response
        
    else:
        return False, None #tool not used, no response

# a function we can call from the Streamlit front end application.
# This function creates an Amazon Bedrock client with Boto3, then passes the input content to Amazon Bedrock.
# It can then optionally handle a tool use request if necessary

def chat_with_model(message_history, new_text=None):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client
    
    tool_list = get_tools()
    
    new_text_message = ChatMessage('user', text=new_text)
    message_history.append(new_text_message)
    
    number_of_messages = len(message_history)
    
    if number_of_messages > MAX_MESSAGES:
        del message_history[0 : (number_of_messages - MAX_MESSAGES) * 2] #make sure we remove both the user and assistant responses
    
    messages = convert_chat_messages_to_converse_api(message_history)
    
    response = bedrock.converse(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=messages,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
        toolConfig={
            "tools": tool_list
        }
    )
    
    response_message = response['output']['message']
    
    tool_used, output = process_tool(response_message, messages, bedrock, tool_list)
    
    if not tool_used: #just use the original non-RAG result if no tool was needed
        output = response['output']['message']['content'][0]['text']
    
    
    print("----FINAL RESPONSE----")
    print(output)
    
    response_chat_message = ChatMessage('assistant', output)
    
    message_history.append(response_chat_message)
    
    return
