# The import statements allow us to use Streamlit elements and call functions in the backing library script

import streamlit as st #all streamlit commands will be available through the "st" alias
import text_lib as glib #reference to local lib script

# Add the page title and configuration
st.set_page_config(page_title="Text to Text") #HTML title
st.title("Text to Text") #page title

# Add the input elements: a multiline text box and button to get the user's prompt and send it to Amazon Bedrock
input_text = st.text_area("Input text", label_visibility="collapsed") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button

# Add the output elements.
# We use the if block below to handle the button click. 
# We display a spinner while the backing function is called, then write the output to the web page
if go_button: #code in this if block will be run when the button is clicked
    
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = glib.get_text_response(input_content=input_text) #call the model through the supporting library
        
        st.write(response_content) #display the response content

# For running the streamlit app: streamlit run text_app.py
# Close the preview tab in the browser. Return to the terminal and press Control-C to exit the application
