# Streamlit is an open-source Python framework for building front-end applications to demo machine learning applications
# Streamlit provides a library of commands for displaying web elements.
# Streamlit is well suited for creating generative AI prototypes
# You run your Streamlit applications from the command line using the streamlit run command

import streamlit as st #all streamlit commands will be available through the "st" alias


# Add the page title and configuration.
# Here we are setting the page title on the actual page and the title shown in the browser tab.

st.set_page_config(page_title="Streamlit Demo") #HTML title
st.title("Streamlit Demo") #page title

# Add the input elements.
# Create an input text box and button to get a color from the user.

color_text = st.text_input("What's your favorite color?") #display a text box
go_button = st.button("Go", type="primary") #display a primary button

# Add the output elements.
# We use the if block below to handle the button click. 
# We then format the submitted color and display it using Streamlit's write function

if go_button: #code in this if block will be run when the button is clicked

    st.write(f"I like {color_text} too!") #display the response content

# For running the app from the Terminal, use the following script
# streamlit run simple_streamlit_app.py
# in the pop-up displayed, select "Open in Browser"
# Return to the terminal and press Control-C to exit the application.
