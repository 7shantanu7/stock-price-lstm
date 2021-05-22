import streamlit as st
import awesome_streamlit as ast
import app
import about

ast.core.services.other.set_logging_format()

PAGES = {
    "About" : about,
    "Company Info/Finance": app,
}

def main():
    #Main function of the App
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
        
#hide the streamlit watermark at the bottom
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
