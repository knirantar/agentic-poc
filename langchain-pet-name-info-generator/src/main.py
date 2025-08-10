import pet_name_generator as lch
import streamlit as st

st.title("Pet Name and Story Generator")

animal_type = st.text_input("Enter the type of animal (e.g., unicorn, dragon):")
animal_colour = st.text_input("Enter the colour of the animal (e.g., green, blue):")
if st.button("Generate Pet Info"):
    if animal_type and animal_colour:
        result = lch.generate_pet_info(animal_type, animal_colour)
        st.write(f"🐾 **Pet Name:** {result['pet_name']}")
        st.write(f"📖 **Backstory:** {result['pet_story']}")
        st.write(f"🇪🇸 **Spanish Translation:** {result['pet_story_spanish']}")
    else:
        st.error("Please enter both animal type and colour.")
else:
    st.write("Enter the details and click the button to generate pet information.")